#!/usr/bin/env python3
"""
Flux-calibrate LLAMAS datacubes using a standard-star datacube.

This script supports staged execution:
  - extract:  extract standard-star spectra from per-channel cubes
  - sensfunc: generate per-channel sensitivity functions
  - apply:    apply per-channel sensfuncs to science cubes
  - all:      run extract -> sensfunc -> apply in sequence (default)

Implementation notes:
  - White-light and centroiding reuse logic from combine_cubes.py.
  - Centroiding uses a convolved peak on a single cube (no coaddition).
  - Sensfunc generation uses PypeIt SensFunc.get_instance.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from combine_cubes import CubeCombiner
from llamas_pyjamas.Flux import magellan_llamas_spectrograph  # noqa: F401 (registers spectrograph)


LOG = logging.getLogger("fluxcal_datacubes")


CHANNELS = ("blue", "green", "red")
BUNIT_FLUX = "1e-17 erg s-1 cm-2 Angstrom-1"
BUNIT_VAR = "(1e-17 erg s-1 cm-2 Angstrom-1)^2"

DEFAULT_CONFIG_FILENAME = "fluxcal_config.yaml"


@dataclass
class StandardInfo:
    name: str
    ra_deg: float
    dec_deg: float
    source: str


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _channel_arg_map(args, prefix: str) -> Dict[str, Optional[str]]:
    return {
        "blue": getattr(args, f"{prefix}_blue", None),
        "green": getattr(args, f"{prefix}_green", None),
        "red": getattr(args, f"{prefix}_red", None),
    }


# Exposure-time header keys to fall through, measured-actual before
# merely-requested (SEXPTIME/CEXPTIME = shutter-open time; REXPTIME =
# requested time, the only one guaranteed present on LLAMAS data today).
EXPTIME_FALLBACK_KEYS: Tuple[str, ...] = ("SEXPTIME", "CEXPTIME", "DEXPTIME", "EXPTIME", "INTTIME", "REXPTIME")

# Airmass header keys to fall through -- some LLAMAS headers only carry the
# HIERARCH telescope-status family, not a plain AIRMASS card.
AIRMASS_FALLBACK_KEYS: Tuple[str, ...] = ("AIRMASS", "TEL AIRMASS")


def _header_value(header: fits.Header, key: str):
    """`header[key]`, HIERARCH-aware (also tries `HIERARCH <key>`); None if absent."""
    if key in header:
        return header[key]
    hierarch = "HIERARCH " + key
    if hierarch in header:
        return header[hierarch]
    return None


def _read_first_positive(header: fits.Header, keys: Tuple[str, ...], fallback: float, what: str) -> float:
    """First positive, numeric value among `keys` (HIERARCH-aware, in order); `fallback` with a
    loud warning if none are usable."""
    for candidate in dict.fromkeys(keys):  # de-dup while preserving priority order
        val = _header_value(header, candidate)
        if val is None:
            continue
        try:
            out = float(val)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    LOG.warning("No usable %s key found (tried %s); falling back to default %s=%.3f",
                what, list(dict.fromkeys(keys)), what, fallback)
    return float(fallback)


def _read_exptime(header: fits.Header, key: str, fallback: float) -> float:
    """Best-available exposure time (seconds): tries EXPTIME_FALLBACK_KEYS first, in
    measured-over-requested priority order, then the caller's `key` (for a genuinely
    custom key not already in that list), then `fallback`. `key` is deliberately *not*
    tried first -- callers here pass a single nominal key (REXPTIME, or --exptime-key),
    and always preferring a truly-measured value over that would be defeated if the
    nominal key won the race merely by being asked for first."""
    return _read_first_positive(header, (*EXPTIME_FALLBACK_KEYS, key), fallback, "exptime")


def _read_airmass(header: fits.Header, key: str, fallback: float) -> float:
    """Airmass: tries `key` first, then falls through AIRMASS_FALLBACK_KEYS
    (the HIERARCH `TEL AIRMASS` family), then `fallback`."""
    return _read_first_positive(header, (key, *AIRMASS_FALLBACK_KEYS), fallback, "airmass")


def resolve_standard_name(name: str) -> StandardInfo:
    """
    Resolve a standard-star name through PypeIt archive classes.
    """
    from pathlib import Path
    from astropy.table import Table
    from pypeit import dataPaths

    target = "".join(name.upper().split())
    standards_root = Path(dataPaths.standards.path)
    info_files = sorted(standards_root.glob("**/*_info.txt"))

    for info_file in info_files:
        try:
            tab = Table.read(info_file, comment="#", format="ascii")
        except Exception:
            continue

        if "Name" not in tab.colnames:
            continue
        if "RA_2000" not in tab.colnames or "DEC_2000" not in tab.colnames:
            continue

        for row in tab:
            row_name = "".join(str(row["Name"]).upper().split())
            if row_name != target:
                continue

            coord = SkyCoord(str(row["RA_2000"]), str(row["DEC_2000"]), unit=(u.hourangle, u.deg))
            source = info_file.parent.name
            return StandardInfo(
                name=str(row["Name"]),
                ra_deg=float(coord.ra.deg),
                dec_deg=float(coord.dec.deg),
                source=source,
            )

    raise ValueError(
        f"Could not resolve standard star '{name}' in PypeIt archives. "
        "Provide a valid name or use --star-ra/--star-dec overrides."
    )


def find_convolved_peak(
    image_2d: np.ndarray,
    wcs_2d,
    kernel_size_arcsec: float,
    peak_max: Optional[float] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Find the centroid from the convolved white-light peak.
    """
    work = np.asarray(image_2d, dtype=np.float32)
    if peak_max is not None:
        # Keep peak clipping only for sparse hot-pixel outliers. If many pixels are above
        # the threshold, they are likely the stellar core and clipping would bias centroiding.
        high = np.isfinite(work) & (work > peak_max)
        n_high = int(np.count_nonzero(high))
        if 0 < n_high <= 3:
            work = np.where(high, np.nan, work)
        elif n_high > 3:
            LOG.info(
                "Skipping peak_max clipping (%d pixels > %.1f); likely stellar core, not cosmic rays.",
                n_high,
                float(peak_max),
            )

    work, _, _, _, _ = CubeCombiner._mask_background_pixels(work, sigma_multiplier=3.0)

    pix_scales_arcsec = proj_plane_pixel_scales(wcs_2d) * 3600.0
    pix_scale = float(np.nanmean(pix_scales_arcsec))
    sigma_pix = max((kernel_size_arcsec / pix_scale) / 2.355, 0.5)
    kernel = Gaussian2DKernel(x_stddev=sigma_pix)

    conv_img = convolve(
        np.where(np.isfinite(work), work, 0.0),
        kernel,
        boundary="extend",
        normalize_kernel=True,
        nan_treatment="fill",
        preserve_nan=False,
    )

    if not np.any(np.isfinite(conv_img)):
        raise ValueError("No finite values in convolved white-light image.")

    peak_flat = int(np.nanargmax(conv_img))
    cy, cx = np.unravel_index(peak_flat, conv_img.shape)
    return float(cx), float(cy), conv_img


def save_extraction_qa(
    qa_path: Path,
    channel: str,
    image_2d: np.ndarray,
    centroid_xy: Tuple[float, float],
    aperture_radius_pix: float,
    wave: np.ndarray,
    counts_per_s_per_ang: np.ndarray,
    title_prefix: str,
) -> None:
    """
    Save two-panel extraction QA plot:
      left: white-light image + centroid/aperture
      right: extracted spectrum in counts/s/Angstrom
    """
    _ensure_dir(qa_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    finite = np.isfinite(image_2d)
    if np.any(finite):
        vmin = np.nanpercentile(image_2d[finite], 5)
        vmax = np.nanpercentile(image_2d[finite], 99)
    else:
        vmin, vmax = None, None
    im = ax0.imshow(image_2d, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    cx, cy = centroid_xy
    ax0.plot(cx, cy, marker="x", color="red", markersize=10, mew=2, label="Centroid")
    circ = Circle((cx, cy), aperture_radius_pix, edgecolor="cyan", facecolor="none", lw=2, label="Boxcar")
    ax0.add_patch(circ)
    ax0.set_title(f"{title_prefix} {channel}: centroid + aperture")
    ax0.set_xlabel("X pixel")
    ax0.set_ylabel("Y pixel")
    ax0.legend(loc="upper right")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    gpm = np.isfinite(wave) & np.isfinite(counts_per_s_per_ang)
    ax1.plot(wave[gpm], counts_per_s_per_ang[gpm], color="black", lw=1.0)
    ax1.set_title(f"{title_prefix} {channel}: extracted spectrum")
    ax1.set_xlabel("Wavelength (Angstrom)")
    ax1.set_ylabel("Counts / s / Angstrom")

    fig.tight_layout()
    fig.savefig(qa_path)
    plt.close(fig)


def _extract_spectrum_for_qa(spec1d_file: Path, exptime: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read extracted standard spectrum and convert to counts/s/Angstrom for QA.
    """
    from pypeit import io, specobjs, utils

    with io.fits_open(str(spec1d_file)) as hdul:
        dmod = hdul[1].header.get("DMODCLS")

    if dmod == "OneSpec":
        with io.fits_open(str(spec1d_file)) as hdul:
            tab = hdul["SPECTRUM"].data
            w = np.asarray(tab["wave"], dtype=float)
            c = np.asarray(tab["flux"], dtype=float)
            if "mask" in tab.names:
                m = np.asarray(tab["mask"]) > 0
            else:
                m = np.ones_like(w, dtype=bool)
    else:
        sobjs = specobjs.SpecObjs.from_fitsfile(str(spec1d_file))
        try:
            std = sobjs.get_std(multi_spec_det=None, split_mosaic=True)
        except TypeError:
            std = sobjs.get_std(multi_spec_det=None)
        wave, counts, _, mask, _, _, _ = std.unpack_object(ret_flam=False, extract_type="BOX", remove_missing=True)
        ivar = np.ones_like(counts, dtype=float)
        wave2, counts2, _, mask2, _, _, _ = utils.spec_atleast_2d(wave, counts, ivar, mask)
        w = wave2[:, 0]
        c = counts2[:, 0]
        m = mask2[:, 0] if mask2 is not None else np.ones_like(w, dtype=bool)

    delta = np.gradient(w)
    cpsa = np.zeros_like(c, dtype=float)
    good = m & np.isfinite(c) & np.isfinite(delta) & (delta > 0)
    cpsa[good] = c[good] / exptime / delta[good]
    cpsa[~good] = np.nan
    return w, cpsa


def run_extract(args, paths: Dict[str, Path]) -> Dict[str, Path]:
    """
    Extract standard spectra for each channel from RSS fiber data and write
    extraction QA products.

    The standard-star cube is used only to centroid the star (via the same
    convolved white-light peak used elsewhere); the actual flux is summed
    from native RSS fiber spectra within a boxcar aperture of the centroid,
    not from resampled cube pixels (cube pixels are correlated/smeared by
    the IDW spatial interpolation and are not flux-conserving).
    """
    from pypeit import specobj, specobjs
    from llamas_pyjamas.Cube.simple_cube_constructor import SimpleCubeConstructor

    std_cubes = _channel_arg_map(args, "std_cube")
    missing_cubes = [ch for ch in CHANNELS if not std_cubes.get(ch)]
    if missing_cubes:
        raise ValueError(f"Missing standard cube inputs for channels: {', '.join(missing_cubes)}")

    std_rss = _channel_arg_map(args, "std_rss")
    missing_rss = [ch for ch in CHANNELS if not std_rss.get(ch)]
    if missing_rss:
        raise ValueError(f"Missing standard RSS inputs for channels: {', '.join(missing_rss)}")

    comb = CubeCombiner(input_folder=".")
    spec1d_paths: Dict[str, Path] = {}

    for ch in CHANNELS:
        cube_file = Path(std_cubes[ch]).expanduser().resolve()
        if not cube_file.exists():
            raise FileNotFoundError(f"Standard cube not found: {cube_file}")

        rss_file = Path(std_rss[ch]).expanduser().resolve()
        if not rss_file.exists():
            raise FileNotFoundError(f"Standard RSS not found: {rss_file}")

        with fits.open(cube_file) as hdul:
            wave_min, wave_max = comb.DEFAULT_COLLAPSE_RANGES[ch]
            wl_img, wcs_2d = comb.collapse_cube_to_whitelight(
                hdul, wave_min_angstrom=wave_min, wave_max_angstrom=wave_max
            )

            peak_max = comb.RED_PEAK_MAX if ch == "red" else None
            cx, cy, _ = find_convolved_peak(
                wl_img,
                wcs_2d,
                kernel_size_arcsec=args.centroid_kernel_arcsec,
                peak_max=peak_max,
            )

            # Read the exact grid-construction parameters used to build this
            # cube from its own header, so the spatial grid we rebuild below
            # from the sibling RSS file is pixel-for-pixel identical to the
            # one the centroid (cx, cy) was measured in. (Cube CRPIX/CDELT
            # cannot be used to convert (cx, cy) to arcsec directly: verified
            # empirically that the cube WCS's world origin does not coincide
            # with the fiber-coordinate origin, and its CDELT differs
            # slightly from the true grid step due to linspace rounding.)
            pixel_size = float(hdul[0].header.get("PIXSIZE", 0.3))
            fiber_pitch = float(hdul[0].header.get("FIBPITCH", 0.75))
            grid_method = str(hdul[0].header.get("GRIDMETH", "oversampled"))
            cube_ny, cube_nx = wl_img.shape

        rcc = SimpleCubeConstructor(
            fiber_pitch=fiber_pitch, pixel_size=pixel_size, grid_method=grid_method
        )
        rcc.load_rss_file(str(rss_file))
        rcc.match_fibers_to_fibermap()
        rcc.create_wavelength_grid()
        rcc.create_spatial_grid()

        if len(rcc.x_grid) != cube_nx or len(rcc.y_grid) != cube_ny:
            raise RuntimeError(
                f"Channel {ch}: spatial grid rebuilt from {rss_file.name} "
                f"({len(rcc.x_grid)}x{len(rcc.y_grid)}) does not match cube "
                f"{cube_file.name} ({cube_nx}x{cube_ny}); they are likely not "
                "from the same exposure/construction."
            )

        ix = int(np.clip(round(cx), 0, len(rcc.x_grid) - 1))
        iy = int(np.clip(round(cy), 0, len(rcc.y_grid) - 1))
        centroid_arcsec = np.array([rcc.x_grid[ix], rcc.y_grid[iy]])

        distances = np.linalg.norm(rcc.fiber_coords - centroid_arcsec, axis=1)
        in_aperture = np.where(distances <= args.boxcar_radius_arcsec)[0]
        if len(in_aperture) == 0:
            raise RuntimeError(
                f"Channel {ch}: no fibers within {args.boxcar_radius_arcsec} arcsec "
                f"of centroid ({centroid_arcsec[0]:.3f}, {centroid_arcsec[1]:.3f}) arcsec."
            )

        LOG.info(
            "Channel %s: summing %d fibers within %.2f arcsec of centroid (%.3f, %.3f) arcsec",
            ch, len(in_aperture), args.boxcar_radius_arcsec, centroid_arcsec[0], centroid_arcsec[1],
        )

        # Fiber flux is each fiber's own total collected counts (same BUNIT
        # as the cube), so no per-pixel area/normalization factor is needed
        # -- unlike summing spaxels out of the IDW-resampled cube.
        nwave = len(rcc.wave_grid)
        flux_sum = np.zeros(nwave, dtype=float)
        var_sum = np.zeros(nwave, dtype=float)
        n_good = np.zeros(nwave, dtype=int)
        have_var = rcc.error is not None

        for local_idx in in_aperture:
            rss_idx = rcc.fiber_indices[local_idx]
            flux_resamp, var_resamp = rcc.resample_fiber_to_grid(rss_idx)
            if flux_resamp is None:
                continue
            good = np.isfinite(flux_resamp)
            flux_sum[good] += flux_resamp[good]
            n_good[good] += 1
            if have_var and var_resamp is not None:
                good_var = good & np.isfinite(var_resamp)
                var_sum[good_var] += var_resamp[good_var]

        counts = np.where(n_good > 0, flux_sum, np.nan)
        if have_var:
            ivar_1d = np.zeros(nwave, dtype=float)
            good_var = np.isfinite(var_sum) & (var_sum > 0) & (n_good > 0)
            ivar_1d[good_var] = 1.0 / var_sum[good_var]
        else:
            ivar_1d = np.ones(nwave, dtype=float)

        good_ivar = np.isfinite(ivar_1d) & (ivar_1d > 0)
        if np.count_nonzero(good_ivar) < max(10, int(0.1 * ivar_1d.size)):
            LOG.warning(
                "Channel %s has too few positive ivar samples in RSS-based extraction; "
                "using unit ivar for robustness.",
                ch,
            )
            ivar_1d = np.ones_like(counts, dtype=float)

        wave = np.asarray(rcc.wave_grid, dtype=float)
        counts = np.nan_to_num(counts, nan=0.0)
        good = np.isfinite(counts) & np.isfinite(ivar_1d) & (ivar_1d > 0)
        mask_1d = np.zeros_like(counts, dtype=int)
        mask_1d[good] = 1

        # Read exptime/airmass directly off the RSS file's own primary
        # header (REXPTIME/AIRMASS) -- this works immediately, independent
        # of the cube header ever carrying EXPTIME/AIRMASS.
        exptime = _read_exptime(rcc.primary_header, "REXPTIME", args.default_exptime)
        airmass = _read_airmass(rcc.primary_header, args.airmass_key, args.default_airmass)

        out_spec = paths["extractions"] / f"spec1d_std_{ch}.fits"

        sobj = specobj.SpecObj.from_arrays(
            "SlicerIFU",
            wave,
            np.asarray(counts, dtype=float),
            np.asarray(ivar_1d, dtype=float),
            mode="BOX",
            DET="DET01",
            SLITID=0,
        )
        sobj.PYP_SPEC = args.pyp_spec
        if np.isfinite(cx) and np.isfinite(cy):
            sky = wcs_2d.pixel_to_world(float(cx), float(cy))
            sobj.RA = float(sky.ra.deg)
            sobj.DEC = float(sky.dec.deg)

        sobj_mask = np.asarray(mask_1d, dtype=bool)
        sobj.BOX_MASK = sobj_mask
        sobj.BOX_COUNTS_IVAR = np.asarray(ivar_1d, dtype=float)
        sobj.BOX_COUNTS = np.asarray(counts, dtype=float)
        sobj.BOX_WAVE = np.asarray(wave, dtype=float)

        subheader = {
            "PYP_SPEC": args.pyp_spec,
            "DISPNAME": ch.upper(),
            "EXPTIME": exptime,
            "AIRMASS": airmass,
            "PYPELINE": "SlicerIFU",
            "RA": float(sobj.RA) if getattr(sobj, "RA", None) is not None else 0.0,
            "DEC": float(sobj.DEC) if getattr(sobj, "DEC", None) is not None else 0.0,
        }
        sobjs = specobjs.SpecObjs(specobjs=np.array([sobj]), header=subheader)
        sobjs.write_to_fits(subheader, str(out_spec), overwrite=args.overwrite)

        spec1d_paths[ch] = out_spec

        pix_scales = proj_plane_pixel_scales(wcs_2d) * 3600.0
        pix_scale = float(np.nanmean(pix_scales))
        aperture_pix = args.boxcar_radius_arcsec / pix_scale

        wave_qa, cpsa = _extract_spectrum_for_qa(out_spec, exptime=exptime)
        qa_file = paths["qa"] / f"std_extract_{ch}_qa.pdf"
        save_extraction_qa(
            qa_file,
            ch,
            wl_img,
            (cx, cy),
            aperture_pix,
            wave_qa,
            cpsa,
            title_prefix="Standard",
        )

        LOG.info("Extracted standard %s -> %s", ch, out_spec)
        LOG.info("Wrote extraction QA -> %s", qa_file)

    return spec1d_paths


def _read_spec1d_wave_mask(spec1d_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (wave, good-pixel-mask) for a spec1d file's standard-star object,
    regardless of OneSpec/SpecObjs format. Uses the same `extract_type="BOX"`,
    `remove_missing=True` unpacking SensFunc.__init__ itself uses, so the
    returned `wave` array has the same length/order PypeIt will actually fit
    against (this matters for _compute_trim_std_pixs below).
    """
    from pypeit import io, specobjs

    with io.fits_open(str(spec1d_file)) as hdul:
        dmod = hdul[1].header.get("DMODCLS")

    if dmod == "OneSpec":
        with io.fits_open(str(spec1d_file)) as hdul:
            tab = hdul["SPECTRUM"].data
            w = np.asarray(tab["wave"], dtype=float)
            if "mask" in tab.names:
                m = np.asarray(tab["mask"]) > 0
            else:
                m = np.isfinite(w)
    else:
        sobjs = specobjs.SpecObjs.from_fitsfile(str(spec1d_file))
        try:
            std = sobjs.get_std(multi_spec_det=None, split_mosaic=True)
        except TypeError:
            std = sobjs.get_std(multi_spec_det=None)
        wave, _, _, mask, _, _, _ = std.unpack_object(ret_flam=False, extract_type="BOX", remove_missing=True)
        w = np.asarray(wave)
        m = np.asarray(mask) if mask is not None else np.isfinite(w)

    return w, m


def _auto_algorithm(spec1d_file: Path) -> str:
    w, m = _read_spec1d_wave_mask(spec1d_file)
    good = np.isfinite(w) & m
    med = np.nanmedian(w[good]) if np.any(good) else np.nanmedian(w)
    return "IR" if med > 7000.0 else "UVIS"


def _compute_trim_std_pixs(
    spec1d_file: Path, wave_min: Optional[float], wave_max: Optional[float]
) -> Optional[Tuple[int, int]]:
    """
    Convert a desired [wave_min, wave_max] sensfunc-fit window into a
    (blue, red) `trim_std_pixs` pixel count against the standard's own
    extracted wavelength array (see `pypeit.par.pypeitpar.SensFuncPar`'s
    `trim_std_pixs`, applied in `Spectrograph.tweak_standard` before either
    the UVIS or IR fit runs -- this is the only wavelength-range restriction
    UVIS exposes; IR also has `IR.fit_wv_min_max`, but trim_std_pixs works
    identically for both, so it's used uniformly here).

    Either `wave_min` or `wave_max` may be None to leave that side untrimmed.
    Returns None (meaning: don't set trim_std_pixs at all) if both are None
    or the spec1d has no usable wavelength data.
    """
    if wave_min is None and wave_max is None:
        return None

    w, _ = _read_spec1d_wave_mask(spec1d_file)
    if w.size == 0:
        return None

    n = len(w)
    trim_blue = int(np.searchsorted(w, wave_min, side="left")) if wave_min is not None else 0
    trim_red = int(n - np.searchsorted(w, wave_max, side="right")) if wave_max is not None else 0
    trim_blue = max(0, min(trim_blue, n - 1))
    # PypeIt's tweak_standard() does `trim_gpm[s:-e] = True`; when e == 0,
    # `-e` is `-0 == 0`, so `[s:0]` is always empty and the *entire* spectrum
    # would be zeroed out, not left untrimmed. Floor at 1 to avoid that.
    trim_red = max(1, min(trim_red, n - 1 - trim_blue))
    return (trim_blue, trim_red)


def run_sensfunc(args, paths: Dict[str, Path], extracted: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
    """
    Build per-channel sensfunc files using PypeIt SensFunc.
    """
    from pypeit import io
    from pypeit.par import pypeitpar
    from pypeit import sensfunc as sensfunc_mod
    from pypeit.spectrographs.util import load_spectrograph

    if args.star_ra is not None and args.star_dec is not None:
        std_info = StandardInfo(
            name=args.standard_star_name or "manual_coordinates",
            ra_deg=float(args.star_ra),
            dec_deg=float(args.star_dec),
            source="manual",
        )
    else:
        if not args.standard_star_name:
            raise ValueError("--standard-star-name is required for sensfunc unless --star-ra/--star-dec are set.")
        std_info = resolve_standard_name(args.standard_star_name)

    LOG.info(
        "Using standard '%s' from %s at RA=%.6f DEC=%.6f",
        std_info.name,
        std_info.source,
        std_info.ra_deg,
        std_info.dec_deg,
    )

    spec1d_args = _channel_arg_map(args, "std_spec1d")
    sens_paths: Dict[str, Path] = {}

    for ch in CHANNELS:
        spec1d_file = None
        if extracted is not None and ch in extracted:
            spec1d_file = extracted[ch]
        elif spec1d_args.get(ch):
            spec1d_file = Path(spec1d_args[ch]).expanduser().resolve()
        else:
            candidate = paths["extractions"] / f"spec1d_std_{ch}.fits"
            if candidate.exists():
                spec1d_file = candidate

        if spec1d_file is None or not spec1d_file.exists():
            raise FileNotFoundError(f"Missing extracted standard spec1d for channel {ch}")

        with io.fits_open(str(spec1d_file)) as hdul:
            try:
                spectrograph = load_spectrograph(hdul[0].header["PYP_SPEC"], pypeit_fits=True)
            except TypeError:
                spectrograph = load_spectrograph(hdul[0].header["PYP_SPEC"])
            cfg_par = spectrograph.config_specific_par(hdul)
            primary_hdr = io.initialize_header()
            add_keys = (
                ["PYP_SPEC", "DATE-OBS", "TELESCOP", "INSTRUME", "DETECTOR"]
                + spectrograph.configuration_keys()
                + spectrograph.raw_header_cards()
            )
            for key in add_keys:
                if key.upper() in hdul[0].header:
                    primary_hdr[key.upper()] = hdul[0].header[key.upper()]

        par = pypeitpar.PypeItPar.from_cfg_lines(cfg_lines=cfg_par.to_config())
        par["sensfunc"]["star_ra"] = std_info.ra_deg
        par["sensfunc"]["star_dec"] = std_info.dec_deg
        par["sensfunc"]["extr"] = "BOX"
        # LLAMAS optical standards include hot subdwarfs/white dwarfs (e.g. GD108,
        # Feige110) with pressure-broadened Balmer/He lines tens of Angstroms wide --
        # PypeIt's defaults (hydrogen_mask_wid=10, mask_helium_lines=False) don't mask
        # far enough into the wings and skip He entirely, letting those absorption
        # lines drag the zeropoint fit instead of just the instrument response.
        # 35 Angstrom half-width covers rs-dev's own per-line widths (20-45 Angstrom
        # for Balmer, 15-20 Angstrom for He); mask_helium_lines scales off the same
        # knob (half of hydrogen_mask_wid), landing at 17.5 Angstrom for He.
        par["sensfunc"]["mask_helium_lines"] = True
        par["sensfunc"]["hydrogen_mask_wid"] = 35.0
        par["sensfunc"]["algorithm"] = _auto_algorithm(spec1d_file) if args.algorithm is None else args.algorithm

        polyorder = _channel_arg_map(args, "polyorder").get(ch)
        if polyorder is not None:
            par["sensfunc"]["polyorder"] = int(polyorder)

        fit_wave_min = _channel_arg_map(args, "fit_wave_min").get(ch)
        fit_wave_max = _channel_arg_map(args, "fit_wave_max").get(ch)
        trim_std_pixs = _compute_trim_std_pixs(spec1d_file, fit_wave_min, fit_wave_max)
        if trim_std_pixs is not None:
            par["sensfunc"]["trim_std_pixs"] = trim_std_pixs
            LOG.info(
                "Channel %s: restricting sensfunc fit to %s-%s Angstrom (trim_std_pixs=%s)",
                ch,
                fit_wave_min if fit_wave_min is not None else "start",
                fit_wave_max if fit_wave_max is not None else "end",
                trim_std_pixs,
            )

        out_sens = paths["sensfunc"] / f"sens_std_{ch}.fits"
        try:
            sensobj = sensfunc_mod.SensFunc.get_instance(
                [str(spec1d_file)],
                str(out_sens),
                par["sensfunc"],
                par_fluxcalib=par["fluxcalib"],
                debug=args.debug,
                chk_version=par["rdx"]["chk_version"],
            )
        except TypeError:
            sensobj = sensfunc_mod.SensFunc.get_instance(
                str(spec1d_file),
                str(out_sens),
                par["sensfunc"],
                par_fluxcalib=par["fluxcalib"],
                debug=args.debug,
                chk_version=par["rdx"]["chk_version"],
            )
        sensobj.run()
        sensobj.to_file(str(out_sens), primary_hdr=primary_hdr, overwrite=args.overwrite)
        sens_paths[ch] = out_sens

        LOG.info("Generated sensfunc %s -> %s", ch, out_sens)

    return sens_paths


def _choose_sens_vectors(sens_obj) -> Tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(sens_obj.wave)
    zp = np.asarray(sens_obj.zeropoint)

    if wave.ndim == 1:
        return wave, zp

    best_col = 0
    best_n = -1
    for i in range(wave.shape[1]):
        n = int(np.sum(np.isfinite(wave[:, i]) & (wave[:, i] > 0) & np.isfinite(zp[:, i])))
        if n > best_n:
            best_n = n
            best_col = i
    return wave[:, best_col], zp[:, best_col]


def _calibrate_cube_file(
    cube_file: Path,
    out_file: Path,
    sens_obj,
    exptime_key: str,
    default_exptime: float,
    airmass_key: str,
    default_airmass: float,
    overwrite: bool,
) -> None:
    from pypeit.core import flux_calib
    from pypeit.spectrographs.util import load_spectrograph

    # SensFunc.from_file() restores the PYP_SPEC string (it's part of the
    # datamodel) but not the live spectrograph instance -- that's only ever
    # built in SensFunc.__init__, which from_file()/from_hdu() bypass
    # (confirmed: sens_obj.spectrograph is None right after from_file()).
    # Rebuild it here; it carries the telescope longitude/latitude needed
    # for the extinction correction below.
    spectrograph = load_spectrograph(sens_obj.PYP_SPEC)
    extinctfilepar = spectrograph.default_pypeit_par()["sensfunc"]["UVIS"]["extinct_file"]

    with fits.open(cube_file) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float64)
        var = None
        if len(hdul) > 1 and hdul[1].data is not None:
            var = np.asarray(hdul[1].data, dtype=np.float64)

        wave = CubeCombiner._extract_wavelength_axis(hdul)
        exptime = _read_exptime(hdul[0].header, exptime_key, default_exptime)
        airmass = _read_airmass(hdul[0].header, airmass_key, default_airmass)

        wave_sens, zp_sens = _choose_sens_vectors(sens_obj)
        # extinct_correct=True applies an airmass-specific atmospheric
        # extinction correction for *this* science exposure -- the
        # standard's own extinction correction (baked into the zeropoint
        # during sensfunc fitting) is specific to the standard's airmass,
        # not the science target's. Mirrors PypeIt's own reference usage in
        # pypeit/specobj.py and pypeit/coadd3d.py.
        factor = flux_calib.get_sensfunc_factor(
            wave,
            wave_sens,
            zp_sens,
            exptime,
            extinct_correct=True,
            airmass=airmass,
            longitude=spectrograph.telescope["longitude"],
            latitude=spectrograph.telescope["latitude"],
            extinctfilepar=extinctfilepar,
            extrap_sens=True,
        )
        factor3d = factor[:, None, None]

        hdul[0].data = (flux * factor3d).astype(np.float32)
        hdul[0].header["BUNIT"] = BUNIT_FLUX
        hdul[0].header["FLXCAL"] = (True, "Flux calibrated by fluxcal_datacubes.py")
        hdul[0].header["SCEXPT"] = (float(exptime), f"Science exposure time from {exptime_key}")
        hdul[0].header["SCAIRM"] = (float(airmass), f"Science airmass from {airmass_key}")

        if var is not None:
            hdul[1].data = (var * (factor3d ** 2)).astype(np.float32)
            hdul[1].header["BUNIT"] = BUNIT_VAR

        _ensure_dir(out_file.parent)
        hdul.writeto(out_file, overwrite=overwrite)


def run_apply(args, paths: Dict[str, Path], sens_map: Optional[Dict[str, Path]] = None) -> None:
    from pypeit import sensfunc as sensfunc_mod

    sens_args = _channel_arg_map(args, "sens")
    sci_globs = _channel_arg_map(args, "science_glob")

    for ch in CHANNELS:
        sens_path = None
        if sens_map is not None and ch in sens_map:
            sens_path = sens_map[ch]
        elif sens_args.get(ch):
            sens_path = Path(sens_args[ch]).expanduser().resolve()
        else:
            candidate = paths["sensfunc"] / f"sens_std_{ch}.fits"
            if candidate.exists():
                sens_path = candidate

        if sens_path is None or not sens_path.exists():
            raise FileNotFoundError(f"Missing sensfunc for channel {ch}")

        if not sci_globs.get(ch):
            raise ValueError(f"Missing --science-glob-{ch} for apply step")
        cube_files = sorted(glob.glob(sci_globs[ch]))
        if not cube_files:
            LOG.warning("No science cubes matched for %s with pattern: %s", ch, sci_globs[ch])
            continue

        sens_obj = sensfunc_mod.SensFunc.from_file(str(sens_path), chk_version=True)
        for cube in cube_files:
            cube_path = Path(cube).expanduser().resolve()
            out_name = cube_path.stem + "_fluxed.fits"
            out_path = paths["cubes"] / out_name
            _calibrate_cube_file(
                cube_path,
                out_path,
                sens_obj,
                exptime_key=args.exptime_key,
                default_exptime=args.default_exptime,
                airmass_key=args.airmass_key,
                default_airmass=args.default_airmass,
                overwrite=args.overwrite,
            )
            LOG.info("Flux calibrated %s -> %s", cube_path.name, out_path)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show argument defaults and keep the epilog's example commands pre-formatted."""


_EXAMPLE_EPILOG = f"""
Recommended usage: a config file instead of a long flag list.

  Running with -h/--help (as you just did, if you're reading this) writes a
  ready-to-run template, pre-filled with a real example for this dataset's
  HD14943 standard, to ./{DEFAULT_CONFIG_FILENAME} if that file doesn't
  already exist. Edit it, then run:

      python fluxcal_cubes.py --config {DEFAULT_CONFIG_FILENAME}

  Any flag also given on the command line overrides the config file's value
  for that one flag, e.g.:

      python fluxcal_cubes.py --config {DEFAULT_CONFIG_FILENAME} --mode apply --overwrite

Direct CLI example (no config file), for one-off overrides or scripting:

  python fluxcal_cubes.py --mode all \\
      --standard-star-name HD14943 \\
      --std-cube-blue  cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_blue.fits \\
      --std-cube-green cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_green.fits \\
      --std-cube-red   cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_red.fits \\
      --std-rss-blue   extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_blue.fits \\
      --std-rss-green  extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_green.fits \\
      --std-rss-red    extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_red.fits \\
      --science-glob-blue  "cubes/FRB20240210A/*_cube_blue.fits" \\
      --science-glob-green "cubes/FRB20240210A/*_cube_green.fits" \\
      --science-glob-red   "cubes/FRB20240210A/*_cube_red.fits" \\
      --output-root fluxcal_HD14943 --overwrite --verbose
"""


# Real, working example for this dataset (see redux/cubes/HD14943 and its
# sibling RSS files in redux/extractions/) -- edit paths/values for your own
# standard and science target, or override individual fields on the CLI.
_CONFIG_TEMPLATE = f"""\
# fluxcal_cubes.py configuration file.
#
# Run with:  python fluxcal_cubes.py --config {DEFAULT_CONFIG_FILENAME}
# Any flag also passed on the command line overrides the corresponding value here.
# Leave a field blank (or delete its line) to fall back to the script's built-in default.

# Pipeline stage: extract | sensfunc | apply | all
mode: all

# --- Standard star identity (sensfunc stage) ---
# Name resolved against PypeIt's standard-star archive. Leave blank and set
# star_ra/star_dec instead if the name doesn't resolve there.
standard_star_name: HD14943
star_ra:
star_dec:

# Force a sensfunc algorithm (UVIS or IR) for every channel instead of
# auto-selecting per channel from wavelength (IR above 7000 Angstrom).
algorithm:

# PypeIt PYP_SPEC spectrograph name.
pyp_spec: magellan_llamas

# --- Sensfunc zeropoint fit tuning (per channel) ---
# Polynomial order for the zeropoint fit. Red defaults higher than PypeIt's
# own default (5) because the O2/H2O telluric-band structure there needs
# more flexibility; set to 5 to restore PypeIt's default.
polyorder_blue:
polyorder_green:
polyorder_red: 7

# Restrict the fit to a wavelength window (Angstrom), excluding the steep
# low-throughput roll-off at the channel edges that otherwise drags a
# whole-range polynomial fit off the well-behaved plateau. Red defaults to
# 6700-10000 based on this dataset's QA plots; blue/green are unrestricted
# by default since no equivalent edge problem has been seen there yet.
fit_wave_min_blue:
fit_wave_max_blue:
fit_wave_min_green:
fit_wave_max_green:
fit_wave_min_red: 6700.0
fit_wave_max_red: 10000.0

# --- Standard star cube + RSS pair (extract stage) ---
# The cube is used only to centroid the star; flux is actually summed from
# the sibling RSS file's native fiber spectra.
std_cube_blue: cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_blue.fits
std_cube_green: cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_green.fits
std_cube_red: cubes/HD14943/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_cube_red.fits

std_rss_blue: extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_blue.fits
std_rss_green: extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_green.fits
std_rss_red: extractions/LLAMAS_2025-12-17_05-21-17.9_SCI22_mef_flat_corrected_extract_RSS_red.fits

# --- Pre-built inputs, to skip earlier stages (leave blank for a fresh --mode all run) ---
# Pre-extracted standard spec1d files (skip the extract stage).
std_spec1d_blue:
std_spec1d_green:
std_spec1d_red:

# Pre-built sensfunc files (skip the extract+sensfunc stages).
sens_blue:
sens_green:
sens_red:

# --- Science cubes to flux-calibrate (apply stage) ---
science_glob_blue: "cubes/FRB20240210A/*_cube_blue.fits"
science_glob_green: "cubes/FRB20240210A/*_cube_green.fits"
science_glob_red: "cubes/FRB20240210A/*_cube_red.fits"

# --- Extraction tuning ---
boxcar_radius_arcsec: 1.0
centroid_kernel_arcsec: 1.0

# --- Science cube header keys (apply stage only; extraction always reads
# the standard RSS file's own REXPTIME/AIRMASS regardless of these) ---
exptime_key: EXPTIME
default_exptime: 1.0
airmass_key: AIRMASS
default_airmass: 1.0

# --- Output / run behavior ---
output_root: fluxcal_HD14943
overwrite: true
verbose: true
debug: false
"""


def _write_default_config_if_missing(path: Path) -> bool:
    """Write the config template to `path` unless a file is already there."""
    if path.exists():
        return False
    path.write_text(_CONFIG_TEMPLATE)
    return True


def _load_config_defaults(config_path: Path, valid_dests: set) -> Dict[str, object]:
    """
    Load a YAML config file into a dict of argparse defaults.

    Keys with a null/blank value (falls back to the script's own default) and
    unrecognized keys (typos, stale fields) are dropped, with a warning
    logged for the latter so a mistyped key doesn't silently do nothing.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    unknown = set(data.keys()) - valid_dests
    if unknown:
        LOG.warning(
            "Ignoring unrecognized key(s) in %s: %s", config_path, ", ".join(sorted(unknown))
        )

    return {k: v for k, v in data.items() if k in valid_dests and v is not None}


def parse_args() -> argparse.Namespace:
    # Generating the template here (rather than in main()) means it's written
    # for -h/--help too, since argparse's own help handling exits the process
    # before main() would otherwise get a chance to.
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        template_path = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if _write_default_config_if_missing(template_path):
            print(f"Wrote a default config template to {template_path} -- edit it and run with --config {DEFAULT_CONFIG_FILENAME}\n")

    p = argparse.ArgumentParser(
        description="Flux calibration pipeline for LLAMAS datacubes.",
        formatter_class=_HelpFormatter,
        epilog=_EXAMPLE_EPILOG,
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            f"Path to a YAML config file providing pipeline arguments (see the "
            f"auto-generated {DEFAULT_CONFIG_FILENAME} template, written on -h/--help). "
            "Config values are used as defaults; any flag also given on the command "
            "line overrides the corresponding config value."
        ),
    )

    p.add_argument(
        "--mode",
        choices=["all", "extract", "sensfunc", "apply"],
        default="all",
        help=(
            "Pipeline stage to run: 'extract' pulls the standard-star spectrum from its "
            "RSS/cube pair, 'sensfunc' builds a PypeIt sensitivity function from the "
            "extracted spectrum, 'apply' flux-calibrates science cubes with a sensfunc, "
            "'all' runs extract -> sensfunc -> apply in sequence."
        ),
    )

    p.add_argument(
        "--standard-star-name",
        type=str,
        default=None,
        help=(
            "Name of the spectrophotometric standard (e.g. 'HD074000'), resolved against "
            "PypeIt's archive of standard-star coordinate/flux tables. Required for the "
            "sensfunc stage unless --star-ra/--star-dec are given instead."
        ),
    )
    p.add_argument(
        "--star-ra",
        type=float,
        default=None,
        help="Standard star RA in decimal degrees. Overrides --standard-star-name lookup if set together with --star-dec.",
    )
    p.add_argument(
        "--star-dec",
        type=float,
        default=None,
        help="Standard star Dec in decimal degrees. Overrides --standard-star-name lookup if set together with --star-ra.",
    )
    p.add_argument(
        "--algorithm",
        choices=["UVIS", "IR"],
        default=None,
        help=(
            "PypeIt sensfunc algorithm to force for every channel. If not set, it is "
            "auto-selected per channel from the extracted spectrum's median wavelength "
            "(IR above 7000 Angstrom, i.e. normally the red channel; UVIS otherwise)."
        ),
    )
    p.add_argument(
        "--pyp-spec",
        type=str,
        default="magellan_llamas",
        help="PypeIt PYP_SPEC spectrograph name written into spec1d/sensfunc headers and used to look up the Spectrograph class.",
    )

    p.add_argument(
        "--polyorder-blue",
        type=int,
        default=None,
        help="Polynomial order for the blue-channel sensfunc zeropoint fit (PypeIt default: 5, used if unset).",
    )
    p.add_argument(
        "--polyorder-green",
        type=int,
        default=None,
        help="Polynomial order for the green-channel sensfunc zeropoint fit (PypeIt default: 5, used if unset).",
    )
    p.add_argument(
        "--polyorder-red",
        type=int,
        default=7,
        help=(
            "Polynomial order for the red-channel sensfunc zeropoint fit. Defaults to 7 "
            "(higher than PypeIt's own default of 5) because the red channel's O2/H2O "
            "telluric-band structure needs more flexibility to fit well; set to 5 to "
            "restore PypeIt's default."
        ),
    )

    p.add_argument(
        "--fit-wave-min-blue",
        type=float,
        default=None,
        help="Restrict the blue-channel sensfunc fit to wavelengths >= this (Angstrom); unset trims nothing off the blue edge.",
    )
    p.add_argument(
        "--fit-wave-max-blue",
        type=float,
        default=None,
        help="Restrict the blue-channel sensfunc fit to wavelengths <= this (Angstrom); unset trims nothing off the red edge.",
    )
    p.add_argument(
        "--fit-wave-min-green",
        type=float,
        default=None,
        help="Restrict the green-channel sensfunc fit to wavelengths >= this (Angstrom); unset trims nothing off the blue edge.",
    )
    p.add_argument(
        "--fit-wave-max-green",
        type=float,
        default=None,
        help="Restrict the green-channel sensfunc fit to wavelengths <= this (Angstrom); unset trims nothing off the red edge.",
    )
    p.add_argument(
        "--fit-wave-min-red",
        type=float,
        default=6700.0,
        help=(
            "Restrict the red-channel sensfunc fit to wavelengths >= this (Angstrom). "
            "Defaults to 6700 to exclude the steep low-throughput roll-off below it, which "
            "otherwise drags the whole-range polynomial fit off the well-behaved plateau; "
            "set to null/None (via config) or override with a lower value to disable."
        ),
    )
    p.add_argument(
        "--fit-wave-max-red",
        type=float,
        default=10000.0,
        help=(
            "Restrict the red-channel sensfunc fit to wavelengths <= this (Angstrom). "
            "Defaults to 10000 to exclude the steep low-throughput roll-off above it, for "
            "the same reason as --fit-wave-min-red."
        ),
    )

    p.add_argument("--std-cube-blue", type=str, default=None, help="Standard-star blue-channel datacube (used only to centroid the star).")
    p.add_argument("--std-cube-green", type=str, default=None, help="Standard-star green-channel datacube (used only to centroid the star).")
    p.add_argument("--std-cube-red", type=str, default=None, help="Standard-star red-channel datacube (used only to centroid the star).")

    p.add_argument(
        "--std-rss-blue",
        type=str,
        default=None,
        help="Standard-star blue-channel RSS file (sibling of --std-cube-blue) that the fiber flux is actually summed from.",
    )
    p.add_argument(
        "--std-rss-green",
        type=str,
        default=None,
        help="Standard-star green-channel RSS file (sibling of --std-cube-green) that the fiber flux is actually summed from.",
    )
    p.add_argument(
        "--std-rss-red",
        type=str,
        default=None,
        help="Standard-star red-channel RSS file (sibling of --std-cube-red) that the fiber flux is actually summed from.",
    )

    p.add_argument(
        "--std-spec1d-blue",
        type=str,
        default=None,
        help="Pre-extracted blue-channel standard-star spec1d file; use instead of --std-cube-blue/--std-rss-blue to skip the extract stage.",
    )
    p.add_argument(
        "--std-spec1d-green",
        type=str,
        default=None,
        help="Pre-extracted green-channel standard-star spec1d file; use instead of --std-cube-green/--std-rss-green to skip the extract stage.",
    )
    p.add_argument(
        "--std-spec1d-red",
        type=str,
        default=None,
        help="Pre-extracted red-channel standard-star spec1d file; use instead of --std-cube-red/--std-rss-red to skip the extract stage.",
    )

    p.add_argument("--sens-blue", type=str, default=None, help="Pre-built blue-channel sensfunc FITS file; use to skip the sensfunc stage when running apply on its own.")
    p.add_argument("--sens-green", type=str, default=None, help="Pre-built green-channel sensfunc FITS file; use to skip the sensfunc stage when running apply on its own.")
    p.add_argument("--sens-red", type=str, default=None, help="Pre-built red-channel sensfunc FITS file; use to skip the sensfunc stage when running apply on its own.")

    p.add_argument("--science-glob-blue", type=str, default=None, help="Glob pattern matching blue-channel science datacubes to flux-calibrate in the apply stage.")
    p.add_argument("--science-glob-green", type=str, default=None, help="Glob pattern matching green-channel science datacubes to flux-calibrate in the apply stage.")
    p.add_argument("--science-glob-red", type=str, default=None, help="Glob pattern matching red-channel science datacubes to flux-calibrate in the apply stage.")

    p.add_argument(
        "--boxcar-radius-arcsec",
        type=float,
        default=1.0,
        help="Radius (arcsec) of the circular aperture, centered on the measured centroid, used to sum standard-star fiber flux in the extract stage.",
    )
    p.add_argument(
        "--centroid-kernel-arcsec",
        type=float,
        default=1.0,
        help="Gaussian smoothing kernel size (arcsec) used to find the convolved white-light peak when centroiding the standard star.",
    )
    p.add_argument(
        "--exptime-key",
        type=str,
        default="EXPTIME",
        help="Science cube header keyword to read exposure time from when applying a sensfunc (does not affect standard-star extraction, which always reads RSS REXPTIME).",
    )
    p.add_argument(
        "--default-exptime",
        type=float,
        default=1.0,
        help="Fallback exposure time (s) used if --exptime-key is missing or invalid in a science cube header; a WARNING is logged whenever this fallback is used.",
    )
    p.add_argument(
        "--airmass-key",
        type=str,
        default="AIRMASS",
        help="Header keyword to read airmass from -- both for the standard RSS file in extract and for science cube headers in apply.",
    )
    p.add_argument(
        "--default-airmass",
        type=float,
        default=1.0,
        help="Fallback airmass used if --airmass-key is missing or invalid; a WARNING is logged whenever this fallback is used.",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="fluxcal",
        help="Root output directory; extractions/, sensfunc/, qa/, and cubes/ subdirectories are created under it.",
    )
    p.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files instead of raising an error.")
    p.add_argument("--verbose", action="store_true", default=False, help="Enable DEBUG-level logging.")
    p.add_argument("--debug", action="store_true", default=False, help="Enable PypeIt's own interactive/verbose debug output during sensfunc generation.")

    # Pre-parse to find --config (if given), then seed it as argparse defaults
    # before the real parse below -- so config values fill in anything not
    # explicitly passed on the command line, while explicit CLI flags still
    # win over the config file.
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config).expanduser().resolve()
        if not config_path.exists():
            p.error(f"Config file not found: {config_path}")
        valid_dests = {a.dest for a in p._actions if a.dest not in ("help", "config")}
        p.set_defaults(**_load_config_defaults(config_path, valid_dests))

    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    out_root = _ensure_dir(Path(args.output_root).expanduser().resolve())
    paths = {
        "root": out_root,
        "extractions": _ensure_dir(out_root / "extractions"),
        "sensfunc": _ensure_dir(out_root / "sensfunc"),
        "qa": _ensure_dir(out_root / "qa"),
        "cubes": _ensure_dir(out_root / "cubes"),
    }

    extracted = None
    sens_map = None

    if args.mode in ("extract", "all"):
        extracted = run_extract(args, paths)

    if args.mode in ("sensfunc", "all"):
        sens_map = run_sensfunc(args, paths, extracted=extracted)

    if args.mode in ("apply", "all"):
        run_apply(args, paths, sens_map=sens_map)

    LOG.info("Completed mode=%s", args.mode)


if __name__ == "__main__":
    main()
