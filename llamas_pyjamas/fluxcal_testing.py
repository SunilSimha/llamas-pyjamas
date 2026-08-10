#!/usr/bin/env python
"""
Independent LLAMAS throughput measurement for the HD14943 standard star.

This is a standalone cross-check, deliberately built without importing or
reusing any code from fluxcal_cubes.py (no PypeIt SensFunc/zeropoint
machinery, no SimpleCubeConstructor). It follows the same physical approach
as llamas_pyjamas/Flux/calcThroughput.py (raw sky-subtracted counts summed
over the brightest fibers, compared to the predicted photon counts from a
reference spectrum), but adapted to run on real pipeline data products:

  - Standard-star position + fiber selection: RSS FIBERMAP (BENCHSIDE,
    FIBER_ID) joined against llamas_pyjamas.Image.WhiteLightModule's
    FiberMap_LUT for fiber (x, y) positions. This is the pipeline's own
    fiber-geometry lookup, also used internally by SimpleCubeConstructor,
    but used directly here rather than through fluxcal_cubes.py.
  - Standard-star spectrum: the RSS "_FF" file's FLUX extension (raw
    detector counts, base sky-subtracted, fiber-to-fiber flat corrected),
    summed over fibers within a boxcar aperture of the flux-weighted
    centroid.
  - Reference spectrum: the CALSPEC model for HD14943, downloaded from
    STScI.
  - Predicted counts: reference flux -> photons -> detector counts, using
    the real Magellan Baade aperture, the exposure's own REXPTIME/AIRMASS
    (read straight from the RSS header), and the La Silla extinction curve
    shipped with PypeIt (the same curve calcThroughput.py had hardcoded
    inline).

Output (written to --output-dir, default redux/fluxcal_testing/):
  - hd14943_mod_004.fits          (downloaded CALSPEC reference, cached)
  - HD14943_extraction_diagnostic.png  (per-channel white-light cutout +
                                         fiber aperture map)
  - HD14943_throughput.png        (throughput vs wavelength, all channels)
  - HD14943_throughput.csv        (wave, throughput_pct, apsum, cts_pred per channel)
"""
import argparse
import csv
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

import pypeit
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT

REDUX_ROOT = Path(__file__).resolve().parent

CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/hd14943_mod_004.fits"

FIBER_PITCH_ARCSEC = 0.75      # LLAMAS hex fiber pitch (instrument design constant)
MAGELLAN_DIAMETER_CM = 650.0   # Baade 6.5m primary
MAGELLAN_OBSCURATION = 0.07    # secondary + spider obscuration fraction
DEFAULT_GAIN = 1.0             # e-/ADU; no EGAIN in LLAMAS headers, matches
                                # Extract/extractLlamas.py's own DEFAULT_GAIN
H_ERG_S = 6.626e-27            # Planck constant, erg s
C_ANG_S = 2.998e18             # speed of light, Angstrom/s

EXTINCTION_FILE = Path(pypeit.__file__).resolve().parent / "data" / "extinction" / "lasillaextinct.dat"

N_BRIGHT_FOR_CENTROID = 40     # brightest-fiber count used only to seed the centroid,
                                # matching calcThroughput.py's convention

LLAMAS_RESOLVING_POWER = 3000.0  # lambda/dlambda; same value used for the telgrid
                                  # file choice in the MagellanLLAMASSpectrograph class

CHANNELS = ("blue", "green", "red")
CHANNEL_COLORS = {"blue": "tab:blue", "green": "tab:green", "red": "tab:red"}


def download_calspec(out_path: Path, url: str = CALSPEC_URL) -> Path:
    if out_path.exists():
        print(f"  Using cached CALSPEC file: {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def load_calspec(path: Path):
    with fits.open(path) as hdul:
        d = hdul[1].data
        wave = np.asarray(d["WAVELENGTH"], dtype=float)   # Angstrom
        flux = np.asarray(d["FLUX"], dtype=float)          # erg/s/cm2/Ang
    return wave, flux


def degrade_calspec_resolution(wave, flux, R=LLAMAS_RESOLVING_POWER):
    """Gaussian-smooth the native R~200000 CALSPEC model down to ~LLAMAS
    resolution so that dividing observed counts by predicted counts doesn't
    blow up at the model's narrow (unresolved-by-LLAMAS) stellar absorption
    lines. Smoothing is done on a uniform log(wavelength) grid so a single
    kernel width gives constant R across the whole spectrum."""
    logw = np.log(wave)
    logw_uniform = np.linspace(logw.min(), logw.max(), len(logw))
    flux_uniform = np.interp(logw_uniform, logw, flux)
    dlogw = logw_uniform[1] - logw_uniform[0]
    fwhm_logw = 1.0 / R   # d(ln lambda) at fixed R defines the FWHM resolution element
    sigma_pixels = (fwhm_logw / 2.3548) / dlogw
    flux_smooth = gaussian_filter1d(flux_uniform, sigma_pixels)
    return np.exp(logw_uniform), flux_smooth


def load_extinction():
    data = np.loadtxt(EXTINCTION_FILE)
    return data[:, 0], data[:, 1]   # wavelength (Ang), extinction (mag/airmass)


def load_rss(path: Path):
    with fits.open(path) as hdul:
        wave = hdul["WAVE"].data.astype(float)
        flux = hdul["FLUX"].data.astype(float)   # sky-subtracted, fiber-flat corrected counts
        fibermap = hdul["FIBERMAP"].data
        header = hdul[0].header.copy()
    benchside = np.array([
        b.decode("utf-8").strip() if isinstance(b, bytes) else str(b).strip()
        for b in fibermap["BENCHSIDE"]
    ])
    fiber_id = np.asarray(fibermap["FIBER_ID"], dtype=int)
    return wave, flux, benchside, fiber_id, header


def fiber_positions_arcsec(benchside, fiber_id):
    """(x, y) fiber positions in arcsec, relative to the IFU bundle center,
    from the pipeline's own FiberMap_LUT (independent of any cube WCS)."""
    coords = np.full((len(benchside), 2), np.nan)
    for i, (b, f) in enumerate(zip(benchside, fiber_id)):
        x, y = FiberMap_LUT(b, int(f))
        if x == -1 and y == -1:
            continue
        coords[i] = (x * FIBER_PITCH_ARCSEC, y * FIBER_PITCH_ARCSEC)
    return coords


def flux_weighted_centroid(coords, total_flux, n_bright=N_BRIGHT_FOR_CENTROID):
    good = np.isfinite(coords[:, 0]) & np.isfinite(total_flux)
    idx = np.where(good)[0]
    order = idx[np.argsort(-total_flux[idx])][:n_bright]
    w = np.clip(total_flux[order], 0, None)
    cx = np.average(coords[order, 0], weights=w)
    cy = np.average(coords[order, 1], weights=w)
    return cx, cy, order


def select_aperture(coords, centroid, radius_arcsec):
    d = np.linalg.norm(coords - np.array(centroid), axis=1)
    return np.where(np.isfinite(d) & (d <= radius_arcsec))[0]


def cube_whitelight_cutout(cube_path: Path):
    with fits.open(cube_path) as hdul:
        flux_cube = hdul["FLUX"].data   # (nwave, ny, nx)
    return np.nanmedian(flux_cube, axis=0)


def compute_channel_throughput(channel, rss_path, calspec_wave, calspec_flux,
                                ext_wave, ext_mag_per_am, boxcar_radius_arcsec,
                                gain=DEFAULT_GAIN):
    wave, flux, benchside, fiber_id, header = load_rss(rss_path)
    coords = fiber_positions_arcsec(benchside, fiber_id)

    total_flux = np.nansum(flux, axis=1)
    cx, cy, _ = flux_weighted_centroid(coords, total_flux)
    aperture_idx = select_aperture(coords, (cx, cy), boxcar_radius_arcsec)
    if len(aperture_idx) == 0:
        raise RuntimeError(f"{channel}: no fibers found within {boxcar_radius_arcsec} arcsec of centroid")

    # Per-fiber wavelength solutions differ by ~tens of Angstroms at fixed
    # pixel index (arc-transfer residuals); for this coarse throughput
    # cross-check we sum in pixel space and use the median wavelength
    # solution across the aperture fibers as the common grid, same
    # approximation calcThroughput.py makes (it just uses fiber[0]'s wave).
    wave_common = np.nanmedian(wave[aperture_idx], axis=0)
    apsum = np.nansum(flux[aperture_idx], axis=0)

    exptime = float(header.get("REXPTIME", 1.0))
    airmass = float(header.get("AIRMASS", 1.0))

    calspec_interp = np.interp(wave_common, calspec_wave, calspec_flux)
    ext_interp = np.interp(wave_common, ext_wave, ext_mag_per_am, right=ext_mag_per_am[-1])
    atmos_trans = 10 ** (-0.4 * ext_interp * airmass)

    A_eff_cm2 = np.pi * (MAGELLAN_DIAMETER_CM / 2.0) ** 2 * (1.0 - MAGELLAN_OBSCURATION)
    e_photon = H_ERG_S * C_ANG_S / wave_common
    dlambda = np.gradient(wave_common)

    cts_pred = calspec_interp * A_eff_cm2 * exptime * dlambda / e_photon / gain * atmos_trans
    throughput_pct = apsum / cts_pred * 100.0

    return {
        "channel": channel,
        "wave": wave_common,
        "throughput": throughput_pct,
        "apsum": apsum,
        "cts_pred": cts_pred,
        "centroid": (cx, cy),
        "coords": coords,
        "aperture_idx": aperture_idx,
        "total_flux": total_flux,
        "exptime": exptime,
        "airmass": airmass,
    }


def make_diagnostic_plot(results, cube_paths, boxcar_radius_arcsec, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, ch in enumerate(CHANNELS):
        r = results[ch]

        ax_img = axes[0, col]
        img = cube_whitelight_cutout(cube_paths[ch])
        ax_img.imshow(img, origin="lower", cmap="viridis",
                      vmin=np.nanpercentile(img, 5), vmax=np.nanpercentile(img, 99.5))
        ax_img.set_title(f"{ch}: cube white light (median)")
        ax_img.set_xlabel("pixel")
        ax_img.set_ylabel("pixel")

        ax_ap = axes[1, col]
        coords = r["coords"]
        good = np.isfinite(coords[:, 0])
        sc = ax_ap.scatter(coords[good, 0], coords[good, 1], c=r["total_flux"][good],
                            cmap="viridis", s=18,
                            vmin=np.nanpercentile(r["total_flux"][good], 5),
                            vmax=np.nanpercentile(r["total_flux"][good], 99.5))
        cx, cy = r["centroid"]
        circle = plt.Circle((cx, cy), boxcar_radius_arcsec, fill=False, edgecolor="red", lw=1.5)
        ax_ap.add_patch(circle)
        ax_ap.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
        ax_ap.set_aspect("equal")
        ax_ap.set_xlabel("arcsec")
        ax_ap.set_ylabel("arcsec")
        ax_ap.set_title(f"{ch}: fiber aperture ({len(r['aperture_idx'])} fibers, r={boxcar_radius_arcsec}\")")
        fig.colorbar(sc, ax=ax_ap, fraction=0.046, pad=0.04)

    fig.suptitle("HD14943 -- independent extraction diagnostic (fluxcal_testing.py)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def make_throughput_plot(results, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ch in CHANNELS:
        r = results[ch]
        gd = np.isfinite(r["throughput"])
        ax.plot(r["wave"][gd], r["throughput"][gd], color=CHANNEL_COLORS[ch], lw=0.8, label=ch)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Throughput (%)")
    ax.set_ylim(0, 60)
    ax.set_title("HD14943 -- independent system throughput (fluxcal_testing.py)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def write_csv(results, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "wave_ang", "throughput_pct", "apsum_counts", "cts_pred_counts"])
        for ch in CHANNELS:
            r = results[ch]
            for wv, tp, ap, cp in zip(r["wave"], r["throughput"], r["apsum"], r["cts_pred"]):
                w.writerow([ch, f"{wv:.4f}", f"{tp:.5f}", f"{ap:.4f}", f"{cp:.4f}"])
    print(f"  Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rss-blue", default=str(REDUX_ROOT / "extractions" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_extract_RSS_blue_FF.fits"))
    p.add_argument("--rss-green", default=str(REDUX_ROOT / "extractions" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_extract_RSS_green_FF.fits"))
    p.add_argument("--rss-red", default=str(REDUX_ROOT / "extractions" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_extract_RSS_red_FF.fits"))
    p.add_argument("--cube-blue", default=str(REDUX_ROOT / "cubes" / "HD14943" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_cube_blue_skysub.fits"))
    p.add_argument("--cube-green", default=str(REDUX_ROOT / "cubes" / "HD14943" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_cube_green_skysub.fits"))
    p.add_argument("--cube-red", default=str(REDUX_ROOT / "cubes" / "HD14943" /
                   "LLAMAS_2025-12-17_05-26-57.1_SCI22_mef_flat_corrected_cube_red_skysub.fits"))
    p.add_argument("--boxcar-radius-arcsec", type=float, default=3.0)
    p.add_argument("--output-dir", default=str(REDUX_ROOT / "fluxcal_testing"))
    p.add_argument("--calspec-url", default=CALSPEC_URL)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reference data...")
    calspec_path = download_calspec(out_dir / "hd14943_mod_004.fits", args.calspec_url)
    calspec_wave, calspec_flux = load_calspec(calspec_path)
    calspec_wave, calspec_flux = degrade_calspec_resolution(calspec_wave, calspec_flux)
    ext_wave, ext_mag = load_extinction()

    rss_paths = {"blue": args.rss_blue, "green": args.rss_green, "red": args.rss_red}
    cube_paths = {"blue": args.cube_blue, "green": args.cube_green, "red": args.cube_red}

    results = {}
    for ch in CHANNELS:
        print(f"Processing {ch} channel...")
        results[ch] = compute_channel_throughput(
            ch, rss_paths[ch], calspec_wave, calspec_flux, ext_wave, ext_mag,
            args.boxcar_radius_arcsec,
        )
        r = results[ch]
        gd = np.isfinite(r["throughput"])
        print(f"  {ch}: {len(r['aperture_idx'])} fibers, exptime={r['exptime']:.2f}s, "
              f"airmass={r['airmass']:.3f}, centroid=({r['centroid'][0]:.2f}, {r['centroid'][1]:.2f}) arcsec")
        print(f"  {ch}: throughput peak={np.nanmax(r['throughput'][gd]):.2f}%, "
              f"median={np.nanmedian(r['throughput'][gd]):.2f}%")

    print("Writing outputs...")
    make_diagnostic_plot(results, cube_paths, args.boxcar_radius_arcsec,
                          out_dir / "HD14943_extraction_diagnostic.png")
    make_throughput_plot(results, out_dir / "HD14943_throughput.png")
    write_csv(results, out_dir / "HD14943_throughput.csv")
    print("Done.")


if __name__ == "__main__":
    main()
