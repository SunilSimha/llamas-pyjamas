#!/usr/bin/env python3
"""
Combine individual LLAMAS datacubes from multiple exposures into channel-combined cubes.

This script:
1. Reads all FITS cube files from an input folder
2. Groups them by channel (red, green, blue)
3. For each channel, reprojects all cubes to a common WCS (from the first file)
4. Combines using variance-weighted mean
5. Saves the combined cube with metadata

Usage:
    python combine_cubes.py <input_folder> [-o <output_folder>] [--overwrite]

Example:
    python combine_cubes.py ./cubes -o ./combined_cubes
"""

import argparse
import csv
import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.signal import correlate2d


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CubeCombiner:
    """Combine LLAMAS cubes by channel."""

    CHANNELS = ('red', 'green', 'blue')
    WEIGHTING_SCHEMES = ('inverse_variance', 'clipped_muse')
    clipped_muse_SCLIP = 10.0
    # Convolved-peak value above which a peak is assumed to be a cosmic ray (red channel only)
    RED_PEAK_MAX = 200.0
    DEFAULT_COLLAPSE_RANGES = {
        'blue': (3811.0, 3823.0),
        'green': (6715.0, 6727.0),
        'red': (7000.0, 7100.0),
    }

    @staticmethod
    def _exposure_key_from_path(file_path):
        """Build a channel-agnostic exposure key from a cube filename."""
        stem = Path(file_path).stem
        parts = stem.split('_')
        if len(parts) >= 4:
            # Example: LLAMAS_2025-12-17_03-19-36.3_SCI22_...
            return '_'.join(parts[:4])
        return stem

    @staticmethod
    def _extract_wavelength_axis(hdul):
        """Return wavelength axis in Angstrom for a cube HDUList."""
        if len(hdul) > 3 and hdul[3].data is not None and 'wavelength' in hdul[3].data.names:
            return np.asarray(hdul[3].data['wavelength'], dtype=np.float64)

        # Fallback: derive from primary WCS spectral keywords.
        hdr = hdul[0].header
        nwave = hdul[0].data.shape[0]
        crval3 = hdr.get('CRVAL3')
        cdelt3 = hdr.get('CDELT3')
        crpix3 = hdr.get('CRPIX3', 1.0)
        if crval3 is None or cdelt3 is None:
            raise ValueError('Unable to derive wavelength axis: no WAVELENGTH extension and missing CRVAL3/CDELT3')

        idx = np.arange(nwave, dtype=np.float64)
        wave = crval3 + (idx + 1.0 - crpix3) * cdelt3

        cunit3 = (hdr.get('CUNIT3') or '').strip().lower()
        if cunit3 in {'m', 'meter', 'metre'}:
            wave = wave * 1.0e10
        elif cunit3 in {'nm'}:
            wave = wave * 10.0
        # Assume Angstrom otherwise.
        return wave

    def collapse_cube_to_whitelight(
        self,
        hdul,
        wave_min_angstrom=6715.0,
        wave_max_angstrom=6727.0,
    ):
        """
        Collapse a cube into a weighted white-light image over a wavelength range.

        Returns
        -------
        image_2d : ndarray
            Weighted 2D image.
        wcs_2d : astropy.wcs.WCS
            Celestial WCS matching the input cube.
        """
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        var = np.asarray(hdul[1].data, dtype=np.float32) if len(hdul) > 1 and hdul[1].data is not None else None
        weight = np.asarray(hdul[2].data, dtype=np.float32) if len(hdul) > 2 and hdul[2].data is not None else None
        wave = self._extract_wavelength_axis(hdul)

        mask = (wave >= wave_min_angstrom) & (wave <= wave_max_angstrom)
        if not np.any(mask):
            logger.warning(
                f"No wavelength samples in [{wave_min_angstrom}, {wave_max_angstrom}] A; "
                "falling back to full cube collapse"
            )
            mask = np.ones_like(wave, dtype=bool)

        flux_sel = flux[mask, :, :]

        if weight is not None:
            weight_sel = weight[mask, :, :]
        else:
            weight_sel = np.ones_like(flux_sel, dtype=np.float32)

        if var is not None:
            var_sel = var[mask, :, :]
            valid_var = np.isfinite(var_sel) & (var_sel > 0)
            valid_weight = np.isfinite(weight_sel) & (weight_sel > 0)
            eff_w = np.zeros_like(weight_sel, dtype=np.float32)
            eff_w[valid_var] = weight_sel[valid_var] / var_sel[valid_var]
            fallback = (~valid_var) & valid_weight
            eff_w[fallback] = weight_sel[fallback]
        else:
            eff_w = np.where(np.isfinite(weight_sel) & (weight_sel > 0), weight_sel, 0.0)

        flux_sel = np.where(np.isfinite(flux_sel), flux_sel, 0.0)
        sum_w = np.sum(eff_w, axis=0)
        sum_wf = np.sum(eff_w * flux_sel, axis=0)
        image_2d = np.divide(
            sum_wf,
            sum_w,
            out=np.zeros_like(sum_wf, dtype=np.float32),
            where=sum_w > 0,
        )

        wcs_2d = WCS(hdul[0].header).celestial
        return image_2d, wcs_2d

    @staticmethod
    @staticmethod
    def _prepare_whitelight_for_correlation(image_2d):
        """Normalize a white-light image for cross-correlation."""
        image = np.asarray(image_2d, dtype=np.float32)
        finite = np.isfinite(image)
        if not np.any(finite):
            raise ValueError('Cannot cross-correlate an image with no finite pixels')

        prepared = np.zeros_like(image, dtype=np.float32)
        finite_values = image[finite]
        median = float(np.nanmedian(finite_values))
        prepared[finite] = finite_values - median

        scale = float(np.nanstd(finite_values))
        if np.isfinite(scale) and scale > 0.0:
            prepared /= scale

        return prepared

    @staticmethod
    def _mask_background_pixels(image_2d, sigma_multiplier=3.0):
        """Mask pixels below a robust background threshold.

        The background level is estimated from the median of finite pixels and the
        background sigma from the MAD. This keeps the cross-correlation focused on
        structured source light instead of detector background structure.
        """
        image = np.asarray(image_2d, dtype=np.float32)
        finite = np.isfinite(image)
        if not np.any(finite):
            raise ValueError('Cannot estimate a background threshold from an empty image')

        finite_values = image[finite]
        background_level = float(np.nanmedian(finite_values))
        mad = float(np.nanmedian(np.abs(finite_values - background_level)))
        background_sigma = 1.4826 * mad
        if not np.isfinite(background_sigma) or background_sigma <= 0.0:
            background_sigma = float(np.nanstd(finite_values))

        if not np.isfinite(background_sigma) or background_sigma <= 0.0:
            threshold = background_level
        else:
            threshold = background_level + sigma_multiplier * background_sigma

        masked = np.where(finite & (image >= threshold), image, np.nan)
        n_masked = int(np.sum(finite & (image < threshold)))
        return masked, threshold, background_level, background_sigma, n_masked

    @staticmethod
    def tabulate_alignment_from_whitelight(
        image_2d,
        reference_image_2d,
        wcs_2d,
        reference_wcs_2d,
        cube_name,
        csv_path,
        kernel_size_arcsec=1.0,
        mode='a',
        write_header=False,
        diagnostic_plot_path=None,
        peak_max=None,
    ):
        """
        Cross-correlate a white-light image against the reference and append alignment data to CSV.

        Parameters
        ----------
        peak_max : float or None
            If set, pixels in the convolved image whose value exceeds this threshold are
            excluded from peak finding (treated as cosmic rays / bad pixels).  If all
            pixels are above the threshold the limit is ignored and the global maximum is
            used with a warning.

        Returns
        -------
        dict
        """
        pix_scales_arcsec = proj_plane_pixel_scales(wcs_2d) * 3600.0
        pix_scale = float(np.nanmean(pix_scales_arcsec))
        sigma_pix = max((kernel_size_arcsec / pix_scale) / 2.355, 0.5)
        kernel = Gaussian2DKernel(x_stddev=sigma_pix)

        conv_img = convolve(
            np.where(np.isfinite(image_2d), image_2d, 0.0),
            kernel,
            boundary='extend',
            normalize_kernel=True,
            nan_treatment='fill',
            preserve_nan=False,
        )

        reference_corr_input = np.asarray(reference_image_2d, dtype=np.float32)
        target_corr_input = np.asarray(image_2d, dtype=np.float32)

        if peak_max is not None:
            ref_mask = np.isfinite(reference_corr_input) & (reference_corr_input > peak_max)
            target_mask = np.isfinite(target_corr_input) & (target_corr_input > peak_max)
            n_ref_masked = int(np.sum(ref_mask))
            n_target_masked = int(np.sum(target_mask))
            if n_ref_masked:
                logger.info(
                    f"  {cube_name}: masked {n_ref_masked} reference white-light pixel(s) above peak_max={peak_max:.1f}"
                )
            if n_target_masked:
                logger.info(
                    f"  {cube_name}: masked {n_target_masked} target white-light pixel(s) above peak_max={peak_max:.1f}"
                )
            if np.any(ref_mask):
                reference_corr_input = np.where(ref_mask, np.nan, reference_corr_input)
            if np.any(target_mask):
                target_corr_input = np.where(target_mask, np.nan, target_corr_input)

            if not np.any(np.isfinite(reference_corr_input)):
                logger.warning(
                    f"  {cube_name}: all reference pixels exceed peak_max={peak_max:.1f}; falling back to unmasked reference"
                )
                reference_corr_input = np.asarray(reference_image_2d, dtype=np.float32)
            if not np.any(np.isfinite(target_corr_input)):
                logger.warning(
                    f"  {cube_name}: all target pixels exceed peak_max={peak_max:.1f}; falling back to unmasked target"
                )
                target_corr_input = np.asarray(image_2d, dtype=np.float32)

        reference_corr_input, ref_bg_threshold, ref_bg_level, ref_bg_sigma, n_ref_bg_masked = (
            CubeCombiner._mask_background_pixels(reference_corr_input, sigma_multiplier=3.0)
        )
        target_corr_input, tgt_bg_threshold, tgt_bg_level, tgt_bg_sigma, n_tgt_bg_masked = (
            CubeCombiner._mask_background_pixels(target_corr_input, sigma_multiplier=3.0)
        )

        logger.info(
            f"  {cube_name}: background mask thresholds - reference={ref_bg_threshold:.3f} "
            f"(level={ref_bg_level:.3f}, sigma={ref_bg_sigma:.3f}, masked={n_ref_bg_masked}), "
            f"target={tgt_bg_threshold:.3f} (level={tgt_bg_level:.3f}, sigma={tgt_bg_sigma:.3f}, masked={n_tgt_bg_masked})"
        )

        reference_corr = CubeCombiner._prepare_whitelight_for_correlation(reference_corr_input)
        target_corr = CubeCombiner._prepare_whitelight_for_correlation(target_corr_input)
        corr_map = correlate2d(target_corr, reference_corr, mode='full', boundary='fill', fillvalue=0.0)
        corr_peak_flat = int(np.nanargmax(corr_map))
        corr_peak_y, corr_peak_x = np.unravel_index(corr_peak_flat, corr_map.shape)

        ref_center_x = (reference_image_2d.shape[1] - 1) / 2.0
        ref_center_y = (reference_image_2d.shape[0] - 1) / 2.0
        shift_x = float(corr_peak_x - (reference_image_2d.shape[1] - 1))
        shift_y = float(corr_peak_y - (reference_image_2d.shape[0] - 1))
        align_x = ref_center_x + shift_x
        align_y = ref_center_y + shift_y

        ref_center_sky = reference_wcs_2d.pixel_to_world(float(ref_center_x), float(ref_center_y))
        align_sky = wcs_2d.pixel_to_world(float(align_x), float(align_y))
        corr_peak_value = float(corr_map[corr_peak_y, corr_peak_x])
        align_ra = float(align_sky.ra.deg)
        align_dec = float(align_sky.dec.deg)
        ref_center_ra = float(ref_center_sky.ra.deg)
        ref_center_dec = float(ref_center_sky.dec.deg)

        if diagnostic_plot_path is not None:
            CubeCombiner.save_whitelight_alignment_diagnostic_plot(
                reference_image_2d=reference_image_2d,
                target_image_2d=image_2d,
                conv_img=conv_img,
                corr_map=corr_map,
                corr_peak_x=corr_peak_x,
                corr_peak_y=corr_peak_y,
                align_x=align_x,
                align_y=align_y,
                cube_name=cube_name,
                output_path=diagnostic_plot_path,
                peak_max=peak_max,
            )

        with open(csv_path, mode, newline='') as fobj:
            writer = csv.writer(fobj)
            if write_header:
                writer.writerow([
                    'cube_file',
                    'corr_peak_x',
                    'corr_peak_y',
                    'corr_peak_value',
                    'shift_x_pix',
                    'shift_y_pix',
                    'align_x_pix',
                    'align_y_pix',
                    'ref_center_ra_deg',
                    'ref_center_dec_deg',
                    'align_ra_deg',
                    'align_dec_deg',
                ])
            writer.writerow([
                cube_name,
                corr_peak_x,
                corr_peak_y,
                corr_peak_value,
                shift_x,
                shift_y,
                align_x,
                align_y,
                ref_center_ra,
                ref_center_dec,
                align_ra,
                align_dec,
            ])

        return {
            'corr_map': corr_map,
            'corr_peak_x': corr_peak_x,
            'corr_peak_y': corr_peak_y,
            'corr_peak_value': corr_peak_value,
            'shift_x': shift_x,
            'shift_y': shift_y,
            'align_x': align_x,
            'align_y': align_y,
            'ref_center_x': ref_center_x,
            'ref_center_y': ref_center_y,
            'ref_center_ra': ref_center_ra,
            'ref_center_dec': ref_center_dec,
            'align_ra': align_ra,
            'align_dec': align_dec,
        }

    @staticmethod
    def save_whitelight_alignment_diagnostic_plot(
        reference_image_2d,
        target_image_2d,
        conv_img,
        corr_map,
        corr_peak_x,
        corr_peak_y,
        align_x,
        align_y,
        cube_name,
        output_path,
        peak_max=None,
    ):
        """Save a diagnostic plot showing the cross-correlation alignment.

        The plot contains the reference image, the target image with the alignment
        point marked, and the 2D cross-correlation map with the selected peak.
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            finite_vals = target_image_2d[np.isfinite(target_image_2d)]
            if finite_vals.size > 0:
                vmin = np.nanpercentile(finite_vals, 5)
                vmax = np.nanpercentile(finite_vals, 99)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin, vmax = None, None
            else:
                vmin, vmax = None, None

            ref_finite_vals = reference_image_2d[np.isfinite(reference_image_2d)]
            if ref_finite_vals.size > 0:
                ref_vmin = np.nanpercentile(ref_finite_vals, 5)
                ref_vmax = np.nanpercentile(ref_finite_vals, 99)
                if not np.isfinite(ref_vmin) or not np.isfinite(ref_vmax) or ref_vmax <= ref_vmin:
                    ref_vmin, ref_vmax = None, None
            else:
                ref_vmin, ref_vmax = None, None

            corr_vmin = np.nanpercentile(corr_map, 5)
            corr_vmax = np.nanpercentile(corr_map, 99)
            if not np.isfinite(corr_vmin) or not np.isfinite(corr_vmax) or corr_vmax <= corr_vmin:
                corr_vmin, corr_vmax = None, None

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

            # --- left panel: reference white-light image ---
            ax = axes[0]
            im0 = ax.imshow(reference_image_2d, origin='lower', cmap='viridis', vmin=ref_vmin, vmax=ref_vmax)
            ref_center_x = (reference_image_2d.shape[1] - 1) / 2.0
            ref_center_y = (reference_image_2d.shape[0] - 1) / 2.0
            circ0 = Circle((ref_center_x, ref_center_y), radius=6.0, fill=False, color='red', linewidth=2.0)
            ax.add_patch(circ0)
            ax.plot(ref_center_x, ref_center_y, marker='x', color='white', markersize=8, markeredgewidth=1.5)
            ax.set_title('Reference white-light image')
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')
            fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

            # --- middle panel: target white-light image ---
            ax = axes[1]
            im1 = ax.imshow(target_image_2d, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            circ1 = Circle((align_x, align_y), radius=6.0, fill=False, color='red', linewidth=2.0)
            ax.add_patch(circ1)
            ax.plot(align_x, align_y, marker='x', color='white', markersize=8, markeredgewidth=1.5)
            ax.set_title('Target white-light image')
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')
            fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

            # --- right panel: 2D cross-correlation map ---
            ax = axes[2]
            corr_x = np.arange(corr_map.shape[1], dtype=np.float32) - (reference_image_2d.shape[1] - 1)
            corr_y = np.arange(corr_map.shape[0], dtype=np.float32) - (reference_image_2d.shape[0] - 1)
            im2 = ax.imshow(
                corr_map,
                origin='lower',
                cmap='magma',
                vmin=corr_vmin,
                vmax=corr_vmax,
                extent=[corr_x[0], corr_x[-1], corr_y[0], corr_y[-1]],
                aspect='auto',
            )
            ax.plot(0.0, 0.0, marker='+', color='cyan', markersize=10, markeredgewidth=1.5)
            ax.plot(
                float(corr_peak_x - (reference_image_2d.shape[1] - 1)),
                float(corr_peak_y - (reference_image_2d.shape[0] - 1)),
                marker='x',
                color='white',
                markersize=8,
                markeredgewidth=1.5,
            )
            ax.set_title('2D cross-correlation')
            ax.set_xlabel('Shift X [pix]')
            ax.set_ylabel('Shift Y [pix]')
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

            shift_x = float(corr_peak_x - (reference_image_2d.shape[1] - 1))
            shift_y = float(corr_peak_y - (reference_image_2d.shape[0] - 1))
            fig.suptitle(
                f"Cross-correlation diagnostic: {cube_name}  (shift_x={shift_x:+.1f}, shift_y={shift_y:+.1f})",
                fontsize=11,
            )
            fig.savefig(output_path, dpi=140)
            plt.close(fig)
            logger.info(f"  Saved cross-correlation diagnostic plot: {output_path}")
        except Exception as exc:
            logger.warning(f"Failed to save cross-correlation diagnostic plot {output_path}: {exc}")

    def create_wcs_corrected_copies(
        self,
        channel,
        cube_files,
        external_wcs_corrections=None,
        external_reference_cube=None,
        reference_cube=None,
    ):
        """
        Measure per-cube white-light cross-correlation shifts, write an alignment table, and save WCS-corrected copies.

        The reference cube defines the reference celestial location at the center of its white-light
        image. All cubes (including the reference itself) keep their measured alignment pixel fixed
        while shifting CRVAL1/2 so that pixel maps to the reference center celestial coordinates.

        Parameters
        ----------
        reference_cube : str, optional
            Path to the FITS cube to compute alignment offsets from. If None, defaults to the
            first file in cube_files (previous behavior).

        If external_wcs_corrections is provided, those per-exposure dRA/dDec corrections
        are applied directly (used for forcing RED/BLUE to reuse GREEN channel alignment).
        """
        corrected_dir = self.output_folder / 'corrected_cubes' / channel
        corrected_dir.mkdir(parents=True, exist_ok=True)
        peak_csv = self.output_folder / f'peak_table_{channel}.csv'

        corrected_files = []
        correction_map = {}
        ref_sky = None
        ref_wl_img = None
        ref_wl_wcs = None
        use_external = external_wcs_corrections is not None
        wave_min, wave_max = self.get_channel_wave_range(channel)
        logger.info(
            f"  Using white-light collapse range for {channel.upper()}: "
            f"{wave_min:.1f}-{wave_max:.1f} A"
        )

        if use_external:
            logger.info(
                f"  Applying WCS corrections from GREEN channel to {channel.upper()}"
            )
        else:
            reference_path = reference_cube if reference_cube is not None else cube_files[0]
            logger.info(f"  Using alignment reference cube: {Path(reference_path).name}")
            with fits.open(reference_path) as ref_hdul:
                ref_wl_img, ref_wl_wcs = self.collapse_cube_to_whitelight(
                    ref_hdul,
                    wave_min_angstrom=wave_min,
                    wave_max_angstrom=wave_max,
                )
            ref_center_x = (ref_wl_img.shape[1] - 1) / 2.0
            ref_center_y = (ref_wl_img.shape[0] - 1) / 2.0
            ref_sky = ref_wl_wcs.pixel_to_world(float(ref_center_x), float(ref_center_y))

        for i, fpath in enumerate(cube_files):
            with fits.open(fpath) as hdul:
                new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])

                if use_external:
                    exposure_key = self._exposure_key_from_path(fpath)
                    if exposure_key not in external_wcs_corrections:
                        raise KeyError(
                            f"No GREEN-channel correction found for {Path(fpath).name} "
                            f"(exposure key: {exposure_key})"
                        )
                    corr = external_wcs_corrections[exposure_key]
                    dra = float(corr['dra'])
                    ddec = float(corr['ddec'])
                    hdr = new_hdul[0].header
                    hdr['CRVAL1'] = float(hdr['CRVAL1'] + dra)
                    hdr['CRVAL2'] = float(hdr['CRVAL2'] + ddec)
                    hdr['WCSCORR'] = (True, 'GREEN-alignment transferred')
                    if external_reference_cube is not None:
                        hdr['WCSREF'] = (Path(external_reference_cube).name[:40], 'GREEN reference cube')
                    hdr['WCSCHNL'] = ('GREEN', 'Channel supplying WCS correction')
                    hdr['WCSDRA'] = (dra, 'Applied dRA [deg] from GREEN')
                    hdr['WCSDDEC'] = (ddec, 'Applied dDec [deg] from GREEN')
                else:
                    wl_img, wl_wcs = self.collapse_cube_to_whitelight(
                        hdul,
                        wave_min_angstrom=wave_min,
                        wave_max_angstrom=wave_max,
                    )
                    diag_plot = self.output_folder / (
                        f"alignment_diagnostic_{channel}_{Path(fpath).stem}.png"
                    )
                    write_header = i == 0
                    peak_max = self.RED_PEAK_MAX if channel == 'red' else None
                    alignment = self.tabulate_alignment_from_whitelight(
                        wl_img,
                        ref_wl_img,
                        wl_wcs,
                        ref_wl_wcs,
                        Path(fpath).name,
                        peak_csv,
                        kernel_size_arcsec=1.0,
                        mode='w' if i == 0 else 'a',
                        write_header=write_header,
                        diagnostic_plot_path=diag_plot,
                        peak_max=peak_max,
                    )

                    cur_sky = SkyCoord(
                        ra=alignment['align_ra'] * u.deg,
                        dec=alignment['align_dec'] * u.deg,
                        frame='icrs',
                    )
                    dra = (ref_sky.ra.deg - cur_sky.ra.deg + 180.0) % 360.0 - 180.0
                    ddec = ref_sky.dec.deg - cur_sky.dec.deg
                    hdr = new_hdul[0].header
                    hdr['CRVAL1'] = float(hdr['CRVAL1'] + dra)
                    hdr['CRVAL2'] = float(hdr['CRVAL2'] + ddec)
                    hdr['WCSCORR'] = (True, 'Cross-correlation aligned')
                    hdr['WCSREF'] = (Path(reference_path).name[:40], 'Reference cube')
                    hdr['WCSPIXX'] = (float(alignment['align_x']), 'Alignment X pixel')
                    hdr['WCSPIXY'] = (float(alignment['align_y']), 'Alignment Y pixel')
                    hdr['WCSDX'] = (float(alignment['shift_x']), 'Cross-correlation shift in X [pix]')
                    hdr['WCSDY'] = (float(alignment['shift_y']), 'Cross-correlation shift in Y [pix]')

                    correction_map[self._exposure_key_from_path(fpath)] = {
                        'dra': float(dra),
                        'ddec': float(ddec),
                        'cube_file': Path(fpath).name,
                    }

                out_path = corrected_dir / Path(fpath).name
                new_hdul.writeto(out_path, overwrite=True)
                corrected_files.append(str(out_path))

        if not use_external:
            logger.info(f"  Wrote peak table: {peak_csv}")
        logger.info(f"  Wrote WCS-corrected cubes: {corrected_dir}")
        return corrected_files, correction_map

    @staticmethod
    def reproject_cube_slice(data_3d, header_3d, target_header, shape_out):
        """
        Reproject a 3D cube by reprojecting each wavelength slice independently.

        Parameters
        ----------
        data_3d : ndarray
            3D data array (nwave, ny, nx)
        header_3d : fits.Header
            Header with 3D WCS
        target_header : fits.Header
            Target header WCS
        shape_out : tuple
            Output shape

        Returns
        -------
        reprojected : ndarray
            Reprojected 3D array
        """
        nwave = data_3d.shape[0]
        reprojected = np.zeros(shape_out, dtype=np.float32)

        # Build 2D celestial WCS objects once. The wavelength axis is unchanged and
        # handled implicitly by reprojecting each spectral plane independently.
        wcs_in_2d = WCS(header_3d).celestial
        wcs_out_2d = WCS(target_header).celestial

        for i in range(nwave):
            # Reproject 2D slice at this wavelength
            data_2d = data_3d[i, :, :]
            reprojected_2d, _ = reproject_interp(
                (data_2d, wcs_in_2d),
                wcs_out_2d,
                shape_out=shape_out[1:]  # Just spatial dimensions (ny, nx)
            )
            reprojected[i, :, :] = reprojected_2d

        return reprojected

    def __init__(
        self,
        input_folder,
        output_folder=None,
        use_skysub=False,
        overwrite=False,
        collapse_wave_ranges=None,
        weighting_scheme='inverse_variance',
        enable_cr_clip=True,
        reference_cube=None,
    ):
        """
        Initialize the combiner.

        Parameters
        ----------
        input_folder : str
            Path to folder containing individual cube FITS files
        output_folder : str, optional
            Path to output folder. If None, uses input_folder/combined
        use_skysub : bool, optional
            Whether to use sky-subtracted cubes (if available) instead of raw cubes
        overwrite : bool, optional
            Whether to overwrite existing output files
        collapse_wave_ranges : dict, optional
            Mapping of channel to (wave_min, wave_max) in Angstrom used for
            white-light image generation.
        weighting_scheme : str, optional
            Combination weighting scheme. One of:
            - 'inverse_variance' (default): WEIGHT/VAR weighting
            - 'clipped_muse': C-style clipped equal-weight averaging
        enable_cr_clip : bool, optional
            Enable sigma clipping for the clipped_muse combiner.
        reference_cube : str, optional
            Path to a FITS cube to use as the alignment reference when computing
            per-exposure WCS offsets. If None, the first cube found for the GREEN
            channel is used (previous behavior).
        """
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        self.reference_cube = None
        if reference_cube is not None:
            self.reference_cube = Path(reference_cube)
            if not self.reference_cube.exists():
                raise FileNotFoundError(f"Reference cube not found: {reference_cube}")

        if output_folder is None:
            self.output_folder = self.input_folder / 'combined'
        else:
            self.output_folder = Path(output_folder)

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        if weighting_scheme not in self.WEIGHTING_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme '{weighting_scheme}'. "
                f"Valid options: {', '.join(self.WEIGHTING_SCHEMES)}"
            )
        self.weighting_scheme = weighting_scheme
        self.use_skysub = bool(use_skysub)
        self.enable_cr_clip = bool(enable_cr_clip)
        self.collapse_wave_ranges = dict(self.DEFAULT_COLLAPSE_RANGES)
        if collapse_wave_ranges is not None:
            for channel, wave_range in collapse_wave_ranges.items():
                if channel not in self.CHANNELS:
                    raise ValueError(f"Unknown channel in collapse_wave_ranges: {channel}")
                if wave_range[0] >= wave_range[1]:
                    raise ValueError(f"Invalid wavelength range for {channel}: {wave_range}")
                self.collapse_wave_ranges[channel] = (float(wave_range[0]), float(wave_range[1]))

        logger.info(f"Input folder: {self.input_folder}")
        logger.info(f"Output folder: {self.output_folder}")
        if self.reference_cube is not None:
            logger.info(f"Alignment reference cube: {self.reference_cube}")
        logger.info(f"Weighting scheme: {self.weighting_scheme}")
        logger.info(f"CR clipping enabled (clipped_muse): {self.enable_cr_clip}")
        for channel in self.CHANNELS:
            wave_min, wave_max = self.collapse_wave_ranges[channel]
            logger.info(
                f"White-light collapse range ({channel.upper()}): "
                f"{wave_min:.1f}-{wave_max:.1f} A"
            )

    @staticmethod
    def _combine_inverse_variance(flux_arr, var_arr, weight_arr):
        """Combine stack using inverse-variance weights with WEIGHT fallback."""
        valid_flux = np.isfinite(flux_arr) & (flux_arr != 0)
        valid_var = valid_flux & np.isfinite(var_arr) & (var_arr > 0)
        valid_weight = np.isfinite(weight_arr) & (weight_arr > 0)
        w = np.zeros_like(var_arr, dtype=np.float32)
        w[valid_var] = weight_arr[valid_var] / var_arr[valid_var]

        fallback = valid_flux & (~valid_var) & valid_weight
        w[fallback] = weight_arr[fallback]

        # Guard against invalid values introduced by reprojection.
        w = np.where(np.isfinite(w), w, 0.0)
        flux_for_sum = np.where(valid_flux, flux_arr, 0.0)

        logger.debug(
            f"    Valid inverse-var voxels: {np.count_nonzero(valid_var)}; "
            f"fallback weight voxels: {np.count_nonzero(fallback)}; "
            f"missing/zero voxels: {np.count_nonzero(~valid_flux)}"
        )

        sum_w = np.sum(w, axis=0)
        weighted_sum = np.sum(w * flux_for_sum, axis=0)
        combined_flux = np.divide(
            weighted_sum,
            sum_w,
            out=np.zeros_like(weighted_sum, dtype=np.float32),
            where=sum_w > 0,
        )

        combined_var = np.divide(
            1.0,
            sum_w,
            out=np.zeros_like(combined_flux, dtype=np.float32),
            where=sum_w > 0,
        )

        combined_weight = np.sum(
            np.where(valid_flux & valid_weight, weight_arr, 0.0),
            axis=0,
        ).astype(np.float32)
        return combined_flux, combined_var, combined_weight

    @staticmethod
    def _estimate_frame_scale_from_cube(flux_cube):
        """
        Estimate per-exposure mmd scale from a representative spectral slice.

        This mirrors the C code approach of using the median absolute deviation
        in a mid-cube plane to define an exposure-level scaling term.
        """
        cube = np.asarray(flux_cube, dtype=np.float32)
        if cube.ndim != 3 or cube.shape[0] == 0:
            return 1.0

        mid_idx = cube.shape[0] // 2
        mid_plane = cube[mid_idx, :, :]

        valid = np.isfinite(mid_plane) & (mid_plane != 0)
        if not np.any(valid):
            return 1.0

        vals = mid_plane[valid].astype(np.float64)
        med_val = float(np.median(vals))
        mad_val = float(np.median(np.abs(vals - med_val)))
        if not np.isfinite(mad_val) or mad_val <= 0.0:
            return 1.0

        return mad_val

    @classmethod
    def _combine_clipped_muse(cls, flux_arr, var_arr, frame_scales, enable_cr_clip=True):
        """Combine stack using C-style mmd-weighted averaging with optional clipping."""
        flux = np.asarray(flux_arr, dtype=np.float32)

        # Use sigma = sqrt(var) for clipping and propagated output uncertainty.
        sigma = np.sqrt(
            np.where(np.isfinite(var_arr) & (var_arr > 0), var_arr, np.nan)
        ).astype(np.float32)

        # Treat exact-zero flux as missing data (e.g. dead-detector regions).
        valid_flux = np.isfinite(flux) & (flux != 0)

        # C code initializes flags from positive sigma.
        flags = valid_flux & np.isfinite(sigma) & (sigma > 0)
        expand_shape = (flux.shape[0],) + (1,) * (flux.ndim - 1)

        if enable_cr_clip:
            # Iterative clipping against median(flux) using median(sigma) scale.
            while True:
                flux_for_med = np.where(flags, flux, np.nan)
                sigma_for_med = np.where(flags, sigma, np.nan)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                    med_flux = np.nanmedian(flux_for_med, axis=0)
                    med_sigma = np.nanmedian(sigma_for_med, axis=0)

                with np.errstate(invalid='ignore', divide='ignore'):
                    dev = np.abs(flux - np.expand_dims(med_flux, axis=0)) / np.expand_dims(med_sigma, axis=0)

                reject = flags & np.isfinite(dev) & (dev > cls.clipped_muse_SCLIP)
                if not np.any(reject):
                    break
                flags = flags & (~reject)

        mmd = np.asarray(frame_scales, dtype=np.float32)
        if mmd.ndim != 1 or mmd.shape[0] != flux.shape[0]:
            raise ValueError(
                f"frame_scales length {mmd.shape[0] if mmd.ndim == 1 else 'invalid'} "
                f"does not match number of frames {flux.shape[0]}"
            )
        mmd = np.where(np.isfinite(mmd) & (mmd > 0), mmd, 1.0)
        inv_mmd2 = (1.0 / (mmd * mmd)).reshape(expand_shape)

        weights = np.where(flags, inv_mmd2, 0.0)
        dom = np.sum(weights, axis=0)

        weighted_flux_sum = np.sum(np.where(flags, flux, 0.0) * inv_mmd2, axis=0)
        combined_flux = np.divide(
            weighted_flux_sum,
            dom,
            out=np.zeros_like(weighted_flux_sum, dtype=np.float32),
            where=dom > 0,
        )

        sigma2_sum = np.sum(np.where(flags, (sigma * inv_mmd2) ** 2, 0.0), axis=0)
        combined_sigma = np.divide(
            np.sqrt(sigma2_sum),
            dom,
            out=np.zeros_like(weighted_flux_sum, dtype=np.float32),
            where=dom > 0,
        )
        combined_var = combined_sigma**2

        kept = np.sum(flags, axis=0)

        # Fallback where all samples were rejected/invalid: use robust median flux.
        no_kept = kept <= 0
        if np.any(no_kept):
            fallback_flux = np.nanmedian(
                np.where(np.isfinite(flux), flux, np.nan),
                axis=0,
            )
            combined_flux[no_kept] = np.where(
                np.isfinite(fallback_flux[no_kept]),
                fallback_flux[no_kept],
                0.0,
            )
            combined_var[no_kept] = 0.0

        combined_weight = kept.astype(np.float32)
        return combined_flux.astype(np.float32), combined_var.astype(np.float32), combined_weight

    def get_channel_wave_range(self, channel):
        """Return (wave_min, wave_max) in Angstrom for a channel."""
        if channel not in self.collapse_wave_ranges:
            raise KeyError(f"No wavelength collapse range configured for channel: {channel}")
        return self.collapse_wave_ranges[channel]

    def find_cubes_by_channel(self, use_skysub=False):
        """
        Find all cube files and group by channel.

        Arguments
        ---------
        use_skysub : bool, optional
            If True, only include cubes that have been sky-subtracted (filename contains 'skysub').

        Returns
        -------
        dict
            Dictionary mapping channel name to list of file paths
        """
        # Find all FITS files that look like cubes
        keyword = 'cube'
        if use_skysub:
            suffix = "_skysub.fits"
        else:
            suffix = ".fits"
        pattern = self.input_folder / f'*{keyword}*{suffix}'
        cube_files = sorted(glob.glob(str(pattern)))

        if not cube_files:
            raise FileNotFoundError(f"No cube FITS files found in {self.input_folder}")

        logger.info(f"Found {len(cube_files)} cube files")

        # Group by channel
        cubes_by_channel = {ch: [] for ch in self.CHANNELS}
        for fpath in cube_files:
            fname = Path(fpath).name.lower()
            for channel in self.CHANNELS:
                if f'{keyword}_{channel}' in fname or f'_{keyword}_{channel}{suffix}' in fname:
                    cubes_by_channel[channel].append(fpath)
                    break

        # Report
        for channel in self.CHANNELS:
            n = len(cubes_by_channel[channel])
            if n > 0:
                logger.info(f"  {channel.upper()}: {n} cube(s)")

        return cubes_by_channel

    def combine_channel(self, channel, cube_files, weighting_scheme=None, reference_file=None):
        """
        Combine cubes for a single channel.

        Parameters
        ----------
        channel : str
            Channel name ('red', 'green', 'blue')
        cube_files : list
            List of file paths for this channel
        weighting_scheme : str, optional
            Weighting scheme for this combination. If None, uses self.weighting_scheme.
        reference_file : str, optional
            Path to the FITS cube whose WCS and footprint (shape) define the output grid
            that all cubes are reprojected onto. If None, defaults to cube_files[0]
            (previous behavior).

        Returns
        -------
        combined_flux : ndarray
            Combined flux cube (nwave, ny, nx)
        combined_var : ndarray
            Combined variance cube
        combined_weight : ndarray
            Combined weight map
        reference_header : fits.Header
            Reference WCS header
        """
        scheme = weighting_scheme or self.weighting_scheme
        if scheme not in self.WEIGHTING_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme '{scheme}'. "
                f"Valid options: {', '.join(self.WEIGHTING_SCHEMES)}"
            )

        logger.info(f"\nProcessing {channel.upper()} channel ({len(cube_files)} file(s))...")

        # Load reference WCS/footprint from the designated reference file (default: first file)
        ref_file = reference_file if reference_file is not None else cube_files[0]
        with fits.open(ref_file) as hdul:
            ref_header = hdul[0].header.copy()
            ref_shape = hdul[0].data.shape
            logger.info(f"  Reference cube: {Path(ref_file).name}, shape: {ref_shape}")

        flux_stack = []
        var_stack = []
        weight_stack = []
        frame_scales = []
        bunit_seen = []

        # Reproject all cubes to reference WCS
        for i, fpath in enumerate(cube_files):
            logger.info(f"  Processing {i+1}/{len(cube_files)}: {Path(fpath).name}")

            with fits.open(fpath) as hdul:
                # WCS is always in the primary HDU header
                primary_header = hdul[0].header
                file_bunit = primary_header.get('BUNIT', 'counts')
                bunit_seen.append(file_bunit)

                # Reproject flux (slice by slice)
                flux_data = hdul[0].data
                frame_scales.append(self._estimate_frame_scale_from_cube(flux_data))
                flux_reprojected = self.reproject_cube_slice(
                    flux_data, primary_header, ref_header, ref_shape
                )
                flux_stack.append(flux_reprojected)

                # Reproject variance (use primary header for WCS, slice by slice)
                if len(hdul) > 1 and hdul[1].name == 'VAR':
                    var_data = hdul[1].data
                    var_reprojected = self.reproject_cube_slice(
                        var_data, primary_header, ref_header, ref_shape
                    )
                elif file_bunit == 'counts':
                    # If no VAR extension, estimate from flux (assume Poisson).
                    # Only valid for raw counts -- see the flux-calibrated branch below.
                    logger.warning(f"    No VAR extension found, using Poisson estimate")
                    var_reprojected = np.abs(flux_reprojected)  # Poisson: var = flux
                else:
                    # BUNIT is flux-calibrated (e.g. '1e-17 erg s-1 cm-2 Angstrom-1'), so
                    # "variance = flux" (Poisson counting statistics) no longer holds --
                    # that would silently fabricate a physically wrong variance. Mark this
                    # frame's variance as unusable; _combine_inverse_variance/_combine_
                    # clipped_muse both already fall back to weight-only combination
                    # wherever var is non-finite.
                    logger.warning(
                        f"    No VAR extension found for flux-calibrated cube (BUNIT={file_bunit!r}); "
                        "Poisson variance is not valid for calibrated flux, falling back to weight-only combination for this frame"
                    )
                    var_reprojected = np.full(ref_shape, np.nan, dtype=np.float32)

                var_stack.append(var_reprojected)

                # Reproject weight (use primary header for WCS, slice by slice)
                if len(hdul) > 2 and hdul[2].name == 'WEIGHT':
                    weight_data = hdul[2].data
                    weight_reprojected = self.reproject_cube_slice(
                        weight_data, primary_header, ref_header, ref_shape
                    )
                    weight_stack.append(weight_reprojected)
                else:
                    # If no WEIGHT extension, use uniform weight
                    logger.warning(f"    No WEIGHT extension found, using uniform weights")
                    weight_reprojected = np.ones(ref_shape)
                    weight_stack.append(weight_reprojected)

        # Refuse to combine cubes with inconsistent flux units (e.g. mixing
        # raw-counts cubes with flux-calibrated ones) -- that would silently
        # average physically incompatible quantities.
        unique_bunits = set(bunit_seen)
        if len(unique_bunits) > 1:
            raise ValueError(
                f"Cannot combine {channel.upper()} cubes with inconsistent BUNIT values "
                f"{sorted(unique_bunits)}; all input cubes for a channel must be in the "
                "same units (e.g. all flux-calibrated, or all raw counts)."
            )
        ref_header['BUNIT'] = bunit_seen[0]

        # Combine according to selected weighting scheme.
        flux_arr = np.array(flux_stack)
        var_arr = np.array(var_stack)
        weight_arr = np.array(weight_stack)
        frame_scales_arr = np.asarray(frame_scales, dtype=np.float32)

        logger.info(
            f"  Combining {len(cube_files)} cube(s) using '{scheme}' weighting..."
        )
        with warnings.catch_warnings():
            # var_arr can be all-NaN (e.g. every frame missing VAR for
            # flux-calibrated cubes, see the fallback above) -- that's an
            # expected, already-warned-about condition, not worth a second
            # numpy warning here.
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            logger.debug(f"    Flux array shape: {flux_arr.shape}, range: [{np.nanmin(flux_arr):.3f}, {np.nanmax(flux_arr):.3f}]")
            logger.debug(f"    Variance array shape: {var_arr.shape}, range: [{np.nanmin(var_arr):.3f}, {np.nanmax(var_arr):.3f}]")
            logger.debug(f"    Weight array shape: {weight_arr.shape}, range: [{np.nanmin(weight_arr):.3f}, {np.nanmax(weight_arr):.3f}]")

        if scheme == 'inverse_variance':
            combined_flux, combined_var, combined_weight = self._combine_inverse_variance(
                flux_arr, var_arr, weight_arr
            )
        else:
            combined_flux, combined_var, combined_weight = self._combine_clipped_muse(
                flux_arr,
                var_arr,
                frame_scales_arr,
                enable_cr_clip=self.enable_cr_clip,
            )

        logger.info(f"  Combined flux shape: {combined_flux.shape}")
        logger.info(f"  Flux range: [{np.nanmin(combined_flux):.3f}, {np.nanmax(combined_flux):.3f}]")

        return combined_flux, combined_var, combined_weight, ref_header

    def save_combined_cube(self, channel, flux, var, weight, header):
        """
        Save combined cube to FITS file.

        Parameters
        ----------
        channel : str
            Channel name
        flux : ndarray
            Flux cube
        var : ndarray
            Variance cube
        weight : ndarray
            Weight map
        header : fits.Header
            Reference WCS header
        """
        output_file = self.output_folder / f'LLAMAS_combined_cube_{channel}.fits'

        if output_file.exists() and not self.overwrite:
            raise FileExistsError(f"Output file exists (use --overwrite): {output_file}")

        logger.info(f"  Saving to: {output_file}")

        hdul = fits.HDUList()

        # Primary HDU with flux and WCS
        # Propagate the real flux units from the input cubes (validated
        # consistent across all inputs in combine_channel) rather than
        # assuming raw counts -- inputs are flux-calibrated physical units
        # whenever combination runs after fluxing.
        flux_bunit = header.get('BUNIT', 'counts')
        var_bunit = 'counts^2' if flux_bunit == 'counts' else f'({flux_bunit})^2'

        primary = fits.PrimaryHDU(data=flux, header=header)
        primary.header['EXTNAME'] = 'FLUX'
        primary.header['BUNIT'] = flux_bunit
        primary.header['COMMENT'] = f'LLAMAS combined datacube - {channel} channel'
        primary.header['PROCTYPE'] = 'COMBINED'
        primary.header['NCOMBINE'] = (flux.shape[0], 'Number of exposures combined')
        primary.header['WGTMETH'] = (self.weighting_scheme, 'Cube combination weighting method')
        hdul.append(primary)

        # Variance extension
        var_hdu = fits.ImageHDU(data=var, name='VAR')
        var_hdu.header['BUNIT'] = var_bunit
        var_hdu.header['COMMENT'] = 'Variance cube'
        hdul.append(var_hdu)

        # Weight extension
        weight_hdu = fits.ImageHDU(data=weight, name='WEIGHT')
        weight_hdu.header['COMMENT'] = 'Combined weight map'
        hdul.append(weight_hdu)

        # Write to file
        hdul.writeto(output_file, overwrite=self.overwrite)

        logger.info(f"  Wrote {len(hdul)} HDUs, file size: {output_file.stat().st_size / 1024**2:.1f} MB")

    def run(self):
        """Run the full combination pipeline."""
        logger.info("=" * 60)
        logger.info("LLAMAS Cube Combination")
        logger.info("=" * 60)

        # Find cubes by channel
        cubes_by_channel = self.find_cubes_by_channel(use_skysub=self.use_skysub)

        # GREEN provides the only trusted alignment solution. Solve it first,
        # then transfer its per-exposure WCS corrections to RED and BLUE.
        channel_order = ('green', 'red', 'blue')
        green_wcs_corrections = None
        green_reference_cube = None

        # Channel-agnostic exposure key identifying which exposure's WCS/footprint
        # the combined output should match, if a reference cube was specified.
        reference_exposure_key = (
            self._exposure_key_from_path(self.reference_cube) if self.reference_cube is not None else None
        )

        # Process each channel
        for channel in channel_order:
            cube_files = cubes_by_channel[channel]
            if not cube_files:
                logger.warning(f"No cubes found for {channel.upper()} channel, skipping")
                continue

            if len(cube_files) == 1:
                logger.warning(f"Only 1 cube for {channel.upper()}, no combination needed")
                continue

            try:
                if channel == 'green':
                    corrected_files, green_wcs_corrections = self.create_wcs_corrected_copies(
                        channel,
                        cube_files,
                        reference_cube=self.reference_cube,
                    )
                    green_reference_cube = (
                        str(self.reference_cube) if self.reference_cube is not None else cube_files[0]
                    )
                else:
                    if green_wcs_corrections is None:
                        raise RuntimeError(
                            f"Cannot process {channel.upper()} before GREEN WCS corrections are available"
                        )
                    corrected_files, _ = self.create_wcs_corrected_copies(
                        channel,
                        cube_files,
                        external_wcs_corrections=green_wcs_corrections,
                        external_reference_cube=green_reference_cube,
                    )
                if reference_exposure_key is not None:
                    match_idx = next(
                        (idx for idx, f in enumerate(cube_files)
                         if self._exposure_key_from_path(f) == reference_exposure_key),
                        None,
                    )
                    if match_idx is None:
                        logger.warning(
                            f"Reference exposure not found among {channel.upper()} cubes; "
                            f"using first {channel.upper()} cube to define output WCS/footprint"
                        )
                        reference_file_for_grid = corrected_files[0]
                    else:
                        reference_file_for_grid = corrected_files[match_idx]
                else:
                    reference_file_for_grid = corrected_files[0]

                flux, var, weight, header = self.combine_channel(
                    channel,
                    corrected_files,
                    weighting_scheme=self.weighting_scheme,
                    reference_file=reference_file_for_grid,
                )
                self.save_combined_cube(channel, flux, var, weight, header)
            except Exception as e:
                logger.error(f"Error processing {channel.upper()} channel: {e}", exc_info=True)
                continue

        logger.info("\n" + "=" * 60)
        logger.info("Combination complete")
        logger.info("=" * 60)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Combine LLAMAS datacubes by channel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine cubes in current folder, output to ./combined
  python combine_cubes.py .

  # Combine cubes from specified folder
  python combine_cubes.py /path/to/cubes -o /path/to/output

  # Overwrite existing output files
  python combine_cubes.py ./cubes --overwrite
        """
    )
    parser.add_argument(
        'input_folder',
        help='Folder containing individual cube FITS files'
    )
    parser.add_argument(
        '-o', '--output',
        dest='output_folder',
        default=None,
        help='Output folder (default: input_folder/combined)'
    )

    parser.add_argument(
            '-s', '--skysub',
            dest='use_skysub',
            default=False,
            action='store_true',
            help='Use sky subtracted cubes (default: False, use raw cubes).'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--weighting-scheme',
        choices=CubeCombiner.WEIGHTING_SCHEMES,
        default='inverse_variance',
        help=(
            'Weighting scheme for cube combination: '
            'inverse_variance (default) or clipped_muse '
            '(C-style clipped equal-weighting)'
        ),
    )
    parser.add_argument(
        '--disable-cr-clip',
        action='store_true',
        help='Disable cosmic-ray clipping for clipped_muse weighting'
    )
    parser.add_argument(
        '--reference-cube',
        dest='reference_cube',
        default=None,
        help=(
            'Path to a FITS cube to use as the alignment reference when computing '
            'per-exposure WCS offsets (default: first GREEN cube found)'
        )
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--wave-range-blue',
        nargs=2,
        type=float,
        metavar=('MIN_A', 'MAX_A'),
        default=None,
        help='White-light collapse range for blue channel in Angstrom (default: 3811 3823)'
    )
    parser.add_argument(
        '--wave-range-green',
        nargs=2,
        type=float,
        metavar=('MIN_A', 'MAX_A'),
        default=None,
        help='White-light collapse range for green channel in Angstrom (default: 6715 6727)'
    )
    parser.add_argument(
        '--wave-range-red',
        nargs=2,
        type=float,
        metavar=('MIN_A', 'MAX_A'),
        default=None,
        help='White-light collapse range for red channel in Angstrom (default: 7000 7100)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    collapse_wave_ranges = {}
    if args.wave_range_blue is not None:
        collapse_wave_ranges['blue'] = tuple(args.wave_range_blue)
    if args.wave_range_green is not None:
        collapse_wave_ranges['green'] = tuple(args.wave_range_green)
    if args.wave_range_red is not None:
        collapse_wave_ranges['red'] = tuple(args.wave_range_red)

    try:
        combiner = CubeCombiner(
            args.input_folder,
            use_skysub=args.use_skysub,
            output_folder=args.output_folder,
            overwrite=args.overwrite,
            collapse_wave_ranges=collapse_wave_ranges if collapse_wave_ranges else None,
            weighting_scheme=args.weighting_scheme,
            enable_cr_clip=(not args.disable_cr_clip),
            reference_cube=args.reference_cube,
        )
        combiner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
