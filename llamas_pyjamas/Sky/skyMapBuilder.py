#!/usr/bin/env python3
"""
Build a sky-region mask from a narrow-band (usually H-alpha) image of a LLAMAS datacube.

Adapted from redux/combine_cubes.py's cube-handling machinery (CubeCombiner._extract_wavelength_axis,
.collapse_cube_to_whitelight, ._mask_background_pixels, and its diagnostic-plot/FITS-saving
conventions), repurposed here for the opposite job: instead of finding a *source* to align on,
this collapses a cube to a narrow band around an emission line (e.g. a host galaxy's H-alpha),
then sigma-clips that image to flag pixels significantly off the robust background as
"source" (i.e. NOT sky) and everything else as sky. The output is a boolean mask FITS file in
the format Sky/skySelect.py's load_sky_map() expects (integer-valued, <=2 distinct levels,
sky = nonzero) -- i.e. directly usable as a `sky_map_file` for sky_selection_method='skymap'.

IMPORTANT: run this on a PLAIN reduced cube, never an already sky-subtracted one. The whole
point of the mask is to tell a later sky-subtraction pass where the sky-dominated fibres are;
a sky-subtracted cube has already had its own (unrelated) sky selection applied and removed,
which would make this circular -- the source region also stops standing out once the sky it
sits on top of has already been subtracted out from under it.

Usage:
    python skyMapBuilder.py <cube.fits> [-o <mask.fits>] [--line-center 6562.8] [--half-width 15]

Example (HD14943, H-alpha in the green channel, plain reduction cube):
    python skyMapBuilder.py cubes/HD14943/LLAMAS_..._cube_green.fits \\
        -o sky_maps/HD14943_green_Halpha_skymap.fits --plot-dir sky_maps/plots
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SkyMapBuilder:
    """Build a sky/source mask from a cube's narrow-band image around an emission line."""

    # Common rest-frame optical line centers (Angstrom), for the --line convenience flag.
    LINES = {
        'halpha': 6562.8,
        'hbeta': 4861.3,
        'hgamma': 4340.5,
        'oiii': 5006.8,
        'oii': 3727.3,
        'nii': 6583.5,
        'sii': 6716.4,
    }

    def __init__(self, sigma=3.0, sigma_lower=None, sigma_upper=None, maxiters=5,
                 stdfunc='mad_std'):
        """
        Parameters
        ----------
        sigma : float
            Symmetric clipping threshold, used when sigma_lower/sigma_upper are not
            given individually.
        sigma_lower, sigma_upper : float, optional
            Asymmetric clipping thresholds, if you want them (both default to
            `sigma`, i.e. plain symmetric clipping). Symmetric is the safer default:
            a source isn't just a positive excess -- the dark subtraction-residual
            trough that often rings a bright, poorly-subtracted point source is
            exactly the kind of pixel a sky model should also exclude, and a
            positive-only clip lets it straight through as "sky".
        maxiters : int
            Maximum sigma-clipping iterations (astropy.stats.sigma_clip).
        stdfunc : str
            Scatter estimator passed to sigma_clip; 'mad_std' (median-absolute-deviation
            based) is the same robust-to-outliers choice combine_cubes.py's
            _mask_background_pixels used (1.4826*MAD), rather than a plain std that
            outliers themselves would inflate.
        """
        self.sigma = float(sigma)
        self.sigma_lower = sigma_lower if sigma_lower is not None else float(sigma)
        self.sigma_upper = sigma_upper if sigma_upper is not None else float(sigma)
        self.maxiters = int(maxiters)
        self.stdfunc = stdfunc

    # ------------------------------------------------------------------
    # Cube I/O -- adapted from redux/combine_cubes.py's CubeCombiner
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_wavelength_axis(hdul):
        """Return wavelength axis in Angstrom for a cube HDUList."""
        if len(hdul) > 3 and hdul[3].data is not None and 'wavelength' in hdul[3].data.names:
            return np.asarray(hdul[3].data['wavelength'], dtype=np.float64)

        hdr = hdul[0].header
        nwave = hdul[0].data.shape[0]
        crval3 = hdr.get('CRVAL3')
        cdelt3 = hdr.get('CDELT3')
        crpix3 = hdr.get('CRPIX3', 1.0)
        if crval3 is None or cdelt3 is None:
            raise ValueError('Unable to derive wavelength axis: no WAVELENGTH extension '
                             'and missing CRVAL3/CDELT3')

        idx = np.arange(nwave, dtype=np.float64)
        wave = crval3 + (idx + 1.0 - crpix3) * cdelt3
        cunit3 = (hdr.get('CUNIT3') or '').strip().lower()
        if cunit3 in {'m', 'meter', 'metre'}:
            wave = wave * 1.0e10
        elif cunit3 in {'nm'}:
            wave = wave * 10.0
        return wave

    def build_narrowband_image(self, hdul, wave_min, wave_max):
        """Collapse a cube into a weighted narrow-band image over [wave_min, wave_max].

        Same VAR/WEIGHT-weighted collapse as combine_cubes.py's
        collapse_cube_to_whitelight, generalized from a fixed sky-line window to an
        arbitrary caller-supplied band.

        Returns
        -------
        image_2d : ndarray
        wcs_2d : astropy.wcs.WCS
        n_wave_used : int
            Number of wavelength planes actually inside the window (0 => no coverage).
        """
        flux = np.asarray(hdul[0].data, dtype=np.float32)
        var = np.asarray(hdul[1].data, dtype=np.float32) if len(hdul) > 1 and hdul[1].data is not None else None
        weight = np.asarray(hdul[2].data, dtype=np.float32) if len(hdul) > 2 and hdul[2].data is not None else None
        wave = self._extract_wavelength_axis(hdul)

        mask = (wave >= wave_min) & (wave <= wave_max)
        n_used = int(np.count_nonzero(mask))
        if n_used == 0:
            wcs_2d = WCS(hdul[0].header).celestial
            return np.full(flux.shape[1:], np.nan, dtype=np.float32), wcs_2d, 0

        flux_sel = flux[mask, :, :]
        weight_sel = weight[mask, :, :] if weight is not None else np.ones_like(flux_sel, dtype=np.float32)

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
        image_2d = np.divide(sum_wf, sum_w, out=np.full_like(sum_wf, np.nan, dtype=np.float32),
                             where=sum_w > 0)

        wcs_2d = WCS(hdul[0].header).celestial
        return image_2d, wcs_2d, n_used

    # ------------------------------------------------------------------
    # Sigma-clipping
    # ------------------------------------------------------------------
    def sigma_clip_sky_mask(self, image_2d):
        """Sigma-clip a narrow-band image to a boolean sky mask (True = sky).

        Iteratively estimates a robust background (median) and scatter (MAD-based
        std) over the finite pixels, then flags any pixel more than sigma_upper
        scatter-widths ABOVE the background (and, symmetrically by default,
        sigma_lower below) as source -- not sky. Unlike combine_cubes.py's
        _mask_background_pixels (a single-pass MAD threshold used to prep images for
        cross-correlation), this uses astropy's iterative sigma_clip so the
        background estimate itself isn't dragged by the source it's trying to find.

        Returns
        -------
        sky_mask : ndarray[bool]  (True = sky)
        stats : dict with 'median', 'std', 'threshold_lo', 'threshold_hi', 'n_sky', 'n_source', 'n_nan'
        """
        finite = np.isfinite(image_2d)
        clipped = sigma_clip(np.where(finite, image_2d, np.nan), sigma=self.sigma,
                             sigma_lower=self.sigma_lower, sigma_upper=self.sigma_upper,
                             maxiters=self.maxiters, cenfunc='median', stdfunc=self.stdfunc,
                             masked=True)

        # sigma_clip's own .mask is True for REJECTED (source) pixels; sky is the
        # unmasked, finite complement.
        sky_mask = finite & ~np.ma.getmaskarray(clipped)

        kept_vals = image_2d[sky_mask]
        median = float(np.nanmedian(kept_vals)) if kept_vals.size else float('nan')
        std = float(np.nanstd(kept_vals)) if kept_vals.size else float('nan')
        stats = {
            'median': median,
            'std': std,
            'threshold_lo': median - self.sigma_lower * std if np.isfinite(self.sigma_lower) else -np.inf,
            'threshold_hi': median + self.sigma_upper * std,
            'n_sky': int(np.count_nonzero(sky_mask)),
            'n_source': int(np.count_nonzero(finite & ~sky_mask)),
            'n_nan': int(np.count_nonzero(~finite)),
        }
        return sky_mask, stats

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    @staticmethod
    def save_mask_fits(sky_mask, wcs_2d, output_path, header_extra=None):
        """Write the sky mask as a load_sky_map()-compatible FITS file.

        Integer-valued (uint8), <=2 distinct levels (0/1) -- Sky/skySelect.py's
        _looks_like_mask() auto-detects this as a mask (sky = nonzero pixels), not a
        flux image to be thresholded.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header = wcs_2d.to_header()
        header['COMMENT'] = 'Sky mask: 1 = sky, 0 = source (sigma-clipped narrow-band image)'
        for key, val in (header_extra or {}).items():
            header[key] = val

        hdu = fits.PrimaryHDU(data=sky_mask.astype(np.uint8), header=header)
        hdu.writeto(output_path, overwrite=True)
        logger.info(f"Wrote sky mask: {output_path}")
        return output_path

    @staticmethod
    def save_diagnostic_plot(image_2d, sky_mask, stats, output_path, title=''):
        """3-panel diagnostic: narrow-band image, sky/source mask overlay, pixel histogram."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        finite_vals = image_2d[np.isfinite(image_2d)]
        if finite_vals.size:
            vmin, vmax = np.nanpercentile(finite_vals, [1, 99])
        else:
            vmin, vmax = None, None

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

        ax = axes[0]
        im0 = ax.imshow(image_2d, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title('Narrow-band image')
        ax.set_xlabel('X [pix]'); ax.set_ylabel('Y [pix]')
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1]
        overlay = np.zeros(image_2d.shape + (4,), dtype=np.float32)
        overlay[~sky_mask & np.isfinite(image_2d)] = (1.0, 0.15, 0.15, 0.85)   # source: red
        overlay[sky_mask] = (0.15, 0.55, 1.0, 0.55)                            # sky: blue
        overlay[~np.isfinite(image_2d)] = (0.0, 0.0, 0.0, 0.0)                 # no data: transparent
        ax.imshow(image_2d, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(overlay, origin='lower')
        ax.set_title(f"Sky mask (blue=sky {stats['n_sky']}, red=source {stats['n_source']})")
        ax.set_xlabel('X [pix]'); ax.set_ylabel('Y [pix]')

        ax = axes[2]
        if finite_vals.size:
            # Range off the robust median/std (not raw data percentiles): a bright
            # point source's narrow, extreme-valued core otherwise dwarfs the sky
            # population's actual width, crushing it into an unreadable spike.
            span = 6.0 * stats['std'] if np.isfinite(stats['std']) and stats['std'] > 0 else 1.0
            lo, hi = stats['median'] - span, stats['median'] + span
            ax.hist(finite_vals, bins=80, range=(lo, hi), color='0.5', label='all pixels')
            ax.set_xlim(lo, hi)
        ax.axvline(stats['median'], color='k', ls='-', lw=1.2, label=f"median={stats['median']:.3g}")
        if np.isfinite(stats['threshold_hi']):
            ax.axvline(stats['threshold_hi'], color='red', ls='--', lw=1.2,
                       label=f"upper clip={stats['threshold_hi']:.3g}")
        if np.isfinite(stats['threshold_lo']):
            ax.axvline(stats['threshold_lo'], color='orange', ls='--', lw=1.2,
                       label=f"lower clip={stats['threshold_lo']:.3g}")
        ax.set_title('Pixel value distribution')
        ax.set_xlabel('Narrow-band flux')
        ax.set_ylabel('N pixels')
        ax.legend(fontsize=8)

        fig.suptitle(title, fontsize=11)
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        logger.info(f"Wrote diagnostic plot: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, cube_path, line_center=None, line_name=None, half_width=15.0,
            wave_min=None, wave_max=None, redshift=0.0,
            output_mask=None, output_plot=None):
        """Build and save a sky mask from one cube.

        Wavelength window resolution (in priority order):
          1. explicit wave_min/wave_max
          2. line_name (looked up in self.LINES) or line_center, redshifted by
             (1+redshift) and padded by +/- half_width
          3. default: line_name='halpha'
        """
        cube_path = Path(cube_path)
        if wave_min is None or wave_max is None:
            if line_center is None:
                line_center = self.LINES[(line_name or 'halpha').lower()]
            obs_center = line_center * (1.0 + redshift)
            wave_min, wave_max = obs_center - half_width, obs_center + half_width

        logger.info(f"Cube: {cube_path.name}")
        logger.info(f"Narrow-band window: [{wave_min:.2f}, {wave_max:.2f}] Angstrom")

        with fits.open(cube_path) as hdul:
            cube_wave = self._extract_wavelength_axis(hdul)
            image_2d, wcs_2d, n_used = self.build_narrowband_image(hdul, wave_min, wave_max)

        if n_used == 0:
            raise ValueError(
                f"No wavelength samples in [{wave_min:.2f}, {wave_max:.2f}] A for {cube_path.name} "
                f"(cube covers {cube_wave.min():.2f}-{cube_wave.max():.2f} A). "
                "Pick a different line/channel, or pass an explicit --wave-min/--wave-max."
            )
        logger.info(f"Narrow-band image built from {n_used} wavelength planes")

        sky_mask, stats = self.sigma_clip_sky_mask(image_2d)
        logger.info(f"Sky mask: {stats['n_sky']} sky / {stats['n_source']} source / "
                    f"{stats['n_nan']} no-data pixels "
                    f"(median={stats['median']:.4g}, std={stats['std']:.4g}, "
                    f"upper clip={stats['threshold_hi']:.4g})")

        if output_mask is None:
            output_mask = cube_path.with_name(cube_path.stem + '_skymap.fits')
        header_extra = {
            'SKYMSRC': (str(cube_path.name)[:45], 'Source cube'),
            'SKYMWMIN': (float(wave_min), 'Narrow-band window min [Ang]'),
            'SKYMWMAX': (float(wave_max), 'Narrow-band window max [Ang]'),
            'SKYMSIG': (float(self.sigma), 'Sigma-clip threshold'),
            'SKYMNIT': (int(self.maxiters), 'Sigma-clip max iterations'),
            'SKYMNSKY': (stats['n_sky'], 'N sky pixels'),
            'SKYMNSRC': (stats['n_source'], 'N source pixels'),
        }
        self.save_mask_fits(sky_mask, wcs_2d, output_mask, header_extra=header_extra)

        if output_plot is None:
            output_plot = Path(output_mask).with_suffix('.png')
        self.save_diagnostic_plot(
            image_2d, sky_mask, stats, output_plot,
            title=f"{cube_path.name}\nwindow=[{wave_min:.1f},{wave_max:.1f}]A  "
                  f"sigma={self.sigma}  sky={stats['n_sky']}  source={stats['n_source']}")

        return output_mask, output_plot, stats


def main():
    parser = argparse.ArgumentParser(
        description='Build a sigma-clipped sky mask from a narrow-band (usually H-alpha) '
                    'cube image, in the format Sky/skySelect.load_sky_map() expects. '
                    'Run on a PLAIN reduced cube, not an already sky-subtracted one.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # H-alpha (default), auto window +/-15A, on a PLAIN (not sky-subtracted) cube
  python skyMapBuilder.py cubes/HD14943/LLAMAS_..._cube_green.fits

  # explicit line + redshift (e.g. an FRB host at z=0.1)
  python skyMapBuilder.py host_cube_red.fits --line halpha --redshift 0.1

  # explicit wavelength window
  python skyMapBuilder.py cube_red.fits --wave-min 7200 --wave-max 7230
        """
    )
    parser.add_argument('cube', help='Path to a LLAMAS datacube FITS file (plain reduction, '
                                     'not sky-subtracted)')
    parser.add_argument('-o', '--output', dest='output_mask', default=None,
                        help='Output mask FITS path (default: <cube>_skymap.fits)')
    parser.add_argument('--plot-dir', dest='plot_dir', default=None,
                        help='Directory for the diagnostic PNG (default: alongside the mask)')
    parser.add_argument('--line', dest='line_name', default=None,
                        choices=sorted(SkyMapBuilder.LINES),
                        help="Named line to center the narrow band on (default: halpha)")
    parser.add_argument('--line-center', dest='line_center', type=float, default=None,
                        help='Explicit rest-frame line center [Angstrom] (overrides --line)')
    parser.add_argument('--half-width', type=float, default=15.0,
                        help='Half-width of the narrow-band window [Angstrom] (default: 15)')
    parser.add_argument('--redshift', type=float, default=0.0,
                        help='Redshift applied to the line center before windowing (default: 0)')
    parser.add_argument('--wave-min', type=float, default=None,
                        help='Explicit window min [Angstrom] (overrides line/redshift/half-width)')
    parser.add_argument('--wave-max', type=float, default=None,
                        help='Explicit window max [Angstrom] (overrides line/redshift/half-width)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Symmetric sigma-clip threshold (default: 3.0)')
    parser.add_argument('--sigma-lower', type=float, default=None,
                        help='Lower sigma-clip threshold (default: --sigma, symmetric)')
    parser.add_argument('--sigma-upper', type=float, default=None,
                        help='Upper sigma-clip threshold (default: --sigma)')
    parser.add_argument('--maxiters', type=int, default=5,
                        help='Max sigma-clip iterations (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if 'skysub' in Path(args.cube).name.lower():
        logger.warning(
            f"'{Path(args.cube).name}' looks like a sky-subtracted cube. This tool is meant "
            "to run on a PLAIN reduced cube (pre sky-subtraction) -- the sky mask feeds INTO "
            "sky subtraction, so building it from an already sky-subtracted cube is circular. "
            "Proceeding anyway, but double-check the input."
        )

    builder = SkyMapBuilder(sigma=args.sigma, sigma_lower=args.sigma_lower,
                            sigma_upper=args.sigma_upper, maxiters=args.maxiters)

    output_mask = Path(args.output_mask) if args.output_mask else None
    output_plot = None
    if args.plot_dir is not None:
        stem = (output_mask.stem if output_mask else Path(args.cube).stem + '_skymap')
        output_plot = Path(args.plot_dir) / f"{stem}.png"

    try:
        builder.run(
            args.cube,
            line_center=args.line_center, line_name=args.line_name,
            half_width=args.half_width, wave_min=args.wave_min, wave_max=args.wave_max,
            redshift=args.redshift, output_mask=output_mask, output_plot=output_plot,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
