"""
astrometry_correction.py

Corrects the spatial WCS of LLAMAS line-flux maps (produced by
emission_line_fitter.py), which is geometrically wrong at the source: the
fiber-map's mechanical (x,y) grid is stamped straight onto RA/Dec with no
compensation for the true instrument-to-sky flip/rotation.

Correction recipe (empirically diagnosed against a DECaLS reference image):
    1. Transpose the data array (data.T -- reflect about the main diagonal).
       An earlier version of this recipe used flip-horizontal +
       rotate-90-clockwise instead; that produced plausible-looking (correct
       translation, correct point-source matches) but subtly WRONG results
       -- confirmed on FRB20240312D, where its Halpha contours reproduced
       the galaxy's overall light distribution but not its fine spiral-arm
       structure. Since the two other point sources it independently matched
       showed no obvious orientation error, and a single global correlation
       can't distinguish a translation-good/rotation-wrong solution from a
       fully-correct one, the actual transform was only pinned down by
       exhaustively testing all 8 dihedral (rotation x parity) combinations
       and comparing each one's Halpha contours against the DECaLS reference
       image by eye -- plain transpose was the one that visibly traced the
       real spiral structure. See orientation_diagnostics/ next to a
       target's *_combined folder for that comparison.
    2. Attach a FRESH, standard-orientation (North-up, East-left) WCS to the
       transformed array -- deliberately NOT derived from the original WCS's
       own rotation/orientation. Only the pixel scale magnitude, CRVAL
       (initial pointing guess), and CRPIX placement (swapped between the two
       axes, matching the transpose) are carried over from the original
       header; its rotational content is discarded entirely -- an earlier
       version of this module that derived the new WCS's CD matrix from the
       old CD via a Jacobian left a residual orientation error that its own
       self-check couldn't catch (it only verified self-consistency with the
       old, already-wrong WCS -- tautologically true by construction, not a
       check against reality).
    3. Cross-correlate the transposed CONTINUUM map (a fitted line-complex's
       linear-continuum level -- smooth, single-peaked) against the reference,
       reprojected onto its pixel grid, to get a robust coarse sky-centroid
       correction (stage 2). Fit a 2D Gaussian to the continuum map to get a
       centroid + FWHM (used to size the next step's search window), and plot
       the located centroid on the continuum map for visual verification.
    4. Refine the centroid with a SECOND, windowed cross-correlation of the
       (clumpier, noisier) line-flux map, restricted to a disk of radius
       1.5xFWHM (window diameter 3xFWHM) around the stage-2 solution (stage 3).
       Galaxies with multiple comparably-bright star-forming clumps, or only
       partially inside the LLAMAS FOV, can otherwise make an unrestricted
       line-map correlation lock onto the wrong clump -- the continuum-anchored
       coarse solution plus a tight window prevents that. No smoothing is used
       anywhere; the continuum tracer's own smoothness is what makes stage 2
       robust.
    5. Apply the SUM of the two stages' CRVAL corrections (they compose
       additively -- see correct_target) and plot the reference image with
       contours of the final corrected line map for visual verification.

Two sign/ordering conventions here are easy to get backwards and were
verified numerically before relying on them:
  - Pixel scale magnitude: scale = sqrt(|det(pixel_scale_matrix)|) (robust to
    CD-only or PC+CDELT headers).
  - After reprojecting onto the reference's pixel grid, the measured
    cross-correlation offset must be SUBTRACTED from CRVAL, not added.

Reference images (e.g. DECaLS cutouts) are not assumed to be stored
North-up/East-left on disk -- only their WCS is trusted. Correlation always
uses the reference's real WCS (orientation-agnostic by construction, since it
only calls pixel_to_world/reproject_interp). plot_alignment_diagnostic
additionally reprojects the reference image onto a fresh standard-orientation
WCS (_display_orientation_wcs) purely so the diagnostic PNG itself reads
North-up/East-left -- this is a display-only step and does not affect any
registration math.

Starts from the *_linemaps.fits products (not the full cube) for speed.

Author: Sunil Simha
"""
import argparse
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import correlate2d
from scipy.ndimage import label as _ndi_label
from reproject import reproject_interp
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_line_extension(fits_path, extname):
    """Read one 2D image extension + its WCS from a *_linemaps.fits file."""
    with fits.open(fits_path) as hdul:
        hdu = hdul[extname]
        data = hdu.data.astype(float).copy()
        header = hdu.header.copy()
    wcs = WCS(header)
    return data, wcs, header


def load_reference_image(path):
    """
    Read a generic WCS-bearing reference FITS image (e.g. a DECaLS cutout).
    No assumptions are made about the reference array's own orientation --
    only its WCS is trusted.
    """
    with fits.open(path) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        data = hdu.data.astype(float).copy()
        header = hdu.header.copy()
    wcs = WCS(header).celestial
    return data, wcs, header


def default_reference_path(linemaps_path):
    """Auto-detect {data_dir}/{target}_decals_r.fits next to a *_combined folder."""
    combined_dir = os.path.dirname(os.path.dirname(os.path.abspath(linemaps_path)))
    data_dir = os.path.dirname(combined_dir)
    target = os.path.basename(combined_dir).replace("_combined", "")
    candidate = os.path.join(data_dir, f"{target}_decals_r.fits")
    return candidate if os.path.exists(candidate) else None


def _infer_target_name(linemaps_path):
    combined_dir = os.path.dirname(os.path.dirname(os.path.abspath(linemaps_path)))
    return os.path.basename(combined_dir).replace("_combined", "")


# ----------------------------------------------------------------------
# Flip + rotate transform (data and WCS)
# ----------------------------------------------------------------------
def flip_rotate_data(data, flip_axis, rotate_ccw_deg):
    """
    Apply an optional axis flip followed by a 90-degree-multiple
    counterclockwise rotation. Both are REQUIRED, with no default, because the
    LLAMAS instrument-to-sky mapping is NOT one fixed transform across
    observations -- confirmed empirically: FRB20240312D needs flip_axis='x',
    rotate_ccw_deg=90 (equivalent to a plain transpose), while FRB20240210A
    needs a different combination. Per-target values belong in the calling
    code (e.g. a target registry), not as a default here. See module
    docstring and each target's own orientation_diagnostics/ folder.

    flip_axis: None, 'x' (mirror across a vertical line, negates the
    column/NAXIS1 index), or 'y' (mirror across a horizontal line, negates
    the row/NAXIS2 index). Applied BEFORE the rotation.
    rotate_ccw_deg: rotation in degrees, must be a multiple of 90 (0, 90, 180,
    or 270), using np.rot90's convention (positive = counterclockwise):
    np.rot90(data, k=rotate_ccw_deg // 90).

    Works on N-D arrays, transforming only the last two (spatial) axes -- so
    the same call also corrects a wavelength cube (nwave, ny, nx) in one shot,
    identically to correcting each wavelength slice on its own.
    """
    if flip_axis == "x":
        data = np.flip(data, axis=-1)
    elif flip_axis == "y":
        data = np.flip(data, axis=-2)
    elif flip_axis is not None:
        raise ValueError(f"flip_axis must be None, 'x', or 'y', got {flip_axis!r}")
    if rotate_ccw_deg % 90 != 0:
        raise ValueError(f"rotate_ccw_deg must be a multiple of 90, got {rotate_ccw_deg}")
    return np.rot90(data, k=(rotate_ccw_deg // 90) % 4, axes=(-2, -1)).copy()


def flip_rotate_wcs(old_wcs, old_shape, flip_axis, rotate_ccw_deg, self_check=True):
    """
    Build a WCS for flip_rotate_data(flip_axis, rotate_ccw_deg)'s output array
    with a fresh, standard (North-up, East-left) orientation -- see module
    docstring for why the original WCS's own rotation/orientation is
    deliberately NOT reused, only its pixel scale magnitude, CRVAL (initial
    guess), and CRPIX placement.

    CRPIX placement is derived NUMERICALLY rather than by hand-algebra (which
    has produced wrong formulas twice already in this module's history --
    see module docstring): a synthetic index grid is run through the exact
    same flip_rotate_data call to discover the (guaranteed affine, since it's
    just a permutation with signs) old-pixel <-> new-pixel relation, which is
    then solved for the CRPIX that keeps the same sky-pointing pixel
    self-consistent across the transform.
    """
    ny, nx = old_shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    t_idx = flip_rotate_data(idx, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)
    new_ny, new_nx = t_idx.shape
    old_row = (t_idx // nx).astype(float)
    old_col = (t_idx % nx).astype(float)
    new_rows, new_cols = np.indices((new_ny, new_nx))
    A = np.column_stack([new_rows.ravel(), new_cols.ravel(), np.ones(new_rows.size)])
    coeff_row, *_ = np.linalg.lstsq(A, old_row.ravel(), rcond=None)
    coeff_col, *_ = np.linalg.lstsq(A, old_col.ravel(), rcond=None)
    resid = max(np.abs(A @ coeff_row - old_row.ravel()).max(),
                np.abs(A @ coeff_col - old_col.ravel()).max())
    if resid > 1e-6:
        raise AssertionError(
            f"flip_rotate_data(flip_axis={flip_axis!r}, rotate_ccw_deg={rotate_ccw_deg}) "
            "is not affine -- unexpected"
        )
    a0, a1, a2 = coeff_row
    b0, b1, b2 = coeff_col

    old_crpix_x0 = old_wcs.wcs.crpix[0] - 1
    old_crpix_y0 = old_wcs.wcs.crpix[1] - 1
    M = np.array([[a0, a1], [b0, b1]])
    rhs = np.array([old_crpix_y0 - a2, old_crpix_x0 - b2])
    new_row, new_col = np.linalg.solve(M, rhs)
    new_crpix = [new_col + 1, new_row + 1]
    return _build_standard_wcs(old_wcs, new_crpix, self_check=self_check)


def _build_standard_wcs(reference_wcs, crpix, crval=None, self_check=True):
    """
    Shared builder for a fresh, standard-orientation (North-up, East-left) WCS
    that reuses only reference_wcs's pixel scale magnitude, CTYPE, CUNIT and
    RADESYS -- its rotational/orientation content is discarded entirely (see
    module docstring). crpix (and, optionally, crval -- defaults to
    reference_wcs's own) are supplied by the caller since they depend on what
    the WCS is being attached to (a flipped/rotated data array with a known
    affine CRPIX relation vs. a reprojection grid sized/centered for display).
    """
    scale = float(np.sqrt(abs(np.linalg.det(reference_wcs.pixel_scale_matrix))))
    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = crpix
    new_wcs.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]])  # standard N-up, E-left
    new_wcs.wcs.crval = crval if crval is not None else reference_wcs.wcs.crval
    new_wcs.wcs.ctype = reference_wcs.wcs.ctype
    new_wcs.wcs.cunit = [str(u) if str(u).strip() else "deg" for u in reference_wcs.wcs.cunit]
    new_wcs.wcs.radesys = getattr(reference_wcs.wcs, "radesys", "") or "ICRS"

    if self_check:
        _assert_standard_orientation(new_wcs)

    return new_wcs


def _display_orientation_wcs(wcs, shape):
    """
    Fresh standard-orientation WCS sharing wcs's pixel scale and array shape
    -- for DISPLAY only. Some reference images (e.g. DECaLS cutouts here) are
    stored on disk with a non-standard orientation (East to the right) while
    their WCS correctly records that flip; plotting them under their own WCS
    as the axes projection then shows RA increasing to the right, which is
    confusing even though everything is numerically correct.

    CRPIX/CRVAL are deliberately NOT copied from wcs verbatim: a cutout's
    CRPIX can sit far outside its own array (it's often inherited unchanged
    from the parent mosaic it was cut from -- confirmed for the DECaLS
    cutouts here, CRPIX2~1478 on a 230-pixel-tall array), so reusing it for a
    same-shape, differently-oriented output grid can place the reprojected
    footprint entirely off that grid (all-NaN). Anchoring on the array's own
    center pixel instead guarantees the output grid is centered on the same
    sky position the input data actually occupies, regardless of where CRPIX
    happened to point.
    """
    ny, nx = shape
    center_pix_0idx = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    center_sky = wcs.pixel_to_world(*center_pix_0idx)
    crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]  # 1-indexed FITS convention
    crval = [center_sky.ra.deg, center_sky.dec.deg]
    return _build_standard_wcs(wcs, crpix, crval=crval, self_check=True)


def _assert_standard_orientation(wcs, step_pix=10.0):
    """Confirm +row increases Dec (North up) and +col decreases RA (East left)."""
    x0, y0 = wcs.wcs.crpix[0] - 1, wcs.wcs.crpix[1] - 1  # FITS 1-indexed -> 0-indexed
    c0 = wcs.pixel_to_world(x0, y0)
    c_row = wcs.pixel_to_world(x0, y0 + step_pix)
    c_col = wcs.pixel_to_world(x0 + step_pix, y0)
    ddec = c_row.dec.deg - c0.dec.deg
    dra = c_col.ra.deg - c0.ra.deg
    if not ddec > 0:
        raise AssertionError(
            f"flip_rotate_wcs self-check FAILED: +row does not increase Dec (ddec={ddec:.3g} deg) "
            "-- North is not up"
        )
    if not dra < 0:
        raise AssertionError(
            f"flip_rotate_wcs self-check FAILED: +col does not decrease RA (dra={dra:.3g} deg) "
            "-- East is not to the left"
        )


# ----------------------------------------------------------------------
# Cross-correlation against a reference image
# (preprocessing adapted from llamas_fork/llamas_pyjamas/combine_cubes.py)
# ----------------------------------------------------------------------
def _prepare_for_correlation(image_2d):
    """Median-subtract and std-normalize finite pixels; non-finite -> 0."""
    image = np.asarray(image_2d, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("Cannot cross-correlate an image with no finite pixels")
    prepared = np.zeros_like(image, dtype=np.float32)
    finite_values = image[finite]
    median = float(np.nanmedian(finite_values))
    prepared[finite] = finite_values - median
    scale = float(np.nanstd(finite_values))
    if np.isfinite(scale) and scale > 0.0:
        prepared /= scale
    return prepared


def _mask_background_pixels(image_2d, sigma_multiplier=3.0):
    """MAD-based robust background threshold; sub-threshold pixels -> NaN."""
    image = np.asarray(image_2d, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("Cannot estimate a background threshold from an empty image")
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
    return masked


def find_calibrator_sources(reference_data, reference_wcs, exclude_coord,
                             exclude_radius_arcsec=5.0, sigma_threshold=8.0, min_npix=3):
    """
    Auto-detect compact, well-separated sources in a reference image, for
    use as orientation calibrators (see score_orientation_candidate):
    predicting where such a source SHOULD land under a candidate
    flip/rotate transform, then checking whether the cube's own continuum
    image actually shows something there, is a much better test of parity
    (mirror/chirality) than correlating the target alone -- a single,
    roughly-symmetric blob's correlation peak is nearly invariant under
    reflection about its own centroid, so it can't reliably discriminate a
    mirrored solution from a correct one; a second, spatially offset source
    can (confirmed empirically this session: for one target, an unrelated
    field star 12.9" from the target showed a flipped East-West sign under
    a wrong-parity transform, while the target itself -- a close QSO+galaxy
    blend too tight to resolve independently -- gave no such signal).

    MAD-based threshold (same recipe as _mask_background_pixels) ->
    scipy.ndimage.label connected components -> flux-weighted centroid per
    blob -> convert to a SkyCoord via reference_wcs.pixel_to_world -> drop
    anything within exclude_radius_arcsec of exclude_coord (the QSO/target
    region itself, handled separately).

    min_npix drops single/few-pixel detections (default: require at least
    3 connected pixels) -- a 1-pixel, marginal-significance blip is cheap
    noise that can spuriously "confirm" at nearly the same score under
    every candidate orientation (confirmed this session: a 1-pixel, 8-sigma
    blip tied all 8 dihedral candidates at an identical score, masking the
    real discriminating signal from a genuine 20-pixel calibrator).

    Returns a list of dicts (sorted by distance from exclude_coord,
    descending -- farther is a better lever arm for parity discrimination):
    {'coord': SkyCoord, 'peak_significance': float, 'npix': int,
    'distance_arcsec': float}. Empty list if nothing qualifies.
    """
    finite = np.isfinite(reference_data)
    if not np.any(finite):
        return []
    med = float(np.nanmedian(reference_data[finite]))
    mad = float(np.nanmedian(np.abs(reference_data[finite] - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(reference_data[finite]))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return []

    sig_map = np.where(finite, (reference_data - med) / sigma, -np.inf)
    labeled, n = _ndi_label(sig_map > sigma_threshold)

    calibrators = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_npix:
            continue
        w = reference_data[ys, xs] - med
        if not np.any(w > 0):
            continue
        cy = float(np.average(ys, weights=w))
        cx = float(np.average(xs, weights=w))
        coord = reference_wcs.pixel_to_world(cx, cy)
        dist_arcsec = coord.separation(exclude_coord).arcsec
        if dist_arcsec <= exclude_radius_arcsec:
            continue
        calibrators.append(dict(
            coord=coord, peak_significance=float(sig_map[ys, xs].max()),
            npix=int(len(ys)), distance_arcsec=dist_arcsec,
        ))
    calibrators.sort(key=lambda c: c["distance_arcsec"], reverse=True)
    return calibrators


def score_orientation_candidate(flipped_cont_data, flipped_cont_wcs, calibrators,
                                 aperture_radius_arcsec=5.0, sigma_threshold=5.0):
    """
    Score how well a candidate (flip_axis, rotate_ccw_deg)'s resulting,
    fully-corrected continuum map explains the calibrator sources found by
    find_calibrator_sources: for each calibrator, predict its pixel position
    under flipped_cont_wcs (world_to_pixel), and check whether the
    (UNMASKED) continuum data actually shows a locally significant peak
    there -- small-aperture peak flux compared to the whole array's own
    MAD-based background level/sigma. aperture_radius_arcsec defaults to
    5.0" (not e.g. 1.5", the FWHM-scale aperture that would be appropriate
    for a converged, already-precise position) since the "predicted"
    position here stacks several independent sources of slop -- the seed
    coordinate itself, refine_point_source_coord's own fit uncertainty, and
    this specific candidate's own stage-2/stage-3 registration residual
    (which can itself run to several arcsec, especially for a WRONG-parity
    candidate that never truly converges) -- confirmed empirically this
    session: a real, 20-pixel calibrator detection was missed entirely at
    1.5" (0.18 sigma local peak) but stood out clearly at 5" (9.0 sigma),
    cleanly separated from every other candidate's <5.1 sigma.

    Returns (score, n_confirmed, details): score is the significance-
    weighted sum of confirmed calibrators (so a strong, distant calibrator
    counts more than a marginal one) -- the primary, parity-sensitive
    ranking signal for auto_correct_target. n_confirmed is a plain count,
    for reporting. details is a per-calibrator list of dicts (coord,
    predicted pixel, local significance, confirmed bool) for diagnostics.
    """
    ny, nx = flipped_cont_data.shape
    finite = np.isfinite(flipped_cont_data)
    details = []
    if not np.any(finite) or not calibrators:
        return 0.0, 0, details

    med = float(np.nanmedian(flipped_cont_data[finite]))
    mad = float(np.nanmedian(np.abs(flipped_cont_data[finite] - med)))
    bg_sigma = 1.4826 * mad
    if not np.isfinite(bg_sigma) or bg_sigma <= 0.0:
        bg_sigma = float(np.nanstd(flipped_cont_data[finite]))

    pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(flipped_cont_wcs))) * 3600.0
    aperture_radius_pix = aperture_radius_arcsec / pix_scale_arcsec
    yy, xx = np.indices((ny, nx))

    score, n_confirmed = 0.0, 0
    for cal in calibrators:
        px, py = flipped_cont_wcs.world_to_pixel(cal["coord"])
        if not (np.isfinite(px) and np.isfinite(py) and 0 <= px < nx and 0 <= py < ny):
            details.append(dict(coord=cal["coord"], pixel=(px, py), local_sig=np.nan,
                                 confirmed=False, off_array=True))
            continue
        aperture = np.hypot(xx - px, yy - py) <= aperture_radius_pix
        aperture_vals = flipped_cont_data[aperture & finite]
        confirmed = False
        local_sig = np.nan
        if aperture_vals.size > 0 and np.isfinite(bg_sigma) and bg_sigma > 0.0:
            local_sig = (float(np.nanmax(aperture_vals)) - med) / bg_sigma
            confirmed = local_sig >= sigma_threshold
        if confirmed:
            score += cal["peak_significance"]
            n_confirmed += 1
        details.append(dict(coord=cal["coord"], pixel=(px, py), local_sig=local_sig,
                             confirmed=confirmed, off_array=False))
    return score, n_confirmed, details


def _mask_disk(image_2d, wcs, radius_arcsec, center_coord=None, keep_inside=False):
    """
    Set pixels either INSIDE (keep_inside=False, the default -- an
    "exclusion" disk, e.g. for a bright point-source contaminant) or OUTSIDE
    (keep_inside=True -- an "inclusion"/restrict-to-disk mask, e.g. for
    restricting a search to near the known pointing) a circular disk of
    radius_arcsec to NaN.

    center_coord: an astropy SkyCoord to center the disk on, converted to
    image_2d's own pixel grid via wcs.world_to_pixel -- same pattern as
    emission_line_fitter.load_halpha_peak_fraction_mask/
    _localization_ellipse_patch. Defaults to None, meaning the array's OWN
    geometric pixel center ((nx-1)/2, (ny-1)/2) -- identical to
    _mask_center_disk's original hardcoded behavior, valid whenever the
    array being masked (a reference cutout built centered on the target, or
    a flip/rotated LLAMAS array whose telescope pointing was centered on/
    near the true target) is itself centered on/near the position of
    interest.
    """
    ny, nx = image_2d.shape
    pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(wcs))) * 3600.0
    radius_pix = radius_arcsec / pix_scale_arcsec
    if center_coord is None:
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    else:
        cx, cy = wcs.world_to_pixel(center_coord)
    yy, xx = np.indices((ny, nx))
    inside = np.hypot(xx - cx, yy - cy) <= radius_pix
    masked = np.array(image_2d, dtype=np.float32, copy=True)
    masked[~inside if keep_inside else inside] = np.nan
    return masked


def _mask_center_disk(image_2d, wcs, radius_arcsec):
    """
    Set pixels within radius_arcsec of the array's own geometric center to
    NaN -- excludes a reference image's bright nuclear/core region from
    correlation when it has no emission-line counterpart (e.g. an evolved
    bulge or AGN continuum source with no cospatial star formation), so the
    correlation matches on genuine Hα-emitting structure (clumps) instead of
    being pulled toward a feature that structurally cannot appear in the line
    map at all. Valid because reference cutouts here are built centered
    exactly on the target coordinate (see default_reference_path/get_image
    usage), so the array's geometric center IS the target center. Thin
    wrapper around _mask_disk (center_coord=None, keep_inside=False) --
    kept as a stable name since cross_correlate_to_reference's
    mask_center_radius_arcsec already calls this directly.
    """
    return _mask_disk(image_2d, wcs, radius_arcsec)


def _safe_mask_background(image_2d, sigma_multiplier):
    try:
        masked = _mask_background_pixels(image_2d, sigma_multiplier)
        if not np.any(np.isfinite(masked)):
            raise ValueError("background threshold masked every pixel")
        return masked
    except ValueError as exc:
        warnings.warn(f"background masking failed ({exc}); falling back to unmasked image")
        return np.asarray(image_2d, dtype=np.float32)


def _mask_background_with_fallback(image_2d, sigma_thresholds=(5.0, 3.0), min_pixels=20):
    """
    Try each sigma threshold in order (strictest first), returning the first
    that retains at least min_pixels finite pixels, plus which threshold was
    actually used. Some quantities (e.g. a per-spaxel linear-fit continuum
    level over a narrow window) may simply never clear a strict cut for a
    given target -- relaxing in a fixed, reported order keeps that transparent
    rather than silently masking everything (or nothing).
    """
    for sigma in sigma_thresholds:
        try:
            masked = _mask_background_pixels(image_2d, sigma)
        except ValueError:
            continue
        if np.isfinite(masked).sum() >= min_pixels:
            return masked, sigma
    warnings.warn(
        f"background masking at all thresholds {sigma_thresholds} left fewer than "
        f"{min_pixels} pixels; using unmasked image"
    )
    return np.asarray(image_2d, dtype=np.float32), None


def reproject_to_reference(data, wcs, reference_wcs, reference_shape):
    """Reproject a 2D image onto a reference WCS/pixel grid (used only for correlation)."""
    reprojected, _footprint = reproject_interp((data, wcs), reference_wcs, shape_out=reference_shape)
    return reprojected


def cross_correlate_to_reference(target_data, target_wcs, reference_data, reference_wcs,
                                  bg_sigma_multiplier=3.0, edge_warn_frac=0.4,
                                  peak_sig_threshold=5.0,
                                  search_center_shift=None, search_radius_pix=None,
                                  mask_center_radius_arcsec=None):
    """
    Reproject target onto the reference's pixel grid, cross-correlate, and
    convert the pixel-shift peak into a sky offset (dra_deg, ddec_deg) using
    only the reference WCS (both points share its grid after reprojection).

    By default the correlation peak is a global argmax over the whole 'full'
    correlation surface. Pass search_center_shift=(shift_x0, shift_y0) (in
    reference-pixel units) and search_radius_pix to restrict the argmax to a
    disk of that radius -- useful when the target has multiple comparably
    bright features (e.g. star-forming clumps) that can cause a global search
    to lock onto the wrong one. peak_significance is computed from the SAME
    region the peak was searched in (the window when one is given, else the
    full corr_map) -- comparing a windowed peak against the full map's stats
    would be misleading, since 'full'-mode correlation surfaces are typically
    dominated by exact-zero boundary padding with degenerate (often zero) MAD,
    which can make an unremarkable in-window peak look spuriously significant.

    mask_center_radius_arcsec, if given, excludes a disk of that radius around
    the reference image's own geometric center (see _mask_center_disk) before
    masking/correlating -- for a reference with a bright nuclear/continuum
    region that has no emission-line counterpart in the target line map,
    which would otherwise pull the correlation toward matching a feature that
    structurally cannot be there.
    """
    ref_shape = reference_data.shape
    reproj = reproject_to_reference(target_data, target_wcs, reference_wcs, ref_shape)

    reference_data_for_corr = reference_data
    if mask_center_radius_arcsec is not None:
        reference_data_for_corr = _mask_center_disk(reference_data, reference_wcs, mask_center_radius_arcsec)

    ref_masked = _safe_mask_background(reference_data_for_corr, bg_sigma_multiplier)
    tgt_masked = _safe_mask_background(reproj, bg_sigma_multiplier)
    reference_corr = _prepare_for_correlation(ref_masked)
    target_corr = _prepare_for_correlation(tgt_masked)

    corr_map = correlate2d(target_corr, reference_corr, mode="full", boundary="fill", fillvalue=0.0)
    ny_ref, nx_ref = ref_shape

    warn_list = []
    corr_for_peak = corr_map
    if search_center_shift is not None:
        if search_radius_pix is None:
            raise ValueError("search_radius_pix is required when search_center_shift is given")
        shift_x0, shift_y0 = search_center_shift
        peak_y0 = shift_y0 + (ny_ref - 1)
        peak_x0 = shift_x0 + (nx_ref - 1)
        yy, xx = np.indices(corr_map.shape)
        within_window = np.hypot(xx - peak_x0, yy - peak_y0) <= search_radius_pix
        windowed = np.where(within_window, corr_map, np.nan)
        if np.any(np.isfinite(windowed)):
            corr_for_peak = windowed
        else:
            warn_list.append("search window contains no finite correlation values; falling back to global peak")

    peak_flat = int(np.nanargmax(corr_for_peak))
    peak_y, peak_x = np.unravel_index(peak_flat, corr_map.shape)
    peak_value = float(corr_map[peak_y, peak_x])

    shift_y = float(peak_y - (ny_ref - 1))
    shift_x = float(peak_x - (nx_ref - 1))

    if abs(shift_x) > edge_warn_frac * nx_ref or abs(shift_y) > edge_warn_frac * ny_ref:
        warn_list.append(
            f"peak shift ({shift_x:.0f},{shift_y:.0f}) px exceeds {edge_warn_frac * 100:.0f}% "
            f"of reference size ({nx_ref}x{ny_ref}); likely spurious"
        )

    finite_corr = corr_for_peak[np.isfinite(corr_for_peak)]
    corr_med = float(np.median(finite_corr))
    corr_mad = float(np.median(np.abs(finite_corr - corr_med)))
    corr_sigma = 1.4826 * corr_mad
    peak_significance = (peak_value - corr_med) / corr_sigma if corr_sigma > 0 else float("inf")
    if peak_significance < peak_sig_threshold:
        warn_list.append(
            f"peak significance {peak_significance:.1f} sigma below threshold "
            f"{peak_sig_threshold} sigma; alignment may be unreliable"
        )

    ref_center = ((nx_ref - 1) / 2.0, (ny_ref - 1) / 2.0)
    aligned_pix = (ref_center[0] + shift_x, ref_center[1] + shift_y)
    ref_center_sky = reference_wcs.pixel_to_world(*ref_center)
    aligned_sky = reference_wcs.pixel_to_world(*aligned_pix)
    dra_deg = (aligned_sky.ra.deg - ref_center_sky.ra.deg + 180.0) % 360.0 - 180.0
    ddec_deg = aligned_sky.dec.deg - ref_center_sky.dec.deg

    return dict(
        shift_x=shift_x, shift_y=shift_y, dra_deg=dra_deg, ddec_deg=ddec_deg,
        peak_value=peak_value, peak_significance=peak_significance,
        corr_map=corr_map, reprojected_target=reproj, warnings=warn_list,
    )


def cross_correlate_windowed_to_reference(target_data, target_wcs, reference_data, reference_wcs,
                                           fwhm_arcsec, window_radius_fwhm_factor=1.5,
                                           search_center_shift=(0.0, 0.0),
                                           bg_sigma_multiplier=3.0, edge_warn_frac=0.4,
                                           peak_sig_threshold=5.0, mask_center_radius_arcsec=None):
    """
    cross_correlate_to_reference restricted to a disk of radius
    window_radius_fwhm_factor * fwhm_arcsec around search_center_shift.
    Default factor 1.5 -> window DIAMETER = 3 * fwhm_arcsec.
    """
    ref_pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(reference_wcs))) * 3600.0
    search_radius_pix = (window_radius_fwhm_factor * fwhm_arcsec) / ref_pix_scale_arcsec
    return cross_correlate_to_reference(
        target_data, target_wcs, reference_data, reference_wcs,
        bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=edge_warn_frac,
        peak_sig_threshold=peak_sig_threshold,
        search_center_shift=search_center_shift, search_radius_pix=search_radius_pix,
        mask_center_radius_arcsec=mask_center_radius_arcsec,
    )


# ----------------------------------------------------------------------
# Continuum centroid + FWHM (drives the stage-3 search window)
# ----------------------------------------------------------------------
def _moment_based_size(x_data, y_data, w, x0, y0):
    """
    Flux-weighted second-moment size around a given centroid -- used as a
    fallback when the parametric Gaussian fit fails. Unlike a Gaussian fit,
    this doesn't require the light profile to turn over within the array, so
    it stays well-behaved for a galaxy that's cut off by a partial FOV.
    """
    var_x = np.average((x_data - x0) ** 2, weights=w)
    var_y = np.average((y_data - y0) ** 2, weights=w)
    fwhm_x = np.sqrt(var_x) * gaussian_sigma_to_fwhm
    fwhm_y = np.sqrt(var_y) * gaussian_sigma_to_fwhm
    return fwhm_x, fwhm_y


def measure_continuum_centroid(cont_data, bg_sigma_thresholds=(5.0, 3.0)):
    """
    Fit a 2D Gaussian (theta fixed at 0) to a continuum map on its own native
    pixel grid, giving both a centroid and a FWHM from a single fit -- the
    correlation peak alone isn't a well-defined "galaxy center" (it's a
    whole-image rigid-translation solution), and the FWHM is needed anyway
    to size the stage-3 search window, so one fit serves both purposes.

    Background masking tries bg_sigma_thresholds in order (strictest first,
    default 5 then 3 sigma) via _mask_background_with_fallback -- a stricter
    cut better isolates the true source from residual background structure
    (e.g. a flat-fielding/reduction banding pattern), but some continuum
    tracers may not have enough dynamic range to clear 5 sigma at all, so the
    threshold actually used is reported back (bg_sigma_used) for transparency.

    Falls back to a flux-weighted moment-based centroid/size (see
    _moment_based_size) if the Gaussian fit fails or runs away to its
    stddev bound -- which happens for a galaxy whose light is truncated by
    a partial FOV (cut off at a corner) and never turns over within the
    array, so an unconstrained fit can converge "successfully" (per
    astropy's own fit_info) to a degenerate, arbitrarily-elongated solution.
    The moment-based fallback doesn't require turnover, so it stays
    well-behaved in exactly that case -- this keeps stage 3 windowed rather
    than falling back to the unrestricted global search this whole change
    exists to avoid.

    Returns a dict with success (bool), method ("gaussian_fit" or "moments"),
    bg_sigma_used (float or None), x0_pix/y0_pix (0-indexed pixel coords of
    cont_data), fwhm_x_pix/fwhm_y_pix, fwhm_eff_pix (geometric mean, used for
    the search-window radius), amplitude, model (None for the moment fallback).

    Thin wrapper around _fit_2d_centroid -- see that function for the actual
    fit implementation, shared with refine_point_source_coord.
    """
    return _fit_2d_centroid(cont_data, bg_sigma_thresholds)


def _fit_2d_centroid(data_2d, bg_sigma_thresholds=(5.0, 3.0)):
    """Fit implementation extracted from measure_continuum_centroid (see its
    docstring for the algorithm) -- reused as-is by refine_point_source_coord
    to centroid a small windowed-out region instead of a whole continuum map."""
    ny, nx = data_2d.shape
    masked, bg_sigma_used = _mask_background_with_fallback(data_2d, bg_sigma_thresholds)
    yy, xx = np.mgrid[0:ny, 0:nx]
    finite = np.isfinite(masked)

    failure = dict(success=False, method=None, bg_sigma_used=bg_sigma_used,
                    x0_pix=np.nan, y0_pix=np.nan,
                    fwhm_x_pix=np.nan, fwhm_y_pix=np.nan, fwhm_eff_pix=np.nan,
                    amplitude=np.nan, model=None)
    if finite.sum() < 20:
        return failure

    x_data, y_data, z_data = xx[finite], yy[finite], masked[finite]
    w = np.clip(z_data, 0, None)
    if not np.any(w > 0):
        return failure
    x0_guess = np.average(x_data, weights=w)
    y0_guess = np.average(y_data, weights=w)

    # A real galaxy's FWHM shouldn't approach the array size; bounding stddev well
    # below nx/ny (rather than allowing it up to the full array) both keeps the fit
    # physically sensible and gives the boundary-hit check below something to catch.
    max_stddev = max(nx, ny) / 3.0
    model = models.Gaussian2D(
        amplitude=float(np.nanmax(z_data)), x_mean=x0_guess, y_mean=y0_guess,
        x_stddev=max(nx, ny) / 6.0, y_stddev=max(nx, ny) / 6.0, theta=0.0,
        bounds={"amplitude": (0, None), "x_mean": (0, nx - 1), "y_mean": (0, ny - 1),
                "x_stddev": (0.5, max_stddev), "y_stddev": (0.5, max_stddev)},
        fixed={"theta": True},
    )
    fitter = fitting.TRFLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_model = fitter(model, x_data, y_data, z_data)

    ok = bool(fitter.fit_info.get("success", False))
    params = [fit_model.x_mean.value, fit_model.y_mean.value,
              fit_model.x_stddev.value, fit_model.y_stddev.value]
    near_bound = (fit_model.x_stddev.value > 0.95 * max_stddev
                  or fit_model.y_stddev.value > 0.95 * max_stddev)
    if not (ok and np.all(np.isfinite(params))) or near_bound:
        fwhm_x, fwhm_y = _moment_based_size(x_data, y_data, w, x0_guess, y0_guess)
        return dict(
            success=True, method="moments", bg_sigma_used=bg_sigma_used,
            x0_pix=float(x0_guess), y0_pix=float(y0_guess),
            fwhm_x_pix=float(fwhm_x), fwhm_y_pix=float(fwhm_y),
            fwhm_eff_pix=float(np.sqrt(fwhm_x * fwhm_y)),
            amplitude=float(np.nanmax(z_data)), model=None,
        )

    fwhm_x = fit_model.x_stddev.value * gaussian_sigma_to_fwhm
    fwhm_y = fit_model.y_stddev.value * gaussian_sigma_to_fwhm
    return dict(
        success=True, method="gaussian_fit", bg_sigma_used=bg_sigma_used,
        x0_pix=float(fit_model.x_mean.value), y0_pix=float(fit_model.y_mean.value),
        fwhm_x_pix=float(fwhm_x), fwhm_y_pix=float(fwhm_y),
        fwhm_eff_pix=float(np.sqrt(fwhm_x * fwhm_y)),
        amplitude=float(fit_model.amplitude.value), model=fit_model,
    )


def refine_point_source_coord(reference_data, reference_wcs, seed_coord,
                               search_radius_arcsec=3.0, bg_sigma_thresholds=(5.0, 3.0)):
    """
    Refine a rough catalog sky coordinate (seed_coord -- e.g. a quasar's
    literature RA/Dec) into a precise centroid on a TRUSTED reference image
    (e.g. a Pan-STARRS cutout): restrict to a small disk of
    search_radius_arcsec around seed_coord (_mask_disk, keep_inside=True),
    then reuse _fit_2d_centroid -- the identical MAD-background-cut +
    2D-Gaussian-fit-with-moment-fallback recipe measure_continuum_centroid
    uses, so the two stay consistent.

    This is a SMALL LOCAL refinement of an already-approximately-known
    position, not a general multi-source detector: it will happily converge
    on the locally brightest thing within search_radius_arcsec, so keep that
    tight relative to how good seed_coord already is.

    Returns a dict: everything _fit_2d_centroid returns, plus 'coord' (a
    SkyCoord, or None if the fit failed) in reference_data's own frame.
    """
    windowed = _mask_disk(reference_data, reference_wcs, search_radius_arcsec,
                           center_coord=seed_coord, keep_inside=True)
    fit = _fit_2d_centroid(windowed, bg_sigma_thresholds)
    coord = reference_wcs.pixel_to_world(fit["x0_pix"], fit["y0_pix"]) if fit["success"] else None
    return dict(fit, coord=coord)


def plot_continuum_centroid_diagnostic(cont_data, cont_wcs, centroid, target_name, out_path,
                                        bg_sigma_multiplier=5.0):
    """WCS-projected continuum map -- background-clipped at bg_sigma_multiplier
    (the SAME threshold and mask used by measure_continuum_centroid, so the
    display matches what actually drove the fit, e.g. suppressing a
    flat-fielding/reduction banding pattern that survives a looser cut) --
    with a marker at the fitted centroid and a dashed circle of radius
    fwhm_eff_pix (the same FWHM sizing stage 3's search window).

    bg_sigma_multiplier=None means no threshold cleared the fallback chain
    (see measure_continuum_centroid's bg_sigma_used) -- show the raw
    (unmasked) map in that case rather than erroring.
    """
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": cont_wcs})
    masked = (cont_data if bg_sigma_multiplier is None
              else _safe_mask_background(cont_data, bg_sigma_multiplier))
    _, med, std = sigma_clipped_stats(masked, sigma=3.0)
    norm = ImageNormalize(stretch=LinearStretch(), vmin=med - std, vmax=med + 8 * std)
    ax.imshow(np.nan_to_num(masked, nan=med - std), origin="lower", cmap="gray_r", norm=norm)

    if centroid["success"]:
        ax.plot(centroid["x0_pix"], centroid["y0_pix"], marker="+", color="red",
                ms=18, mew=2, transform=ax.get_transform("pixel"))
        circ = plt.Circle((centroid["x0_pix"], centroid["y0_pix"]), centroid["fwhm_eff_pix"],
                           edgecolor="red", facecolor="none", ls="--", lw=1.0,
                           transform=ax.get_transform("pixel"))
        ax.add_patch(circ)
    else:
        ax.text(0.5, 0.5, "centroid fit failed", color="red", ha="center", va="center",
                transform=ax.transAxes)

    method_note = f" [{centroid['method']}]" if centroid["success"] else ""
    sigma_note = f"{bg_sigma_multiplier:.0f}sigma-clipped" if bg_sigma_multiplier is not None else "unmasked"
    ax.set_title(f"{target_name} -- continuum centroid (stage 2), {sigma_note}{method_note}")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.grid(color="lightgray", lw=0.5, ls="--")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def apply_crval_shift(wcs, dra_deg, ddec_deg):
    """Apply a measured (dra, ddec) correction to CRVAL1/CRVAL2 (subtracted -- see module docstring)."""
    new_wcs = wcs.deepcopy()
    new_wcs.wcs.crval = [wcs.wcs.crval[0] - dra_deg, wcs.wcs.crval[1] - ddec_deg]
    return new_wcs


# ----------------------------------------------------------------------
# Output FITS
# ----------------------------------------------------------------------
def write_corrected_fits(linemaps_path, dra_deg, ddec_deg, old_shape, target_name, outdir,
                          flip_axis, rotate_ccw_deg):
    """Apply the same flip/rotate/CRVAL-shift to every line extension; NaNs pass through untouched."""
    hdus = [fits.PrimaryHDU()]
    with fits.open(linemaps_path) as hdul:
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.ImageHDU) or hdu.data is None:
                continue
            if hdu.data.shape != old_shape:
                warnings.warn(f"skipping {hdu.name}: shape {hdu.data.shape} != {old_shape}")
                continue
            data_t = flip_rotate_data(hdu.data, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)
            wcs_t = flip_rotate_wcs(WCS(hdu.header), old_shape,
                                     flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg, self_check=False)
            wcs_t = apply_crval_shift(wcs_t, dra_deg, ddec_deg)
            new_hdr = wcs_t.to_header()
            new_hdr["WCSCORR"] = (True, "Astrometry corrected (flip+rotate+CRVAL)")
            new_hdr["WCSFLIP"] = (str(flip_axis), "flip_axis used (see flip_rotate_data)")
            new_hdr["WCSROT"] = (rotate_ccw_deg, "[deg CCW] rotation used (see flip_rotate_data)")
            new_hdr["WCSDRA"] = (dra_deg, "[deg] Applied CRVAL1 correction")
            new_hdr["WCSDDEC"] = (ddec_deg, "[deg] Applied CRVAL2 correction")
            hdus.append(fits.ImageHDU(data_t, header=new_hdr, name=hdu.name))

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{target_name}_linemaps_corrected.fits")
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    print(f"Wrote {out_path}")
    return out_path


def _build_cube_wcs3d(new_wcs2d, spectral_header):
    """
    Combine a corrected 2D celestial WCS with a cube header's UNCHANGED
    spectral axis (axis 3: CRPIX3/CRVAL3/CDELT3/CTYPE3/CUNIT3) into a fresh
    3D WCS -- no spatial-spectral cross terms, matching the physical setup
    (wavelength is independent of spaxel position).

    The effective spectral step is read via pixel_scale_matrix, NOT a raw
    spectral_header["CDELT3"] dict lookup: a header produced by this same
    function's own to_header() round-trips a CD-matrix WCS through the
    PC+CDELT convention, leaving CDELT3=1.0 as a vestigial placeholder with
    the real step stored in PC3_3 -- a naive CDELT3 lookup silently reads
    1.0 (meters, i.e. ~1e10x too large) if this is ever called on an
    already-corrected header (e.g. correct_cube invoked a second time on
    its own output). pixel_scale_matrix combines cdelt*pc correctly under
    either convention, so it's correct for both a genuinely raw header and
    an already-corrected one.
    """
    spectral_scale = WCS(spectral_header).pixel_scale_matrix[2, 2]
    new_wcs3d = WCS(naxis=3)
    new_wcs3d.wcs.crpix = [new_wcs2d.wcs.crpix[0], new_wcs2d.wcs.crpix[1], spectral_header["CRPIX3"]]
    cd = np.eye(3)
    cd[:2, :2] = new_wcs2d.wcs.cd
    cd[2, 2] = spectral_scale
    new_wcs3d.wcs.cd = cd
    new_wcs3d.wcs.crval = [new_wcs2d.wcs.crval[0], new_wcs2d.wcs.crval[1], spectral_header["CRVAL3"]]
    new_wcs3d.wcs.ctype = [new_wcs2d.wcs.ctype[0], new_wcs2d.wcs.ctype[1], spectral_header["CTYPE3"]]
    new_wcs3d.wcs.cunit = [new_wcs2d.wcs.cunit[0], new_wcs2d.wcs.cunit[1], spectral_header["CUNIT3"]]
    new_wcs3d.wcs.radesys = new_wcs2d.wcs.radesys
    return new_wcs3d


def _replace_wcs_in_header(old_header, new_wcs):
    """
    Strip exactly the WCS cards astropy would generate for the header's
    existing WCS (harmless no-op if it has none, e.g. this cube format's VAR/
    WEIGHT extensions, which carry no WCS of their own and rely on FLUX's),
    then merge in new_wcs's cards. Avoids leaving stale PC+CDELT-style
    keywords that would conflict with a freshly built CD-matrix WCS, without
    needing to guess/hardcode every possible WCS keyword name.
    """
    new_header = old_header.copy()
    old_wcs = WCS(old_header)
    for k in old_wcs.to_header(relax=True):
        if k in new_header:
            del new_header[k]
    new_header.update(new_wcs.to_header())
    return new_header


def correct_cube(cube_path, corrected_linemaps_path, outdir=None, cube_label="green"):
    """
    Apply the SAME flip/rotate/CRVAL-shift correction already computed for a
    target's line maps (read back from corrected_linemaps_path's WCSFLIP/
    WCSROT/WCSDRA/WCSDDEC header keywords -- see write_corrected_fits) to its
    full 3D wavelength cube. Every spatial-image extension (FLUX, VAR,
    WEIGHT, ...) gets the identical spatial data transform and a freshly
    built WCS combining the corrected 2D celestial solution with the cube's
    unchanged spectral axis. Reading the correction back from the already-
    corrected line maps (rather than re-deriving or re-specifying it) is what
    guarantees the cube and line maps end up on the exact same corrected sky
    frame.
    """
    with fits.open(corrected_linemaps_path) as lm_hdul:
        ref_hdr = next(hdu.header for hdu in lm_hdul if isinstance(hdu, fits.ImageHDU) and hdu.data is not None)
        flip_axis_hdr = ref_hdr["WCSFLIP"]
        flip_axis = None if flip_axis_hdr == "None" else flip_axis_hdr
        rotate_ccw_deg = int(ref_hdr["WCSROT"])
        dra_deg = float(ref_hdr["WCSDRA"])
        ddec_deg = float(ref_hdr["WCSDDEC"])
    outdir = outdir or os.path.dirname(os.path.abspath(cube_path))

    with fits.open(cube_path) as hdul:
        primary_header = hdul[0].header
        old_spatial_shape = hdul[0].data.shape[-2:]
        old_wcs2d = WCS(primary_header).celestial
        new_wcs2d = flip_rotate_wcs(old_wcs2d, old_spatial_shape,
                                     flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg, self_check=False)
        new_wcs2d = apply_crval_shift(new_wcs2d, dra_deg, ddec_deg)
        new_wcs3d = _build_cube_wcs3d(new_wcs2d, primary_header)

        hdus = []
        for i, hdu in enumerate(hdul):
            if hdu.data is None or hdu.data.ndim != 3:
                hdus.append(hdu.copy())
                continue
            if hdu.data.shape[-2:] != old_spatial_shape:
                warnings.warn(f"skipping {hdu.name}: spatial shape {hdu.data.shape[-2:]} != {old_spatial_shape}")
                hdus.append(hdu.copy())
                continue

            data_t = flip_rotate_data(hdu.data, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)
            new_header = _replace_wcs_in_header(hdu.header, new_wcs3d)
            new_header["WCSCORR"] = (True, "Astrometry corrected (flip+rotate+CRVAL)")
            new_header["WCSFLIP"] = (str(flip_axis), "flip_axis used (see flip_rotate_data)")
            new_header["WCSROT"] = (rotate_ccw_deg, "[deg CCW] rotation used (see flip_rotate_data)")
            new_header["WCSDRA"] = (dra_deg, "[deg] Applied CRVAL1 correction")
            new_header["WCSDDEC"] = (ddec_deg, "[deg] Applied CRVAL2 correction")

            if i == 0:
                hdus.append(fits.PrimaryHDU(data_t, header=new_header))
            else:
                hdus.append(fits.ImageHDU(data_t, header=new_header, name=hdu.name))

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"LLAMAS_combined_cube_{cube_label}_corrected.fits")
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    print(f"Wrote {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Diagnostic plot
# ----------------------------------------------------------------------
def plot_alignment_diagnostic(reference_data, reference_wcs, line_data, corrected_line_wcs,
                               target_name, line_name, out_path,
                               contour_sigma_multiples=(3, 6, 14, 30, 65, 140, 300),
                               vmin_sigma_floor=10.0, qso_coord=None):
    # Display-only reprojection: the reference image on disk may not be
    # North-up/East-left (see _display_orientation_wcs) even though its WCS
    # correctly records whatever orientation it actually has. Statistics
    # (vmin/vmax) are computed from the original reference_data either way --
    # reprojection is a resampling, not a value change.
    display_wcs = _display_orientation_wcs(reference_wcs, reference_data.shape)
    display_data = reproject_to_reference(reference_data, reference_wcs, display_wcs, reference_data.shape)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": display_wcs})
    _, ref_med, ref_std = sigma_clipped_stats(reference_data, sigma=3.0)
    vmax = float(np.nanmax(reference_data))
    # Log stretch with a floor above the background level (LogStretch needs
    # positive values), vmax near the image peak. A higher vmin_sigma_floor
    # clips more of the smooth diffuse galaxy light, revealing bright compact
    # structure instead of a smooth blob; a lower one keeps more of the
    # extended, fainter light visible at the cost of some background noise
    # texture -- the right choice depends on whether you're checking gross
    # alignment against faint extended structure or fine alignment against a
    # bright compact feature. The floor is capped at half the image's actual
    # peak significance: a fixed high floor (e.g. 10 sigma) works for a
    # high-contrast reference (e.g. a bright galaxy core well above
    # background) but can exceed vmax entirely for a fainter/lower-contrast
    # one (confirmed for FRB20250227A's g-band cutout, peak only ~9.5 sigma
    # above background) -- ImageNormalize requires vmin < vmax, so this keeps
    # the floor meaningful instead of erroring or silently degrading to a
    # near-empty display.
    peak_sigma = (vmax - ref_med) / ref_std if ref_std > 0 else 10.0
    floor_sigma = min(vmin_sigma_floor, 0.5 * peak_sigma) if peak_sigma > 0 else 1.0
    vmin = ref_med + floor_sigma * ref_std
    norm = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
    ax.imshow(display_data, origin="lower", cmap="Greys_r", norm=norm)

    _, med, std = sigma_clipped_stats(line_data, sigma=3.0)
    levels = sorted(med + n * std for n in contour_sigma_multiples)
    plot_data = np.where(np.isfinite(line_data), line_data, -np.inf)
    # Drop any level with no pixel reaching it before calling ax.contour: astropy's
    # WCS contour transform (transform_contour_set_inplace) raises an IndexError on
    # a fully-empty contour collection rather than just drawing nothing, and a
    # per-spaxel fit-flux map (mostly hard zeros outside its own narrowband fit
    # mask, real signal only in a small compact population) can push
    # sigma_clipped_stats' std well above the map's actual peak for a faint
    # target -- confirmed for FRB20211127I's Hbeta map, where even the lowest
    # (3-sigma) level had zero qualifying pixels. Skip contouring entirely (with
    # a warning) if every level is empty, rather than crashing the whole plot.
    levels = [lv for lv in levels if np.any(plot_data >= lv)]
    if levels:
        ax.contour(plot_data, levels=levels, transform=ax.get_transform(corrected_line_wcs),
                   colors="red", linewidths=1.0)
    else:
        warnings.warn(f"[{target_name}] {line_name}: no contour level has any qualifying pixel "
                       "(map too faint/sparse relative to contour_sigma_multiples); skipping contours")

    if qso_coord is not None:
        # Plotted directly in the sky/world frame (get_transform("world")), so this
        # is correct regardless of display_wcs's own reprojection -- an at-a-glance
        # parity check: this marker should land exactly on the visible point source
        # in the reference image; if the line-map contours end up mirrored to the
        # wrong side of it, that's a chirality (flip_axis) error, not a translation one.
        ax.scatter(qso_coord.ra.deg, qso_coord.dec.deg, transform=ax.get_transform("world"),
                   marker="x", s=150, linewidths=2.2, color="cyan", zorder=6,
                   label="QSO (refined)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.7)

    ax.set_title(f"{target_name} -- {line_name} astrometry check")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.grid(color="lightgray", lw=0.5, ls="--")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def correct_target(linemaps_path, flip_axis, rotate_ccw_deg, reference_path=None, target_name=None,
                    line="Halpha",
                    use_continuum=True, continuum_extname="CONTINUUM_IMG",
                    search_radius_arcsec=None, mask_center_radius_arcsec=None,
                    qso_coord=None, qso_exclude_radius_arcsec=2.0,
                    qso_refine=True, qso_refine_search_radius_arcsec=3.0,
                    qso_refine_bg_sigma_thresholds=(5.0, 3.0),
                    pointing_center_radius_arcsec=None,
                    outdir=None, verify_after=True, make_plot=True,
                    bg_sigma_multiplier=3.0, centroid_bg_sigma_thresholds=(5.0, 3.0),
                    edge_warn_frac=0.4, peak_sig_threshold=5.0,
                    window_radius_fwhm_factor=1.5,
                    contour_sigma_multiples=(3, 6, 14, 30, 65, 140, 300),
                    astrom_check_vmin_sigma=10.0):
    """
    flip_axis/rotate_ccw_deg select the initial instrument-to-sky data
    transform (see flip_rotate_data) -- REQUIRED, with no default, since they
    must be determined per target, not assumed universal: confirmed
    empirically that FRB20240312D and FRB20240210A need different values (see
    module docstring and each target's orientation_diagnostics/ folder).
    Per-target values (and the rest of a target's bespoke settings --
    use_continuum, search_radius_arcsec, mask_center_radius_arcsec, etc.)
    belong in the calling code (e.g. a target registry in a project-specific
    driver script), not as defaults in this general-purpose function.

    Three-stage registration (use_continuum=True, the default):
      1. flip_rotate_data/flip_rotate_wcs on both the line map and continuum map.
      2. Global (unwindowed) cross-correlation of the CONTINUUM map against the
         reference -- smooth/single-peaked, robust coarse centroid. The
         continuum map (continuum_extname, default CONTINUUM_IMG -- a
         broadband image integrated over a line-free wavelength window, see
         emission_line_fitter.compute_broadband_continuum) has much better
         S/N than any single per-spaxel line-fit's linear-continuum parameter.
      3. Windowed cross-correlation of the (clumpier) line map, restricted to a
         disk of radius window_radius_fwhm_factor * continuum-FWHM around the
         stage-2 solution, so it can't wander off to a different clump.
    The two stages' CRVAL corrections are additive (see comment below).

    use_continuum=False falls back to a SINGLE global (unwindowed unless
    search_radius_arcsec is given) correlation of the line map itself against
    the reference -- no continuum stage, no centroid/FWHM measurement, no
    continuum diagnostic plot, and continuum_extname is never read (the FITS
    doesn't even need that extension). This is the pre-continuum-anchoring
    behavior, useful when the continuum map's own S/N is too poor to reliably
    isolate the target galaxy -- e.g. a compact reduction artifact getting
    misidentified as "the galaxy" and dragging stage 2 to a spurious large
    offset, confirmed for FRB20240210A via its continuum centroid diagnostic
    plot.

    search_radius_arcsec, if given, restricts the correlation (stage 2's
    continuum correlation when use_continuum=True, else the single line-map
    correlation) to a disk of that radius around ZERO shift relative to the
    reference image's own center -- i.e. it trusts the reference was already
    centered close to the true target position (true here since
    default_reference_path/the reference cutout is built centered on the
    target coordinate) and the original telescope pointing is only off by a
    modest amount. Needed for FRB20250227A: its line map has a real, diffuse,
    low-contrast host galaxy sitting almost exactly at the reference center,
    but an unrestricted correlation instead locks onto a much higher-contrast
    foreground star ~32 arcsec away. A tight radius (e.g. 15") that excludes
    the star's separation but comfortably covers ordinary pointing error
    resolves this.

    mask_center_radius_arcsec, if given, excludes a disk of that radius
    around the reference image's own geometric center from the correlation
    (see _mask_center_disk) -- for a reference whose bright nuclear/continuum
    core has no counterpart in the emission-line map at all (e.g. an evolved
    bulge or AGN continuum source with no cospatial star formation), which
    would otherwise pull the correlation toward a feature the line map
    structurally cannot show, instead of the real (fainter, clumpier)
    star-forming regions actually present in both images.

    qso_coord, if given (a SkyCoord -- e.g. a catalog quasar position), is
    refined to a precise reference-image position (refine_point_source_coord,
    unless qso_refine=False) and then excluded (a disk of
    qso_exclude_radius_arcsec, see _mask_disk) from BOTH the reference and
    target sides of every correlation/centroiding step below, including the
    continuum centroid fit itself (measure_continuum_centroid has no spatial
    prior at all -- see module investigation notes -- and will otherwise
    happily lock onto a bright point-source contaminant instead of the
    intended target). pointing_center_radius_arcsec, if given, additionally
    restricts every such step to a disk of that radius around each array's
    own geometric center -- catches a bright, unrelated interloper elsewhere
    in the field of view that mask_center_radius_arcsec/qso_coord can't (it
    isn't the QSO and isn't at the array center). Both are no-ops (None,
    the default) for every existing single-source target -- fully
    backward compatible.
    """
    target_name = target_name or _infer_target_name(linemaps_path)
    reference_path = reference_path or default_reference_path(linemaps_path)
    if reference_path is None:
        raise ValueError(f"No DECaLS default found for {target_name}; pass reference_path explicitly")
    outdir = outdir or os.path.dirname(os.path.abspath(linemaps_path))

    reg_data, reg_wcs, _ = load_line_extension(linemaps_path, f"{line.upper()}_FLUX")
    old_shape = reg_data.shape
    flipped_reg = flip_rotate_data(reg_data, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)
    flipped_wcs = flip_rotate_wcs(reg_wcs, old_shape, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)
    # continuum (if used) shares the same pixel grid/WCS
    ref_data, ref_wcs, _ = load_reference_image(reference_path)
    ref_pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(ref_wcs))) * 3600.0
    search_kwargs = {}
    if search_radius_arcsec is not None:
        search_kwargs = dict(search_center_shift=(0.0, 0.0),
                              search_radius_pix=search_radius_arcsec / ref_pix_scale_arcsec)

    qso_refined_coord = None
    if qso_coord is not None:
        if qso_refine:
            refined = refine_point_source_coord(
                ref_data, ref_wcs, qso_coord,
                search_radius_arcsec=qso_refine_search_radius_arcsec,
                bg_sigma_thresholds=qso_refine_bg_sigma_thresholds,
            )
            if refined["success"]:
                qso_refined_coord = refined["coord"]
            else:
                warnings.warn(f"[{target_name}] QSO coordinate refinement failed; "
                               "using the given coordinate as-is for exclusion masking")
                qso_refined_coord = qso_coord
        else:
            qso_refined_coord = qso_coord

    def _restrict_for_corr(arr, wcs):
        """Apply QSO-exclusion + pointing-center-inclusion masking (each a
        no-op unless its parameter is set) to an array before it feeds
        centroiding/correlation. wcs must be the array's CURRENT WCS at the
        point of the call (its CRVAL may already have been shifted by an
        earlier stage), since qso_refined_coord's pixel position within arr
        depends on it."""
        out = arr
        if qso_refined_coord is not None:
            out = _mask_disk(out, wcs, qso_exclude_radius_arcsec, center_coord=qso_refined_coord)
        if pointing_center_radius_arcsec is not None:
            out = _mask_disk(out, wcs, pointing_center_radius_arcsec, keep_inside=True)
        return out

    ref_data_for_corr = _restrict_for_corr(ref_data, ref_wcs)

    if not use_continuum:
        warnings.warn(f"[{target_name}] use_continuum=False: skipping continuum stage, "
                       "using a single global correlation on the line map")
        corr = cross_correlate_to_reference(
            _restrict_for_corr(flipped_reg, flipped_wcs), flipped_wcs, ref_data_for_corr, ref_wcs,
            bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=edge_warn_frac,
            peak_sig_threshold=peak_sig_threshold, mask_center_radius_arcsec=mask_center_radius_arcsec,
            **search_kwargs,
        )
        for w in corr["warnings"]:
            warnings.warn(f"[{target_name}] single-stage (line map): {w}")
        dra2 = ddec2 = 0.0
        dra3, ddec3 = corr["dra_deg"], corr["ddec_deg"]
        stage2_wcs = flipped_wcs
        centroid = dict(success=False, method=None, fwhm_eff_pix=np.nan)
        native_pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(flipped_wcs))) * 3600.0
        corr2_warnings = []
        corr3 = corr
    else:
        cont_data, _, _ = load_line_extension(linemaps_path, continuum_extname)
        if cont_data.shape != old_shape:
            raise ValueError(
                f"continuum extension {continuum_extname} shape {cont_data.shape} != line map shape {old_shape}"
            )
        flipped_cont = flip_rotate_data(cont_data, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)

        # ---- Stage 2: global correlation on the continuum map ----
        corr2 = cross_correlate_to_reference(
            _restrict_for_corr(flipped_cont, flipped_wcs), flipped_wcs, ref_data_for_corr, ref_wcs,
            bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=edge_warn_frac,
            peak_sig_threshold=peak_sig_threshold, mask_center_radius_arcsec=mask_center_radius_arcsec,
            **search_kwargs,
        )
        for w in corr2["warnings"]:
            warnings.warn(f"[{target_name}] stage 2 (continuum): {w}")
        corr2_warnings = corr2["warnings"]

        dra2, ddec2 = corr2["dra_deg"], corr2["ddec_deg"]
        stage2_wcs = apply_crval_shift(flipped_wcs, dra2, ddec2)

        # ---- Centroid + FWHM on the continuum map's own native pixel grid ----
        # Uses a tighter background cut than the correlation steps (5 sigma vs 3
        # by default, auto-relaxing per centroid_bg_sigma_thresholds if 5 clears
        # nothing for this target's continuum tracer): a moment-based centroid is
        # pulled by any residual background passing the cut (e.g. a
        # flat-fielding/reduction banding pattern spread across many pixels),
        # unlike the correlation's own peak-based matching, so identifying the
        # galaxy itself needs a stricter threshold than aligning against it.
        centroid = measure_continuum_centroid(_restrict_for_corr(flipped_cont, flipped_wcs), centroid_bg_sigma_thresholds)
        if not centroid["success"]:
            warnings.warn(f"[{target_name}] continuum centroid fit failed; stage 3 falling back to global correlation")
        elif centroid["bg_sigma_used"] != centroid_bg_sigma_thresholds[0]:
            warnings.warn(
                f"[{target_name}] continuum centroid: {centroid_bg_sigma_thresholds[0]:.0f} sigma cleared no "
                f"pixels, relaxed to {centroid['bg_sigma_used']:.0f} sigma"
            )

        if make_plot:
            plot_continuum_centroid_diagnostic(
                flipped_cont, stage2_wcs, centroid, target_name,
                os.path.join(outdir, f"{target_name}_continuum_centroid.png"),
                bg_sigma_multiplier=centroid["bg_sigma_used"],
            )

        # ---- Stage 3: windowed correlation on the line map ----
        native_pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(flipped_wcs))) * 3600.0
        if centroid["success"]:
            fwhm_arcsec = centroid["fwhm_eff_pix"] * native_pix_scale_arcsec
            corr3 = cross_correlate_windowed_to_reference(
                _restrict_for_corr(flipped_reg, stage2_wcs), stage2_wcs, ref_data_for_corr, ref_wcs, fwhm_arcsec,
                window_radius_fwhm_factor=window_radius_fwhm_factor,
                search_center_shift=(0.0, 0.0),
                bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=edge_warn_frac,
                peak_sig_threshold=peak_sig_threshold, mask_center_radius_arcsec=mask_center_radius_arcsec,
            )
        else:
            corr3 = cross_correlate_to_reference(
                _restrict_for_corr(flipped_reg, stage2_wcs), stage2_wcs, ref_data_for_corr, ref_wcs,
                bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=edge_warn_frac,
                peak_sig_threshold=peak_sig_threshold, mask_center_radius_arcsec=mask_center_radius_arcsec,
            )
        for w in corr3["warnings"]:
            warnings.warn(f"[{target_name}] stage 3 (line map): {w}")

        # Only trust stage 3's refinement if its windowed peak is actually significant
        # against the noise WITHIN that window (see cross_correlate_to_reference's
        # significance fix) -- the line map is much clumpier/lower-S/N than the
        # continuum map, and within a small window there may be no real detection at
        # all, just noise. Applying a non-significant "refinement" on top of an
        # already-good stage-2 solution can only make things worse, never better.
        if corr3["peak_significance"] >= peak_sig_threshold:
            dra3, ddec3 = corr3["dra_deg"], corr3["ddec_deg"]
        else:
            warnings.warn(
                f"[{target_name}] stage 3 peak significance {corr3['peak_significance']:.1f} sigma "
                f"below threshold {peak_sig_threshold} sigma; keeping stage 2's continuum-only solution"
            )
            dra3, ddec3 = 0.0, 0.0

    # Compose, don't overwrite: apply_crval_shift subtracts its argument from CRVAL at
    # each stage, so the total correction relative to the ORIGINAL flipped_wcs is the
    # simple sum of the two stages' measured offsets (both in the same raw-degree
    # convention already used internally by apply_crval_shift/dra_deg). When
    # use_continuum=False, dra2/ddec2 are 0 so this is just the single correlation's offset.
    dra_total, ddec_total = dra2 + dra3, ddec2 + ddec3
    corrected_wcs = apply_crval_shift(stage2_wcs, dra3, ddec3)

    resid_pix = None
    if verify_after:
        check = cross_correlate_to_reference(
            _restrict_for_corr(flipped_reg, corrected_wcs), corrected_wcs, ref_data_for_corr, ref_wcs,
            bg_sigma_multiplier=bg_sigma_multiplier, edge_warn_frac=1.0, peak_sig_threshold=0.0,
            mask_center_radius_arcsec=mask_center_radius_arcsec,
        )
        resid_pix = float(np.hypot(check["shift_x"], check["shift_y"]))
        if resid_pix > 2.0:
            warnings.warn(
                f"[{target_name}] post-correction residual shift is {resid_pix:.1f} px "
                "(expected ~0) -- CRVAL correction may not have converged"
            )

    out_fits = write_corrected_fits(linemaps_path, dra_total, ddec_total, old_shape, target_name, outdir,
                                     flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg)

    out_png = None
    if make_plot:
        ext = f"{line.upper()}_FLUX"
        corrected_flux, corrected_line_wcs, _ = load_line_extension(out_fits, ext)
        out_png = plot_alignment_diagnostic(
            ref_data, ref_wcs, corrected_flux, corrected_line_wcs,
            target_name, line, os.path.join(outdir, f"{target_name}_astrom_check.png"),
            contour_sigma_multiples, vmin_sigma_floor=astrom_check_vmin_sigma,
            qso_coord=qso_refined_coord,
        )

    print(
        f"[{target_name}] stage2 dRA={dra2 * 3600:.3f}\" dDec={ddec2 * 3600:.3f}\" | "
        f"stage3 dRA={dra3 * 3600:.3f}\" dDec={ddec3 * 3600:.3f}\" | "
        f"total dRA={dra_total * 3600:.3f}\" dDec={ddec_total * 3600:.3f}\" "
        f"resid_px={resid_pix}"
    )

    return dict(
        target_name=target_name, dra_arcsec=dra_total * 3600, ddec_arcsec=ddec_total * 3600,
        stage2_dra_arcsec=dra2 * 3600, stage2_ddec_arcsec=ddec2 * 3600,
        stage3_dra_arcsec=dra3 * 3600, stage3_ddec_arcsec=ddec3 * 3600,
        fwhm_arcsec=centroid["fwhm_eff_pix"] * native_pix_scale_arcsec if centroid["success"] else None,
        shift_x_pix=corr3["shift_x"], shift_y_pix=corr3["shift_y"],
        peak_significance=corr3["peak_significance"], warnings=corr2_warnings + corr3["warnings"],
        verify_residual_pix=resid_pix, out_fits=out_fits, out_png=out_png,
        qso_coord_refined=qso_refined_coord,
    )


# The 8 dihedral (flip_axis, rotate_ccw_deg) combinations flip_rotate_data
# collapses onto -- confirmed this session: flip_axis='y' at any rotation
# duplicates flip_axis='x' at rotation+180deg, so {None,'x'} x all 4
# rotations already exhausts the group without duplicates.
ORIENTATION_CANDIDATES = [(fa, rot) for fa in (None, "x") for rot in (0, 90, 180, 270)]


def auto_correct_target(linemaps_path, calibration_reference_path, science_reference_path=None,
                         target_name=None, qso_coord=None, candidates=None, outdir=None,
                         calibrator_exclude_radius_arcsec=5.0, calibrator_sigma_threshold=8.0,
                         calibrator_aperture_radius_arcsec=5.0, calibrator_confirm_sigma=5.0,
                         **correct_target_kwargs):
    """
    Automatically determine (flip_axis, rotate_ccw_deg) for correct_target
    instead of trusting a single manual guess: cross-correlation is provably
    blind to reflection (a single, roughly-symmetric target's correlation
    peak is nearly invariant under mirroring about its own centroid), so a
    wrong-parity guess can pass every one of correct_target's own self-
    consistency checks. This tries all 8 dihedral combinations
    (ORIENTATION_CANDIDATES) and scores each against auto-detected
    calibrator sources (find_calibrator_sources/score_orientation_candidate)
    -- a second, spatially offset source is what actually discriminates
    parity, not the target itself (confirmed empirically this session: for
    one target, an unrelated field star 12.9" away showed a flipped East-
    West sign under a wrong-parity transform, while the QSO+galaxy blend
    itself, too tight to resolve independently, gave no such signal).

    calibration_reference_path should be a WIDE reference cutout -- wide
    enough to plausibly contain a usable secondary source; a target's own
    science-sized reference cutout is often too small (confirmed for one
    target this session, whose QSO+galaxy blend has no other >8sigma source
    within 30 arcsec). science_reference_path (passed to correct_target
    itself) defaults to calibration_reference_path when not given
    separately.

    qso_coord, if given, is both (a) excluded from every candidate's
    correlation/centroiding (see correct_target's own qso_coord handling --
    forwarded via **correct_target_kwargs is NOT how this works; qso_coord
    is a named parameter here specifically so it can also seed
    find_calibrator_sources' own exclusion) and (b) the exclusion center
    used when auto-detecting calibrator sources on calibration_reference_path
    (so the QSO/target region itself is never mistaken for a calibrator).

    Every candidate's full correct_target() output (FITS, continuum-centroid
    PNG, astrom-check PNG) is written to
    {outdir}/orientation_search/{flip_label}_{rotate_ccw_deg}/ for visual
    review, not just the winner's -- "more compute, but thorough" was an
    explicit choice here over a cheaper numbers-only ranking.

    Returns a dict:
      best -- the winning candidate's correct_target() result dict, plus
        'flip_axis'/'rotate_ccw_deg'/'calibrator_score'/'n_calibrators_confirmed'.
      ranked -- list of every candidate's result (same shape as best, minus
        the "winning" framing), best-first.
      calibrators -- find_calibrator_sources' own output list.
      orientation_ambiguous -- True if zero calibrators were found at all,
        OR none were ever confirmed by ANY candidate (both mean nothing
        actually discriminated parity, so the ranking fell back to stage-3
        peak_significance alone) -- treat the "best" pick with appropriate
        caution in either case.
    """
    target_name = target_name or _infer_target_name(linemaps_path)
    science_reference_path = science_reference_path or calibration_reference_path
    outdir = outdir or os.path.dirname(os.path.abspath(linemaps_path))
    candidates = candidates or ORIENTATION_CANDIDATES
    continuum_extname = correct_target_kwargs.get("continuum_extname", "CONTINUUM_IMG")

    calib_data, calib_wcs, _ = load_reference_image(calibration_reference_path)
    calibrators = []
    if qso_coord is not None:
        calibrators = find_calibrator_sources(
            calib_data, calib_wcs, qso_coord,
            exclude_radius_arcsec=calibrator_exclude_radius_arcsec,
            sigma_threshold=calibrator_sigma_threshold,
        )
    if not calibrators:
        warnings.warn(f"[{target_name}] no calibrator sources found (qso_coord={qso_coord}); "
                       "orientation ranking will fall back to stage-3 peak_significance alone "
                       "and should be treated as unverified")

    ranked = []
    for flip_axis, rotate_ccw_deg in candidates:
        flip_label = flip_axis if flip_axis is not None else "none"
        cand_outdir = os.path.join(outdir, "orientation_search", f"{flip_label}_{rotate_ccw_deg}")
        os.makedirs(cand_outdir, exist_ok=True)
        try:
            res = correct_target(
                linemaps_path, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg,
                reference_path=science_reference_path, target_name=target_name,
                qso_coord=qso_coord, outdir=cand_outdir, **correct_target_kwargs,
            )
        except Exception as exc:
            warnings.warn(f"[{target_name}] orientation candidate flip={flip_axis} rot={rotate_ccw_deg} failed: {exc}")
            continue

        cal_score, n_confirmed, cal_details = 0.0, 0, []
        if calibrators:
            cont_corrected, cont_corrected_wcs, _ = load_line_extension(res["out_fits"], continuum_extname)
            cal_score, n_confirmed, cal_details = score_orientation_candidate(
                cont_corrected, cont_corrected_wcs, calibrators,
                aperture_radius_arcsec=calibrator_aperture_radius_arcsec,
                sigma_threshold=calibrator_confirm_sigma,
            )

        ranked.append(dict(
            res, flip_axis=flip_axis, rotate_ccw_deg=rotate_ccw_deg,
            calibrator_score=cal_score, n_calibrators_confirmed=n_confirmed,
            calibrator_details=cal_details,
        ))

    if not ranked:
        raise RuntimeError(f"[{target_name}] every orientation candidate failed")

    ranked.sort(key=lambda r: (r["calibrator_score"], r["peak_significance"]), reverse=True)
    best = ranked[0]
    orientation_ambiguous = not calibrators or best["n_calibrators_confirmed"] == 0
    if orientation_ambiguous and calibrators:
        warnings.warn(f"[{target_name}] {len(calibrators)} calibrator(s) found but none were "
                       "confirmed by ANY orientation candidate; ranking fell back to stage-3 "
                       "peak_significance alone and should be treated as unverified")

    print(f"[{target_name}] orientation search ranking "
          f"({'AMBIGUOUS -- peak_significance only, see warnings' if orientation_ambiguous else 'calibrator-scored'}):")
    for r in ranked:
        flip_label = r["flip_axis"] if r["flip_axis"] is not None else "none"
        print(f"    flip={flip_label:>4s} rot={r['rotate_ccw_deg']:>3d}  "
              f"calibrator_score={r['calibrator_score']:.1f} "
              f"n_confirmed={r['n_calibrators_confirmed']}/{len(calibrators)}  "
              f"peak_sig={r['peak_significance']:.1f}")

    return dict(best=best, ranked=ranked, calibrators=calibrators,
                orientation_ambiguous=orientation_ambiguous)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--linemaps", required=True, help="Path to {target}_linemaps.fits from emission_line_fitter.py")
    parser.add_argument("--correct-cube-from-linemaps", default=None, metavar="CUBE_PATH",
                         help="Instead of the normal registration flow, apply the flip/rotate/CRVAL "
                              "correction already stored in --linemaps (must be a *_linemaps_corrected.fits "
                              "produced by a prior run) to this full 3D cube FITS file. Requires --outdir "
                              "or writes next to CUBE_PATH; use --target-name's absence to keep the "
                              "default LLAMAS_combined_cube_{label}_corrected.fits naming, or set "
                              "--cube-label.")
    parser.add_argument("--cube-label", default="green", help="Label used in the output cube filename (default: green)")
    parser.add_argument("--reference", default=None,
                         help="WCS-bearing reference FITS image (e.g. DECaLS r-band cutout). "
                              "Default: auto-detect {target}_decals_r.fits next to the *_combined folder")
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--line", default="Halpha", help="Registration/contour line for stage 3 (default: Halpha)")
    parser.add_argument("--flip-axis", choices=["x", "y", "none"], default=None,
                         help="Axis to flip before rotating ('none' = no flip). REQUIRED (unless "
                              "--correct-cube-from-linemaps is used) -- no default, since this is "
                              "not universal across targets: confirmed FRB20240312D and FRB20240210A "
                              "need different values. Determine per target (see each target's "
                              "orientation_diagnostics/) and record it in your project's target registry "
                              "rather than relying on a default here.")
    parser.add_argument("--rotate-ccw-deg", type=int, default=None, choices=[0, 90, 180, 270],
                         help="Rotation in degrees counterclockwise, applied after --flip-axis. REQUIRED "
                              "(unless --correct-cube-from-linemaps is used) -- see --flip-axis.")
    parser.add_argument("--no-continuum", dest="use_continuum", action="store_false", default=True,
                         help="Skip the continuum-anchored stage 2/3 and instead do a single global "
                              "correlation directly on --line's flux map (pre-continuum-anchoring "
                              "behavior). Use when the continuum map's own S/N is too poor to reliably "
                              "isolate the galaxy (e.g. a reduction artifact getting misidentified as "
                              "the source and dragging stage 2 to a spurious offset).")
    parser.add_argument("--search-radius-arcsec", type=float, default=None,
                         help="Restrict the continuum (or, with --no-continuum, line-map) correlation "
                              "to this radius around zero shift relative to the reference image's own "
                              "center. Use when a higher-contrast, off-target feature (e.g. a foreground "
                              "star) is winning an unrestricted correlation over the true, fainter/more "
                              "diffuse target that sits near the reference center.")
    parser.add_argument("--mask-center-radius-arcsec", type=float, default=None,
                         help="Exclude a disk of this radius around the reference image's own "
                              "geometric center from the correlation. Use when the reference has a "
                              "bright nuclear/continuum core with no counterpart at all in the "
                              "emission-line map (e.g. an evolved bulge or AGN continuum source with "
                              "no cospatial star formation), which would otherwise pull the correlation "
                              "toward a feature the line map structurally cannot show.")
    parser.add_argument("--continuum-extname", default="CONTINUUM_IMG",
                         help="FITS extension driving stage 2 (default: CONTINUUM_IMG, a broadband "
                              "continuum image -- see emission_line_fitter.compute_broadband_continuum)")
    parser.add_argument("--outdir", default=None, help="Default: same directory as --linemaps")
    parser.add_argument("--bg-sigma", type=float, default=3.0,
                         help="Background-clipping threshold for correlation masking (default: 3.0)")
    parser.add_argument("--centroid-bg-sigma", type=float, nargs="+", default=[5.0, 3.0],
                         help="Background-clipping thresholds for continuum centroid/FWHM identification, "
                              "tried in order until one retains enough pixels (default: 5.0 3.0)")
    parser.add_argument("--edge-warn-frac", type=float, default=0.4)
    parser.add_argument("--peak-sig-threshold", type=float, default=5.0)
    parser.add_argument("--window-radius-fwhm-factor", type=float, default=1.5,
                         help="Stage-3 search radius = this x continuum FWHM "
                              "(default 1.5 -> window diameter = 3xFWHM)")
    parser.add_argument("--contour-sigma", type=float, nargs="+", default=[3, 6, 14, 30, 65, 140, 300])
    parser.add_argument("--no-verify", dest="verify_after", action="store_false", default=True)
    parser.add_argument("--no-plot", dest="make_plot", action="store_false", default=True)
    parser.add_argument("--astrom-check-vmin-sigma", type=float, default=10.0,
                         help="Background-sigma floor for the astrom_check plot's reference-image "
                              "display stretch (default: 10.0, capped at half the image's peak "
                              "significance). Lower this to keep more of the reference's faint, "
                              "extended galaxy light visible; the default clips it to emphasize "
                              "bright compact structure instead.")
    args = parser.parse_args()

    if args.correct_cube_from_linemaps is not None:
        correct_cube(args.correct_cube_from_linemaps, args.linemaps,
                     outdir=args.outdir, cube_label=args.cube_label)
        return

    if args.reference is None and default_reference_path(args.linemaps) is None:
        parser.error("--reference is required (no default *_decals_r.fits found next to the cube folder)")
    if args.flip_axis is None:
        parser.error("--flip-axis is required (no universal default -- see --flip-axis's help)")
    if args.rotate_ccw_deg is None:
        parser.error("--rotate-ccw-deg is required (no universal default -- see --rotate-ccw-deg's help)")

    flip_axis = None if args.flip_axis == "none" else args.flip_axis
    correct_target(
        args.linemaps, reference_path=args.reference, target_name=args.target_name,
        line=args.line, flip_axis=flip_axis, rotate_ccw_deg=args.rotate_ccw_deg,
        use_continuum=args.use_continuum, continuum_extname=args.continuum_extname,
        search_radius_arcsec=args.search_radius_arcsec,
        mask_center_radius_arcsec=args.mask_center_radius_arcsec,
        outdir=args.outdir, verify_after=args.verify_after, make_plot=args.make_plot,
        bg_sigma_multiplier=args.bg_sigma, centroid_bg_sigma_thresholds=tuple(args.centroid_bg_sigma),
        edge_warn_frac=args.edge_warn_frac, peak_sig_threshold=args.peak_sig_threshold,
        window_radius_fwhm_factor=args.window_radius_fwhm_factor,
        contour_sigma_multiples=tuple(args.contour_sigma),
        astrom_check_vmin_sigma=args.astrom_check_vmin_sigma,
    )


if __name__ == "__main__":
    main()
