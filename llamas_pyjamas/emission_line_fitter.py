"""
emission_line_fitter.py

Per-spaxel continuum + Gaussian emission-line fitting for LLAMAS combined
datacubes (FLUX/VAR/WEIGHT format), producing 2D flux (Gaussian amplitude)
maps saved as PNGs and a multi-extension FITS file.

Fits four line complexes independently per spaxel, each as a linear
continuum plus a shared-width sum of Gaussians:
    - [OII]3727 / [OII]3729
    - Hbeta / [OIII]4959 / [OIII]5007
    - [NII]6548 / Halpha / [NII]6584
    - [SII]6716 / [SII]6731

[SII] is its own group rather than sharing Halpha/[NII]'s complex: it sits
close to the red channel's low-throughput edge, and different complexes can
genuinely register at slightly different redshifts/positions (atmospheric
dispersion) -- forcing it to share Halpha's z/sigma solution would mask that
rather than measure it.

Only lines that fall inside a cube's actual wavelength coverage (at the
given redshift) are included in each complex's model (see available_lines).

Within a group (see fit_cube), fitting is NOT a free-for-all per-spaxel pass
over the whole array. First, a fit-free narrowband image spanning the
group's own line span (compute_broadband_continuum) is masked with
mad_background_mask -- the same masking astrometry_correction.py uses on its
own images before correlation -- and spaxels outside it are skipped
entirely, never even reaching curve_fit. Second, the group's line width
sigma is determined ONCE, from the median of independent free-sigma fits at
the narrowband image's own highest-S/N spaxels (determine_group_sigma) --
this dataset's spectral resolution doesn't resolve any intrinsic line width,
so sigma is purely instrumental and has no business varying spaxel to
spaxel. Every masked spaxel is then fit with sigma held fixed at that value
(see build_model/fit_spaxel's sigma_fixed) -- a genuinely smaller parameter
vector, which also stabilizes the density-sensitive-doublet ratio fit
(RATIO_BOUNDS) at high S/N.

A target's blue/green/red channel cubes are fit and merged together (see
fit_target) so a redshift-dependent split -- e.g. [SII] landing in the red
cube instead of green -- is handled automatically rather than assumed fixed
to one channel. Each channel is, by default, first truncated to the
wavelength range where its own sensfunc THROUGHPUT is at least half its peak
value (see throughput_wave_range) -- a channel's edges can have real signal
that's too low-throughput to trust, and PypeIt's sensfunc fit can be
unstable right at the edge of its own valid range, so this keeps low-quality
edge pixels out of the fit rather than letting them masquerade as a
detection. The green channel's own auto-detected upper bound has been found
too conservative for this dataset; callers should override it to 7000
Angstrom (see fit_target's wave_range_overrides) rather than trust the raw
throughput-derived value there -- blue and red's auto-detected limits are
fine as-is.

Author: Sunil Simha
"""
import argparse
import csv
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LinearStretch
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse


# ----------------------------------------------------------------------
# Rest-frame vacuum wavelengths (Angstrom) -- single source of truth
# ----------------------------------------------------------------------
LINES = {
    "OII_3727":  3727.092,
    "OII_3729":  3729.875,
    "Hbeta":     4862.683,
    "OIII_4959": 4960.295,
    "OIII_5007": 5008.239,
    "NII_6548":  6549.852,
    "Halpha":    6564.613,
    "NII_6584":  6585.277,
    "SII_6716":  6718.294,
    "SII_6731":  6732.673,
}

# weak_line: (strong_line, fixed flux ratio weak/strong), from atomic physics.
# [OII] 3727/3729 and [SII] 6716/6731 are deliberately NOT here -- their
# ratios are electron-density dependent, not a fixed atomic ratio (see
# RATIO_BOUNDS below, which bounds rather than fixes them).
DOUBLET_RATIOS = {
    "OIII_4959": ("OIII_5007", 1 / 2.98),
    "NII_6548":  ("NII_6584",  1 / 2.96),
}

# weak_line: (strong_line, ratio_lo, ratio_hi). Density-sensitive doublets --
# ratio = weak/strong varies with local electron density rather than being
# fixed by atomic physics (see DOUBLET_RATIOS above), but is bounded by the
# well-known low-/high-density limiting values (Osterbrock & Ferland, AGN3):
# R = F(3729)/F(3727) and R = F(6716)/F(6731) both run from ~0.35 (high
# density, >~1e4 cm^-3) to ~1.5 (low density, <~10 cm^-3). Enforced by fitting
# the weak member as (strong member's amplitude * a ratio parameter bounded
# to [ratio_lo, ratio_hi]) instead of two fully independent [0, inf)
# amplitudes -- see build_model -- so curve_fit structurally cannot return an
# unphysical ratio even where the two lines are barely resolved (e.g. [OII]
# 3727/3729, ~2.8 Angstrom apart rest-frame) and two independent amplitudes
# would be only weakly constrained individually.
RATIO_BOUNDS = {
    "OII_3729": ("OII_3727", 0.35, 1.5),
    "SII_6716": ("SII_6731", 0.35, 1.5),
}

LINE_GROUPS = {
    "oii_doublet": ["OII_3727", "OII_3729"],
    "hbeta_oiii":  ["Hbeta", "OIII_4959", "OIII_5007"],
    "halpha_nii":  ["NII_6548", "Halpha", "NII_6584"],
    "sii_doublet": ["SII_6716", "SII_6731"],
}

C_KMS = 2.99792458e5

# Default reference line + S/N threshold used to build the "galaxy footprint"
# mask applied to plots (see build_reference_mask) -- Halpha because it's
# normally the brightest, most robustly detected line feature, so masking on
# it is far less prone to edge-of-coverage artifacting than masking on a
# fainter line directly.
DEFAULT_REFERENCE_LINE = "Halpha"
DEFAULT_MASK_SIGMA = 2.0

# ----------------------------------------------------------------------
# FRB localization ellipse
# ----------------------------------------------------------------------
def load_localization(csv_path, target_name):
    """
    Look up target_name (e.g. 'FRB20240312D', or 'FRB20240312D_corrected')
    in a localization CSV -- matched against the CSV's 'frb' column after
    stripping a leading 'FRB' and any trailing '_'-suffix from target_name.
    Returns (SkyCoord, a_arcsec, b_arcsec, pa_deg), or None if csv_path is
    None or doesn't exist, or no matching row is found, so callers can skip
    the ellipse rather than fail the whole plot. csv_path is None whenever
    a caller doesn't have a bespoke localization table to point at -- this
    module has no built-in default path (see the redux driver script, which
    supplies one).
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None
    key = target_name.replace("FRB", "", 1).split("_")[0]
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["frb"].strip() == key:
                coord = SkyCoord(ra=row["ra"].strip(), dec=row["dec"].strip(), unit=(u.hourangle, u.deg))
                return coord, float(row["a_arcsec"]), float(row["b_arcsec"]), float(row["PA_deg"])
    return None


def _localization_ellipse_patch(wcs2d, loc, n_sigma=2.0):
    """
    Build (but don't draw) the n_sigma FRB localization Ellipse patch in
    wcs2d's own pixel coordinates -- shared by add_localization_ellipse
    (drawing) and localization_ellipse_mask (spaxel selection) so the two
    stay exactly consistent. PA is measured East of North (standard IAU
    convention); converted to matplotlib Ellipse's angle (CCW from +x)
    assuming the standard North-up/East-left orientation used throughout
    this module's WCS-projected plots (+x pixel = West, +y pixel = North,
    so a PA measured from North towards East maps to screen angle 90+PA).
    """
    coord, a_arcsec, b_arcsec, pa_deg = loc
    x_pix, y_pix = wcs2d.world_to_pixel(coord)
    pix_scale_arcsec = float(np.nanmean(proj_plane_pixel_scales(wcs2d))) * 3600.0
    width_pix = 2 * n_sigma * a_arcsec / pix_scale_arcsec
    height_pix = 2 * n_sigma * b_arcsec / pix_scale_arcsec
    return Ellipse((x_pix, y_pix), width=width_pix, height=height_pix, angle=90.0 + pa_deg)


def add_localization_ellipse(ax, wcs2d, loc, n_sigma=2.0):
    """
    Draw the FRB localization ellipse (loc from load_localization) on a
    WCS-projected axes at n_sigma confidence (the CSV's a/b are taken as
    1-sigma semi-axes).

    Styled with a white halo behind a black outline (matplotlib path
    effects) rather than a single fixed color, so it stays legible against
    any of this module's colormaps (viridis, Greens, Reds, OrRd, Purples)
    regardless of what value a given colormap renders light or dark.
    """
    ellipse = _localization_ellipse_patch(wcs2d, loc, n_sigma=n_sigma)
    ellipse.set(edgecolor="black", facecolor="none", lw=1.5,
                transform=ax.get_transform("pixel"), zorder=5)
    ellipse.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
    ax.add_patch(ellipse)
    return ellipse


def localization_ellipse_mask(shape, wcs2d, loc, n_sigma=2.0):
    """
    Boolean (ny, nx) mask, True for pixels whose center falls within the
    n_sigma FRB localization ellipse (see _localization_ellipse_patch --
    built with the identical center/size/angle used for the drawn ellipse,
    via matplotlib's own path geometry, so spaxel selection and the visual
    ellipse can never disagree).
    """
    ellipse = _localization_ellipse_patch(wcs2d, loc, n_sigma=n_sigma)
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = ellipse.get_path().transformed(ellipse.get_patch_transform()).contains_points(points)
    return inside.reshape(ny, nx)


# ----------------------------------------------------------------------
# Cube I/O
# ----------------------------------------------------------------------
def load_cube(fits_path):
    """
    Read a LLAMAS combined cube (FLUX/VAR/WEIGHT extensions).

    Returns
    -------
    wave_AA : ndarray, shape (nwave,)
        Wavelength array in Angstrom.
    flux, var, weight : ndarray, shape (nwave, ny, nx)
    wcs : astropy.wcs.WCS
        Full 3D WCS (header CUNIT3 fixed to Angstrom).
    header : astropy.io.fits.Header
    """
    with fits.open(fits_path) as hdul:
        flux = hdul["FLUX"].data if "FLUX" in hdul else hdul[0].data
        var = hdul["VAR"].data
        weight = hdul["WEIGHT"].data
        header = (hdul["FLUX"].header if "FLUX" in hdul else hdul[0].header).copy()

    # header stores the spectral axis in meters -- read the TRUE effective
    # step via pixel_scale_matrix (not a raw CDELT3 dict lookup): a raw cube
    # encodes it as a bare CDELT3, but an astrometry-corrected cube (see
    # astrometry_correction._build_cube_wcs3d) has it split across a
    # vestigial CDELT3=1.0 and the real value in PC3_3 -- to_header() always
    # round-trips a CD-matrix WCS through the PC+CDELT convention.
    # pixel_scale_matrix resolves either convention correctly.
    cdelt3_m = WCS(header).pixel_scale_matrix[2, 2]
    header["CRVAL3"] = header["CRVAL3"] * 1e10
    header["CDELT3"] = cdelt3_m * 1e10
    for k in ("PC3_1", "PC3_2", "PC3_3", "PC1_3", "PC2_3", "CD3_1", "CD3_2", "CD3_3", "CD1_3", "CD2_3"):
        header.pop(k, None)
    header["CUNIT3"] = "Angstrom"
    wcs = WCS(header)

    nwave = flux.shape[0]
    crpix3 = header.get("CRPIX3", 1.0)
    wave_AA = header["CRVAL3"] + (np.arange(nwave) + 1 - crpix3) * header["CDELT3"]

    return wave_AA, flux, var, weight, wcs, header


def throughput_wave_range(sensfunc_fits_path, frac_of_peak=0.5, smooth_aa=30.0):
    """
    Wavelength range [wave_lo, wave_hi] (observed-frame Angstrom) over which a
    PypeIt SensFunc's THROUGHPUT extension is at least frac_of_peak of its own
    peak value, read directly from the sensfunc FITS file's WAVE/THROUGHPUT
    image extensions.

    PypeIt fills THROUGHPUT with -1 outside the wavelength range actually
    covered by the fit, and can show a narrow, spurious spike right at that
    fit-edge boundary (seen in a real LLAMAS green-channel sensfunc -- an
    edge-of-fit instability, not a real throughput feature: it rises to ~3x
    the plateau value over ~10 Angstrom right where the -1 fill ends, then
    collapses to ~0). To stay robust to that, the -1 fill is masked out
    first, then the curve is smoothed with a median filter over a
    smooth_aa-Angstrom window (wide enough to reject both single-sample noise
    spikes and that ~10 Angstrom edge artifact, while narrow enough not to
    wash out the channel's real, hundreds-of-Angstrom-wide throughput
    rolloff) before the peak and its half-max crossings are located. The
    returned range walks outward from the smoothed curve's global peak to the
    nearest point on each side where it first drops below
    frac_of_peak * peak, so a secondary bump elsewhere in the band can't be
    mistaken for part of the usable range.
    """
    with fits.open(sensfunc_fits_path) as hdul:
        wave = np.ravel(np.asarray(hdul["WAVE"].data, dtype=np.float64))
        thr = np.ravel(np.asarray(hdul["THROUGHPUT"].data, dtype=np.float64))

    good = np.isfinite(wave) & np.isfinite(thr) & (wave > 0) & (thr > -0.5)
    if good.sum() < 3:
        raise ValueError(f"{sensfunc_fits_path}: no valid THROUGHPUT samples")
    w, t = wave[good], thr[good]
    order = np.argsort(w)
    w, t = w[order], t[order]

    dw = np.median(np.diff(w))
    kernel = max(3, int(round(smooth_aa / dw)))
    if kernel % 2 == 0:
        kernel += 1
    t_smooth = medfilt(t, kernel_size=kernel)

    peak_idx = int(np.argmax(t_smooth))
    threshold = frac_of_peak * t_smooth[peak_idx]
    above = t_smooth >= threshold

    lo = peak_idx
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak_idx
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1

    return float(w[lo]), float(w[hi])


def compute_broadband_continuum(cube_path, wave_min=6400.0, wave_max=6700.0):
    """
    Inverse-variance-weighted broadband continuum image, collapsing the cube
    directly over [wave_min, wave_max] (observed-frame Angstrom) -- a genuine
    continuum IMAGE integrated over many wavelength pixels of line-free flux,
    rather than the much noisier single linear-fit intercept a per-spaxel
    Gaussian+continuum fit produces over a narrow window. Default window
    (6400-6700 A observed) avoids Hbeta/[OIII]/[NII]/Halpha/[SII] for the
    LLAMAS green channel at the low redshifts (z~0.02-0.05) typical of these
    targets -- check line coverage at other redshifts before reusing as-is.

    Returns
    -------
    img, img_err : ndarray, shape (ny, nx)
    header2d : astropy.io.fits.Header (2D celestial WCS)
    """
    wave, flux, var, weight, wcs, header = load_cube(cube_path)
    sel = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(sel):
        raise ValueError(f"cube has no wavelength coverage in [{wave_min}, {wave_max}] Angstrom")

    f, v, w = flux[sel], var[sel], weight[sel]
    valid = np.isfinite(f) & np.isfinite(v) & (v > 0) & (w > 0)
    safe_v = np.where(valid, v, 1.0)  # avoid divide-by-zero warnings on discarded elements
    ivar = np.where(valid, w / safe_v, 0.0)  # pipeline weight combined with inverse variance
    num = np.sum(np.where(valid, f * ivar, 0.0), axis=0)
    den = np.sum(ivar, axis=0)

    ny, nx = den.shape
    img = np.full((ny, nx), np.nan)
    img_err = np.full((ny, nx), np.nan)
    good = den > 0
    img[good] = num[good] / den[good]
    img_err[good] = np.sqrt(1.0 / den[good])

    header2d = wcs.celestial.to_header()
    return img, img_err, header2d


def _cube_wave_bounds(fits_path):
    """
    Wavelength coverage [wave_lo, wave_hi] (observed-frame Angstrom) of a
    combined cube, read from the FLUX header alone (CRVAL3/CDELT3/NAXIS3) --
    no FLUX/VAR/WEIGHT data loaded. Used to cheaply pick which channel's
    cube covers a given wavelength before loading the (large) data arrays.
    """
    header = fits.getheader(fits_path, extname="FLUX")
    nwave = header["NAXIS3"]
    crpix3 = header.get("CRPIX3", 1.0)
    crval3 = header["CRVAL3"] * 1e10
    # See load_cube's docstring comment: read the true spectral step via
    # pixel_scale_matrix, not a raw CDELT3 lookup, so this also works for an
    # astrometry-corrected cube's header.
    cdelt3 = WCS(header).pixel_scale_matrix[2, 2] * 1e10
    w0 = crval3 + (1 - crpix3) * cdelt3
    w1 = crval3 + (nwave - crpix3) * cdelt3
    return (w0, w1) if w0 <= w1 else (w1, w0)


def compute_narrowband_image(cube_path, center_wave, half_width=5.0):
    """
    Inverse-variance-weighted narrowband image, collapsing the cube over
    [center_wave - half_width, center_wave + half_width] (observed-frame
    Angstrom) -- the same mechanism as compute_broadband_continuum, just a
    much narrower, line-centered window. No fitting: a straight flux sum,
    so any real continuum level under the line is included in the image,
    not subtracted (unlike the Gaussian+continuum per-spaxel fit elsewhere
    in this module).

    Returns None (not an exception) if the cube has no wavelength coverage
    at all in the requested window, so callers can try the next channel.
    """
    try:
        return compute_broadband_continuum(cube_path, center_wave - half_width, center_wave + half_width)
    except ValueError:
        return None


def find_line_cube(cube_paths, rest_wave, z, half_width=5.0, preferred_order=("green", "red", "blue")):
    """
    Pick which channel's cube (from cube_paths, e.g. {'blue':.., 'green':..,
    'red':..}) fully covers [rest_wave*(1+z) - half_width, + half_width],
    checking only each cube's header (see _cube_wave_bounds) -- no data
    loaded here.

    Deliberately channel-agnostic and NOT gated by throughput_wave_range:
    this is for the fit-free narrowband panel (see plot_narrowband_panel),
    whose whole point is to show real flux in a line-centered window even
    where the per-spaxel Gaussian fit's throughput-based available_lines()
    check would exclude it -- e.g. Halpha/[NII] landing in the gap between
    the green and red channels' own 50%-of-peak-throughput ranges for
    z~0.047-0.05 targets (see throughput_wave_range docstring).

    Returns (channel, path) for the first channel (in preferred_order)
    whose cube covers the window, or (None, None) if none do.
    """
    center = rest_wave * (1 + z)
    for ch in preferred_order:
        path = cube_paths.get(ch)
        if path is None or not os.path.exists(path):
            continue
        lo, hi = _cube_wave_bounds(path)
        if lo <= center - half_width and hi >= center + half_width:
            return ch, path
    return None, None


# Narrowband panel's own line list -- like PANEL_LINES but without [SII]:
# both [SII]6716/6731 sit close enough to the red channel's own
# low-throughput edge (see throughput_wave_range) that even a fit-free
# narrowband cutout of them is usually dominated by edge artifacting rather
# than real signal, so they're dropped here rather than shown misleadingly.
# Hbeta was dropped too (3-panel, not 4) per user request.
NARROWBAND_PANEL_LINES = [
    ("OIII_5007", "Greens"),
    ("Halpha", "Reds"),
    ("NII_6584", "OrRd"),
]

# LaTeX (real usetex, not mathtext) titles for the narrowband panel.
NARROWBAND_LINE_TITLES = {
    "Hbeta": r"H$\beta$",
    "OIII_5007": r"[O\,\textsc{iii}]\,$\lambda 5007$",
    "Halpha": r"H$\alpha$",
    "NII_6584": r"[N\,\textsc{ii}]\,$\lambda 6584$",
    "SII_6716": r"[S\,\textsc{ii}]\,$\lambda 6716$",
    "SII_6731": r"[S\,\textsc{ii}]\,$\lambda 6731$",
}

# 1e-17 erg/s/cm^2/Angstrom (a narrowband cutout's native units, see
# compute_narrowband_image/BUNIT) divided by this many arcsec^2 -> the
# surface-brightness units requested for the narrowband panel.
SURFACE_BRIGHTNESS_LABEL = r"Surface brightness ($10^{-17}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ arcsec^{-2}}$)"


def flip_rotate_narrowband(data, flip_axis, rotate_ccw_deg):
    """
    Apply the same flip-then-rotate transform as astrometry_correction.py's
    flip_rotate_data(): an optional mirror ('x' negates the last axis, 'y'
    the second-to-last) followed by a multiple-of-90-degree counterclockwise
    rotation (np.rot90 convention). Used to bring a narrowband image
    computed here -- in its source cube's own native pixel-grid orientation
    -- into the SAME orientation as the WCS stored in a target's
    *_linemaps_corrected.fits (see _load_corrected_orientation), so the two
    can be plotted together.
    """
    if flip_axis == "x":
        data = np.flip(data, axis=-1)
    elif flip_axis == "y":
        data = np.flip(data, axis=-2)
    elif flip_axis not in (None, "None"):
        raise ValueError(f"flip_axis must be None, 'x', or 'y', got {flip_axis!r}")
    rotate_ccw_deg = int(rotate_ccw_deg)
    if rotate_ccw_deg % 90 != 0:
        raise ValueError(f"rotate_ccw_deg must be a multiple of 90, got {rotate_ccw_deg}")
    return np.rot90(data, k=(rotate_ccw_deg // 90) % 4, axes=(-2, -1)).copy()


def _load_corrected_orientation(corrected_linemaps_path):
    """
    Read the celestial WCS and the (flip_axis, rotate_ccw_deg) transform
    astrometry_correction.py applied for this target, from the first
    populated line extension of its *_linemaps_corrected.fits. Every
    extension in that file shares the same transform and the same
    already-registered (North-up, East-left), CRVAL-corrected WCS -- see
    astrometry_correction.write_corrected_fits -- so any one of them will do.
    """
    with fits.open(corrected_linemaps_path) as hdul:
        for hdu in hdul[1:]:
            if isinstance(hdu, fits.ImageHDU) and hdu.data is not None:
                flip_axis = hdu.header.get("WCSFLIP", "None")
                flip_axis = None if flip_axis == "None" else flip_axis
                rotate_ccw_deg = int(hdu.header.get("WCSROT", 0))
                return WCS(hdu.header).celestial, flip_axis, rotate_ccw_deg
    raise ValueError(f"No populated image extensions found in {corrected_linemaps_path}")


def load_halpha_fit_mask(linemaps_path, flip_axis, rotate_ccw_deg, zero_thresh=1e-6):
    """
    Boolean footprint mask (True = keep), built from a target's *_linemaps.fits
    (the raw, per-spaxel Gaussian+continuum FIT output -- NOT the fit-free
    narrowband images) HALPHA_FLUX extension: False wherever the fit has no
    valid amplitude (NaN -- not enough good pixels, or the fit didn't
    converge) or the fitted amplitude is at its hard lower bound of 0 (see
    fit_spaxel's `lower = [..., 0.0]*n_amp, ...]`), True elsewhere.

    "At the bound" is checked as < zero_thresh rather than an exact `== 0`
    comparison: curve_fit's bounded optimizer lands asymptotically close to
    the bound (empirically ~1e-140 to ~1e-9 across real spaxels) rather than
    exactly on it, but there is a clean many-orders-of-magnitude gap between
    that population and genuine detections (which start around 1e-2 to 1e0
    in these targets) -- confirmed by checking the flux percentiles
    directly. 1e-6 sits safely in that gap for every target checked.

    This is a real, physically-motivated filter, not just another S/N cut:
    a per-spaxel fit constrained to a Gaussian line PROFILE at Halpha's
    wavelength won't return a large amplitude for a flat/broadband
    systematic (e.g. this dataset's persistent banding artifact) the way an
    unmodeled narrowband flux sum will -- the model's own linear continuum
    term soaks that up instead. Used to mask narrowband panels (see
    plot_narrowband_panel) with something more selective than their own
    narrowband S/N.

    The returned mask is flipped/rotated (see flip_rotate_narrowband) into
    the same orientation as flip_axis/rotate_ccw_deg would put a narrowband
    image built from the same target's raw cubes.
    """
    with fits.open(linemaps_path) as hdul:
        flux = np.asarray(hdul["HALPHA_FLUX"].data, dtype=np.float64)
    mask = np.isfinite(flux) & (flux >= zero_thresh)
    return flip_rotate_narrowband(mask, flip_axis, rotate_ccw_deg)


def load_halpha_peak_fraction_mask(corrected_linemaps_path, target_name,
                                    localization_csv=None,
                                    search_radius_arcsec=10.0, peak_fraction=0.03):
    """
    Boolean footprint mask (True = keep), built from a target's Halpha line
    FIT map (HALPHA_FLUX in *_linemaps_corrected.fits -- already in the
    astrometry-corrected, North-up/East-left WCS, so unlike
    load_halpha_fit_mask no separate flip/rotate is needed here): find the
    peak Halpha flux within search_radius_arcsec of the FRB host's
    localization (see load_localization), then keep only pixels at or above
    peak_fraction of that peak.

    Anchoring the peak search to the known host position (rather than the
    map's global max) keeps this robust to a bright artifact or a
    neighboring source elsewhere in the field. Raises ValueError if
    target_name has no entry in localization_csv, or if no finite Halpha
    flux exists within the search radius.
    """
    with fits.open(corrected_linemaps_path) as hdul:
        hdu = hdul["HALPHA_FLUX"]
        flux = np.asarray(hdu.data, dtype=np.float64)
        wcs2d = WCS(hdu.header).celestial

    loc = load_localization(localization_csv, target_name)
    if loc is None:
        raise ValueError(f"No localization entry for {target_name!r} in {localization_csv}")
    host_coord = loc[0]
    host_x, host_y = wcs2d.world_to_pixel(host_coord)

    ny, nx = flux.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    pixel_scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs2d))) * 3600.0
    radius_pix = search_radius_arcsec / pixel_scale_arcsec
    within_radius = (xx - host_x) ** 2 + (yy - host_y) ** 2 <= radius_pix ** 2

    local_flux = np.where(within_radius, flux, np.nan)
    if not np.any(np.isfinite(local_flux)):
        raise ValueError(f"No finite Halpha flux within {search_radius_arcsec} arcsec of "
                          f"{target_name}'s host position")
    peak = float(np.nanmax(local_flux))
    return np.isfinite(flux) & (flux >= peak_fraction * peak)


def mad_background_mask(image_2d, sigma_multiplier=2.5):
    """
    Boolean footprint mask (True = keep), replicating astrometry_correction.py's
    _mask_background_pixels: a MAD-based robust background threshold
    (median + sigma_multiplier * 1.4826*MAD, falling back to a plain
    standard deviation if the MAD-derived sigma is non-finite or zero),
    applied directly to a single 2D image.

    This is the same mechanism astrometry_correction.py uses to mask each
    line/continuum image before cross-correlation -- and, empirically, it's
    what gives that module's own line maps their clean, banding-free look in
    astrom_check PNGs (confirmed for FRB20240210A/240312D/250227A), unlike
    the Halpha-fit-based masks tried earlier in this module (see
    load_halpha_fit_mask, load_halpha_peak_fraction_mask), which left this
    dataset's banding artifact fully visible. Applied here PER PANEL (each
    line image gets its own background level/sigma), matching how
    astrometry_correction.py always computes it per-image rather than once
    for a whole target.
    """
    image = np.asarray(image_2d, dtype=np.float64)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=bool)
    finite_values = image[finite]
    background_level = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - background_level)))
    background_sigma = 1.4826 * mad
    if not np.isfinite(background_sigma) or background_sigma <= 0.0:
        background_sigma = float(np.std(finite_values))
    if not np.isfinite(background_sigma) or background_sigma <= 0.0:
        threshold = background_level
    else:
        threshold = background_level + sigma_multiplier * background_sigma
    return finite & (image >= threshold)


def plot_narrowband_panel(cube_paths, z, outdir, target_name, corrected_linemaps_path, half_width=5.0,
                           sn_min=5.0, localization_csv=None, panel_lines=None,
                           fontsize_scale=1.5, halpha_fit_linemaps_path=None, halpha_fit_zero_thresh=1e-6,
                           halpha_peak_mask=False, halpha_peak_search_radius_arcsec=10.0,
                           halpha_peak_fraction=0.03, mad_sigma=None):
    """
    Fit-free counterpart to plot_combined_panel: a 1x3 figure of narrowband
    (2*half_width Angstrom wide, default 10 Angstrom) surface-brightness
    images centered on each line's redshifted wavelength, built by directly
    collapsing the cube (see compute_narrowband_image) rather than a
    per-spaxel Gaussian+continuum fit. No continuum subtraction, no
    channel-throughput gating (see find_line_cube) -- just "what's actually
    in the cube at this wavelength," independent of the fitting pipeline's
    own availability/truncation assumptions.

    Per-pixel S/N (used for the sn_min mask) comes from the cube's own VAR
    extension via compute_narrowband_image's inverse-variance combination,
    not a post-hoc estimate. Displayed values are converted from the cube's
    native per-spaxel flux density (BUNIT, 1e-17 erg/s/cm^2/Angstrom) to
    surface brightness (1e-17 erg/s/cm^2/arcsec^2) by dividing by the pixel
    solid angle from the WCS pixel scale.

    corrected_linemaps_path: a target's *_linemaps_corrected.fits (from
    astrometry_correction.py) -- every panel is flipped/rotated (see
    flip_rotate_narrowband) into that file's already North-up/East-left,
    CRVAL-corrected WCS before plotting, so this panel lines up with the
    astrometry-corrected line maps rather than a cube's own raw orientation.

    halpha_fit_linemaps_path: a target's plain *_linemaps.fits (the raw
    per-spaxel FIT output, NOT the narrowband images computed here). When
    given (and halpha_peak_mask is False), every panel is additionally
    masked wherever the Halpha line FIT has no valid, non-zero amplitude
    there (see load_halpha_fit_mask) -- applied on top of each panel's own
    sn_min cut. In practice this caught only ~13% of pixels and did not
    meaningfully suppress this dataset's banding artifact (it turns out the
    banding gets absorbed into the fitted line amplitude too, not just an
    unmodeled narrowband sum) -- see halpha_peak_mask for the follow-up
    approach.

    halpha_peak_mask: when True (and corrected_linemaps_path has a
    HALPHA_FLUX extension), use load_halpha_peak_fraction_mask instead --
    peak Halpha flux within halpha_peak_search_radius_arcsec of the FRB
    host position, keep only pixels >= halpha_peak_fraction of that peak.
    Takes precedence over halpha_fit_linemaps_path if both are given.

    mad_sigma: when given, additionally mask each panel with
    mad_background_mask(img, sigma_multiplier=mad_sigma) -- computed
    per-panel, on that panel's own surface-brightness image, and ANDed
    together with whatever halpha_fit_mask above produced (if any). This is
    the masking approach that actually works for this dataset's banding
    (see mad_background_mask's docstring).

    panel_lines defaults to NARROWBAND_PANEL_LINES (OIII_5007, Halpha,
    NII_6584 -- no Hbeta or [SII], see its comment).

    fontsize_scale multiplies matplotlib's ambient font.size (applies to
    panel titles, axis labels, tick labels, and colorbar labels alike).
    """
    os.makedirs(outdir, exist_ok=True)
    loc = load_localization(localization_csv, target_name)
    lines = panel_lines or NARROWBAND_PANEL_LINES

    wcs2d, flip_axis, rotate_ccw_deg = _load_corrected_orientation(corrected_linemaps_path)
    pixel_scale_arcsec = proj_plane_pixel_scales(wcs2d) * 3600.0  # deg -> arcsec, per axis
    pixel_area_arcsec2 = float(pixel_scale_arcsec[0] * pixel_scale_arcsec[1])

    halpha_fit_mask = None
    if halpha_peak_mask:
        halpha_fit_mask = load_halpha_peak_fraction_mask(
            corrected_linemaps_path, target_name, localization_csv=localization_csv,
            search_radius_arcsec=halpha_peak_search_radius_arcsec, peak_fraction=halpha_peak_fraction)
    elif halpha_fit_linemaps_path is not None:
        halpha_fit_mask = load_halpha_fit_mask(halpha_fit_linemaps_path, flip_axis, rotate_ccw_deg,
                                                zero_thresh=halpha_fit_zero_thresh)

    base_fontsize = plt.rcParams["font.size"]
    with plt.rc_context({"text.usetex": True, "font.family": "serif",
                          "font.size": base_fontsize * fontsize_scale}):
        fig, axes = plt.subplots(1, len(lines), figsize=(6 * len(lines), 6),
                                  subplot_kw={"projection": wcs2d})
        if len(lines) == 1:
            axes = [axes]
        for i, ((name, cmap), ax) in enumerate(zip(lines, axes)):
            # A plain bbox rather than a PathEffects stroke -- text.usetex plus
            # patheffects on a Text artist hits a matplotlib backend bug
            # (draw_tex/_draw_text_as_path signature mismatch) in this
            # environment's matplotlib version.
            ax.text(0.05, 0.95, target_name, transform=ax.transAxes, ha="left", va="top",
                    zorder=6, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                         edgecolor="none", alpha=0.75))

            rest_wave = LINES[name]
            channel, cube_path = find_line_cube(cube_paths, rest_wave, z, half_width=half_width)
            title = NARROWBAND_LINE_TITLES.get(name, name)
            if cube_path is None:
                print(f"[{name}] no channel covers {rest_wave * (1 + z):.1f} +- {half_width:.1f} "
                      f"Angstrom, leaving panel blank")
                ax.set_title(f"{title} (unavailable)")
                ax.set_xlabel("RA")
                if i != 0:
                    ax.coords[1].set_ticklabel_visible(False)
                continue

            img, img_err, _ = compute_narrowband_image(cube_path, rest_wave * (1 + z), half_width=half_width)
            img = flip_rotate_narrowband(img, flip_axis, rotate_ccw_deg) / pixel_area_arcsec2
            img_err = flip_rotate_narrowband(img_err, flip_axis, rotate_ccw_deg) / pixel_area_arcsec2

            panel_mask = halpha_fit_mask
            if mad_sigma is not None:
                mad_mask = mad_background_mask(img, sigma_multiplier=mad_sigma)
                panel_mask = mad_mask if panel_mask is None else (panel_mask & mad_mask)

            display, norm = _display_and_norm(img, img_err, sn_min, galaxy_mask=panel_mask)
            if display is None:
                print(f"[{name}] no spaxels above S/N >= {sn_min}, panel will be all zero")
                display, norm = np.zeros_like(img), None

            im = ax.imshow(display, origin="lower", norm=norm, cmap=cmap)
            plt.colorbar(im, ax=ax, label=SURFACE_BRIGHTNESS_LABEL, pad=0.02, fraction=0.047)
            if loc is not None:
                add_localization_ellipse(ax, wcs2d, loc)
            ax.set_title(title)
            ax.set_xlabel("RA")
            if i == 0:
                ax.set_ylabel("Dec")
            else:
                ax.coords[1].set_ticklabel_visible(False)
            ax.grid(color="lightgray", lw=0.5, ls="--")

        out_path = os.path.join(outdir, f"{target_name}_panel_narrowband_OIII_Ha_NII.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_path}")


def compute_metallicity_map(nii_flux, halpha_flux, nii_err=None, halpha_err=None, sn_min=3.0):
    """
    Per-spaxel gas-phase metallicity via the Pettini & Pagel (2004) N2
    calibration:
        12 + log10(O/H) = 8.90 + 0.57 * log10([NII]6584 / Halpha)

    NaN wherever [NII]6584 or Halpha has no valid measurement: NaN flux, a
    non-positive ratio (log10 undefined), or -- when nii_err/halpha_err are
    given -- S/N < sn_min. A per-spaxel Gaussian fit can converge (non-NaN
    flux) on pure noise, so without an S/N cut this is dominated by spaxels
    with no real line detection at all (confirmed visually: without it, the
    map is filled with unphysical values across the whole array, wildly
    outside the realistic ~7-9.5 range); sn_min=3.0 matches the cut already
    used everywhere else in this module (_display_and_norm) to decide what
    counts as a real detection.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = nii_flux / halpha_flux
        metallicity = 8.90 + 0.57 * np.log10(ratio)
    invalid = ~np.isfinite(nii_flux) | ~np.isfinite(halpha_flux) | ~(ratio > 0)
    if nii_err is not None and halpha_err is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            nii_sn = nii_flux / nii_err
            halpha_sn = halpha_flux / halpha_err
        invalid |= ~(nii_sn >= sn_min) | ~(halpha_sn >= sn_min)
    metallicity[invalid] = np.nan
    return metallicity


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def gaussian(x, amp, mu, sigma):
    """Amplitude times a normalized Gaussian (amp = integrated line flux)."""
    return amp / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def available_lines(group_lines, z, wave_min, wave_max, pad=15.0):
    """Return the subset of group_lines whose redshifted wavelength (+-pad) lies within [wave_min, wave_max]."""
    avail = []
    for name in group_lines:
        obs = LINES[name] * (1 + z)
        if (obs - pad) >= wave_min and (obs + pad) <= wave_max:
            avail.append(name)
    return avail


def group_wave_window(group_lines, z, pad):
    """
    Observed-frame [wave_lo, wave_hi] spanning the bluest-to-reddest member
    of group_lines at redshift z, padded by pad Angstrom on each side.
    Shared by fit_spaxel's per-spaxel fit window and fit_cube's narrowband
    fit-restriction-mask window (see determine_group_sigma) -- keeps both
    "whole-complex span" computations mechanically identical.
    """
    obs = [LINES[name] * (1 + z) for name in group_lines]
    return min(obs) - pad, max(obs) + pad


def build_model(line_names, tie_doublets, sigma_fixed=None, poly_order=1,
                 wave_ref=0.0, wave_scale=1.0):
    """
    Build a model function for a set of lines, ALL sharing the same single
    `sigma` (one line width per complex) and the same `z` (one redshift per
    complex), on top of a degree-`poly_order` continuum polynomial in
    x = (wave - wave_ref) / wave_scale (default poly_order=1 reproduces the
    original linear continuum: cont0 + cont1*x).

    By default (sigma_fixed=None) sigma is a free fit parameter:
    model(wave, z, *cont_coeffs, sigma, *free_amps, *ratios). When
    sigma_fixed is given, sigma is NOT a fit parameter at all -- it's baked
    into the model as a constant, and the returned model_fn instead takes
    model(wave, z, *cont_coeffs, *free_amps, *ratios) -- a genuinely smaller
    parameter vector, not just a tightly bounded one. This is how fit_cube
    holds sigma fixed at a per-group value determined once from the
    brightest spaxels (see determine_group_sigma) rather than re-fitting it
    independently (and un-constrainedly) at every spaxel -- this dataset's
    spectral resolution doesn't resolve any intrinsic line width, so sigma is
    purely instrumental+atmospheric and has no business varying spaxel to
    spaxel. cont_coeffs has poly_order+1 entries, increasing power order
    (numpy.polynomial.polynomial convention): cont0 + cont1*x + cont2*x**2 +
    ... -- see fit_continuum_polynomial for the matching stage-A pre-fit
    that seeds these as fit_spaxel's initial guess.

    wave_ref/wave_scale are fixed by the caller rather than recomputed from
    whatever wave array a given model_fn call happens to receive, so a
    stage-A continuum-only pre-fit (fit_continuum_polynomial) and this
    function's joint fit agree on the same x basis.

    When tie_doublets is True, a fixed-atomic-ratio doublet's weak member (if
    present in line_names) is not given its own free amplitude; its flux is
    derived from the strong member via the fixed ratio (DOUBLET_RATIOS).
    `derived` maps such a tied weak line name to (strong line name, ratio).

    Independently of tie_doublets, a density-sensitive doublet's weak member
    (RATIO_BOUNDS; e.g. [OII] 3729, [SII] 6716), when its strong partner is
    also present in line_names, is likewise not given its own free [0, inf)
    amplitude -- instead its flux is the strong member's amplitude times a
    ratio parameter bounded to RATIO_BOUNDS' [lo, hi]. This is a physical
    bound rather than a simplifying tie, so it's always active. `ratio_lines`
    gives the order these ratio parameters appear in *after* `free_amps`
    (built from what's left of `free_lines` once both kinds of tied/derived
    lines are removed).

    Returns
    -------
    model_fn, free_lines (list[str], the amplitude parameters),
    derived (dict[str, tuple[str, float]]),
    ratio_lines (list[str], the weak member of each active density-sensitive
    pair, in the order their ratio parameters appear after free_lines' amps)
    """
    derived = {}
    free_lines = []
    for name in line_names:
        if tie_doublets and name in DOUBLET_RATIOS:
            strong, ratio = DOUBLET_RATIOS[name]
            if strong in line_names:
                derived[name] = (strong, ratio)
                continue
        free_lines.append(name)

    ratio_lines = [name for name in free_lines
                   if name in RATIO_BOUNDS and RATIO_BOUNDS[name][0] in free_lines]
    free_lines = [name for name in free_lines if name not in ratio_lines]

    n_cont = poly_order + 1

    def _flux(wave, z, cont_coeffs, sigma, params):
        amp_of = dict(zip(free_lines, params[:len(free_lines)]))
        ratio_of = dict(zip(ratio_lines, params[len(free_lines):]))
        x = (wave - wave_ref) / wave_scale
        flux = np.zeros_like(wave, dtype=float)
        for k, c in enumerate(cont_coeffs):
            flux = flux + c * x ** k
        for name in line_names:
            if name in derived:
                strong, ratio = derived[name]
                amp = ratio * amp_of[strong]
            elif name in ratio_of:
                strong = RATIO_BOUNDS[name][0]
                amp = ratio_of[name] * amp_of[strong]
            else:
                amp = amp_of[name]
            flux = flux + gaussian(wave, amp, LINES[name] * (1 + z), sigma)
        return flux

    if sigma_fixed is None:
        def model_fn(wave, z, *rest):
            cont_coeffs, sigma, params = rest[:n_cont], rest[n_cont], rest[n_cont + 1:]
            return _flux(wave, z, cont_coeffs, sigma, params)
    else:
        def model_fn(wave, z, *rest):
            cont_coeffs, params = rest[:n_cont], rest[n_cont:]
            return _flux(wave, z, cont_coeffs, sigma_fixed, params)

    return model_fn, free_lines, derived, ratio_lines


def fit_continuum_polynomial(wave, flux, var, group_lines, z, sigma_for_mask, poly_order,
                              wave_ref, wave_scale, mask_halfwidth_sigma=4.0):
    """
    Stage-A pre-fit: robust weighted-linear-least-squares polynomial fit to
    the continuum ALONE, with every line in group_lines masked out (+/-
    mask_halfwidth_sigma * sigma_for_mask around its redshifted center).
    wave/flux/var are assumed already restricted to finite, good pixels
    within the group's fit window (see fit_spaxel) -- this function only
    additionally excludes the line windows.

    Used solely to seed fit_spaxel's stage-B joint (continuum+lines)
    curve_fit with a good initial guess for the continuum coefficients --
    stage B still refits them freely alongside the lines, so this is not the
    final continuum solution. Doing this in two stages (rather than seeding
    the joint fit with a flat guess, as before) matters once poly_order > 1:
    curve_fit's local optimizer can otherwise settle on a poor local minimum
    for the higher-order terms, especially at lower S/N.

    Returns poly_order+1 coefficients in the SAME (wave-wave_ref)/wave_scale
    basis build_model's continuum uses (increasing power order). Falls back
    to a flat (median-level) guess if too few continuum pixels remain after
    masking out the lines.
    """
    line_centers = [LINES[name] * (1 + z) for name in group_lines]
    line_mask = np.zeros(wave.shape, dtype=bool)
    for c in line_centers:
        line_mask |= np.abs(wave - c) <= mask_halfwidth_sigma * sigma_for_mask
    cont_only = ~line_mask

    coeffs = np.zeros(poly_order + 1)
    if cont_only.sum() < poly_order + 2:
        coeffs[0] = np.median(flux) if flux.size else 0.0
        return coeffs

    x = (wave[cont_only] - wave_ref) / wave_scale
    lsq_weights = 1.0 / np.sqrt(var[cont_only])
    return np.polynomial.polynomial.polyfit(x, flux[cont_only], poly_order, w=lsq_weights)


# ----------------------------------------------------------------------
# Per-spaxel fit
# ----------------------------------------------------------------------
def fit_spaxel(wave, flux, var, weight, group_lines, z, dz_max, sigma_init,
               sigma_bounds, tie_doublets, window_pad=60.0, min_good_pix=10,
               sigma_fixed=None, poly_order=3, continuum_mask_halfwidth_sigma=4.0):
    """
    Fit one spaxel's spectrum for a given line group.

    sigma_fixed: when given, sigma is held fixed at this value (not a fit
    parameter at all -- see build_model) and sigma_init/sigma_bounds are
    ignored. When None (default), sigma is fit freely as before, bounded by
    sigma_init/sigma_bounds. Both call shapes are otherwise identical, so the
    same call site can be reused for a free-sigma pre-fit (sigma_fixed
    omitted) and the main, sigma-held-fixed fitting pass (see fit_cube).

    The continuum is a degree-poly_order polynomial (default 3, i.e. cubic --
    a plain linear continuum (poly_order=1) systematically failed to track
    real curvature/slope even at high S/N, confirmed by eye against the
    halpha_nii complex's own per-spaxel fit diagnostics). Fitting proceeds in
    two stages: (a) fit_continuum_polynomial fits the continuum ALONE to the
    line-masked pixels (+/- continuum_mask_halfwidth_sigma * sigma around
    each line), (b) that result seeds the initial guess for the full joint
    (continuum + lines) curve_fit below, which still refits every continuum
    coefficient freely -- stage A only prevents the joint optimizer from
    settling on a poor local minimum for the higher-order terms, it is not
    itself the final continuum.

    Returns a dict with keys: 'amp', 'amp_err' (dict per line in group_lines),
    'sigma', 'z', 'cont0'/'cont1' (first two continuum coefficients, kept for
    backward compatibility), 'cont_coeffs'/'cont_coeffs_err' (all poly_order+1
    coefficients and their errors), 'wave_ref'/'wave_scale' (the continuum's
    x-basis, needed to evaluate cont_coeffs), 'chi2_reduced', 'success'.
    """
    result = {
        "amp": {name: np.nan for name in group_lines},
        "amp_err": {name: np.nan for name in group_lines},
        "sigma": np.nan,
        "z": np.nan,
        "cont0": np.nan,
        "cont0_err": np.nan,
        "cont1": np.nan,
        "cont1_err": np.nan,
        "cont_coeffs": None,
        "cont_coeffs_err": None,
        "wave_ref": np.nan,
        "wave_scale": np.nan,
        "chi2_reduced": np.nan,
        "success": False,
    }
    if len(group_lines) == 0:
        return result

    wlo, whi = group_wave_window(group_lines, z, window_pad)
    sel = (wave >= wlo) & (wave <= whi)

    w = wave[sel]
    f = flux[sel]
    v = var[sel]
    wt = weight[sel]

    good = np.isfinite(f) & np.isfinite(v) & (v > 0) & (wt > 0)
    if good.sum() < min_good_pix:
        return result

    w, f, v = w[good], f[good], v[good]
    sigma_y = np.sqrt(v)

    n_cont = poly_order + 1
    wave_ref = float(w.mean())
    wave_scale = max(0.5 * float(w.max() - w.min()), 1e-6)
    sigma_for_mask = sigma_fixed if sigma_fixed is not None else sigma_init
    cont_p0 = fit_continuum_polynomial(w, f, v, group_lines, z, sigma_for_mask, poly_order,
                                        wave_ref, wave_scale,
                                        mask_halfwidth_sigma=continuum_mask_halfwidth_sigma)

    model_fn, free_lines, derived, ratio_lines = build_model(
        group_lines, tie_doublets, sigma_fixed=sigma_fixed, poly_order=poly_order,
        wave_ref=wave_ref, wave_scale=wave_scale)
    n_amp, n_ratio = len(free_lines), len(ratio_lines)
    n_lead = 1 + n_cont + (1 if sigma_fixed is None else 0)

    # Ratio params start at 1.0 -- a neutral, roughly-central guess within
    # every RATIO_BOUNDS pair's [lo, hi] (both are currently [0.35, 1.5]).
    if sigma_fixed is None:
        lead_p0 = [z] + list(cont_p0) + [sigma_init]
        lead_lower = [z - dz_max] + [-np.inf] * n_cont + [sigma_bounds[0]]
        lead_upper = [z + dz_max] + [np.inf] * n_cont + [sigma_bounds[1]]
    else:
        lead_p0 = [z] + list(cont_p0)
        lead_lower = [z - dz_max] + [-np.inf] * n_cont
        lead_upper = [z + dz_max] + [np.inf] * n_cont

    p0 = lead_p0 + [max(np.ptp(f), 1.0)] * n_amp + [1.0] * n_ratio
    lower = lead_lower + [0.0] * n_amp + [RATIO_BOUNDS[name][1] for name in ratio_lines]
    upper = lead_upper + [np.inf] * n_amp + [RATIO_BOUNDS[name][2] for name in ratio_lines]

    try:
        popt, pcov = curve_fit(
            model_fn, w, f, p0=p0, sigma=sigma_y, absolute_sigma=True,
            bounds=(lower, upper), max_nfev=5000,
        )
    except (RuntimeError, ValueError):
        return result

    if not np.all(np.isfinite(pcov)):
        return result

    perr = np.sqrt(np.diag(pcov))
    z_fit = popt[0]
    cont_fit = popt[1:1 + n_cont]
    cont_err = perr[1:1 + n_cont]
    if sigma_fixed is None:
        sigma_fit = popt[1 + n_cont]
    else:
        sigma_fit = sigma_fixed
    amp_idx = {name: n_lead + i for i, name in enumerate(free_lines)}
    ratio_idx = {name: n_lead + n_amp + j for j, name in enumerate(ratio_lines)}
    amp_fit = {name: popt[i] for name, i in amp_idx.items()}
    amp_err_fit = {name: perr[i] for name, i in amp_idx.items()}

    for name in group_lines:
        if name in derived:
            # fixed atomic ratio -- a constant, so linear error propagation
            # (Var(c*X) = c^2 Var(X)) is exact.
            strong, ratio = derived[name]
            result["amp"][name] = ratio * amp_fit[strong]
            result["amp_err"][name] = ratio * amp_err_fit[strong]
        elif name in ratio_idx:
            # bounded ratio is itself a FITTED (correlated) parameter here,
            # so amp = ratio * amp_strong needs full error propagation for a
            # product of two correlated variables, using their covariance
            # from pcov -- not just independent-error combination.
            strong = RATIO_BOUNDS[name][0]
            i_amp, i_ratio = amp_idx[strong], ratio_idx[name]
            amp_strong, ratio_val = popt[i_amp], popt[i_ratio]
            var = (ratio_val ** 2 * pcov[i_amp, i_amp]
                   + amp_strong ** 2 * pcov[i_ratio, i_ratio]
                   + 2 * ratio_val * amp_strong * pcov[i_amp, i_ratio])
            result["amp"][name] = ratio_val * amp_strong
            result["amp_err"][name] = np.sqrt(max(var, 0.0))
        else:
            result["amp"][name] = amp_fit[name]
            result["amp_err"][name] = amp_err_fit[name]

    resid = f - model_fn(w, *popt)
    dof = max(len(w) - len(popt), 1)
    result["sigma"] = sigma_fit
    result["z"] = z_fit
    result["cont0"] = cont_fit[0]
    result["cont0_err"] = cont_err[0]
    result["cont1"] = cont_fit[1] if n_cont > 1 else 0.0
    result["cont1_err"] = cont_err[1] if n_cont > 1 else 0.0
    result["cont_coeffs"] = cont_fit
    result["cont_coeffs_err"] = cont_err
    result["wave_ref"] = wave_ref
    result["wave_scale"] = wave_scale
    result["chi2_reduced"] = np.sum((resid / sigma_y) ** 2) / dof
    result["success"] = True
    return result


# ----------------------------------------------------------------------
# Per-group fit-restriction mask + fixed instrumental sigma
# ----------------------------------------------------------------------
def select_topk_by_snr(fit_mask, snr_image, topk_frac=0.05, min_topk=1):
    """
    (iy, ix) index arrays of the top topk_frac (by count, floored at
    min_topk, capped at the number of finite-S/N True spaxels) of fit_mask's
    True spaxels, ranked by snr_image descending. Used to pick the
    highest-S/N spaxels for determine_group_sigma -- ranks by the
    narrowband image's own S/N (fit-free), not any fitted quantity.

    Returns (empty, empty) if fit_mask has no spaxels with finite S/N.
    """
    iy_all, ix_all = np.nonzero(fit_mask)
    sn_all = snr_image[iy_all, ix_all]
    finite = np.isfinite(sn_all)
    iy_all, ix_all, sn_all = iy_all[finite], ix_all[finite], sn_all[finite]
    n_valid = len(sn_all)
    if n_valid == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_topk = min(n_valid, max(min_topk, int(np.ceil(topk_frac * n_valid))))
    order = np.argsort(sn_all)[::-1][:n_topk]
    return iy_all[order], ix_all[order]


def determine_group_sigma(wave, flux, var, weight, avail, z, dz_max, sigma_init,
                           sigma_bounds, tie_doublets, window_pad, topk_iy, topk_ix,
                           poly_order=3, continuum_mask_halfwidth_sigma=4.0):
    """
    Free-sigma pre-fit at each (topk_iy[k], topk_ix[k]) spaxel individually
    (via fit_spaxel with sigma_fixed=None, i.e. the ordinary free-sigma
    path) -- deliberately NOT stacked/co-added, since the top-S/N spaxels
    can span real velocity structure (e.g. galaxy rotation) that would
    artificially broaden a naively summed spectrum.

    Returns (sigma_fixed, sigma_values): sigma_fixed = float(np.median(the
    successful fits' sigma values)), the single value fit_cube then holds
    fixed for every spaxel in this group's main fitting pass. Returns
    (None, empty array) if none of the pre-fits succeeded, so the caller can
    skip the whole group.
    """
    sigma_values = []
    for iy, ix in zip(topk_iy, topk_ix):
        res = fit_spaxel(wave, flux[:, iy, ix], var[:, iy, ix], weight[:, iy, ix],
                          avail, z, dz_max, sigma_init, sigma_bounds, tie_doublets,
                          window_pad=window_pad, poly_order=poly_order,
                          continuum_mask_halfwidth_sigma=continuum_mask_halfwidth_sigma)
        if res["success"]:
            sigma_values.append(res["sigma"])
    sigma_values = np.array(sigma_values, dtype=float)
    if sigma_values.size == 0:
        return None, sigma_values
    return float(np.median(sigma_values)), sigma_values


def plot_sigma_histogram(sigma_values, sigma_fixed, outdir, target_name, group_name):
    """
    Histogram PNG of determine_group_sigma's per-spaxel sigma values (each
    from an independent free-sigma fit at one of the group's highest-S/N
    narrowband spaxels), with a dashed line at the median -- the value then
    held FIXED for every spaxel in this group's main fitting pass (see
    fit_cube).
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = min(20, max(5, sigma_values.size))
    ax.hist(sigma_values, bins=bins, color="steelblue", edgecolor="black")
    ax.axvline(sigma_fixed, color="red", ls="--", lw=2, label=f"median = {sigma_fixed:.3f} Angstrom")
    ax.set_xlabel("Fitted line width sigma (Angstrom)")
    ax.set_ylabel("Number of spaxels")
    ax.set_title(f"{target_name} -- {group_name} sigma distribution "
                 f"(top-S/N spaxels, N={sigma_values.size})")
    ax.legend()
    out_path = os.path.join(outdir, f"{target_name}_{group_name}_sigma_hist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def plot_fit_selection_diagnostic(nb_img, fit_mask, topk_iy, topk_ix, outdir, target_name, group_name):
    """
    2D diagnostic PNG (native cube pixel orientation, no WCS projection --
    this is an internal fitting diagnostic, not a final science plot):
    the group's narrowband image (see compute_narrowband_image) as
    background, with three regions visually distinct -- excluded from
    fitting entirely (outside fit_mask, see mad_background_mask), included
    but not used for the sigma pre-fit, and the top-S/N pixels actually used
    to determine sigma_fixed (see determine_group_sigma/select_topk_by_snr).
    """
    os.makedirs(outdir, exist_ok=True)
    ny, nx = nb_img.shape

    overlay = np.zeros((ny, nx), dtype=np.uint8)  # 0=excluded, 1=included, 2=top-S/N
    overlay[fit_mask] = 1
    overlay[topk_iy, topk_ix] = 2

    from matplotlib.colors import ListedColormap
    cmap_overlay = ListedColormap(["0.5", "none", "red"])

    fig, ax = plt.subplots(figsize=(7, 7))
    finite = np.isfinite(nb_img)
    vmax = np.nanpercentile(nb_img[finite], 99) if np.any(finite) else 1.0
    ax.imshow(np.nan_to_num(nb_img, nan=0.0), origin="lower", cmap="viridis",
              vmin=0, vmax=max(vmax, 1e-10))
    ax.imshow(overlay, origin="lower", cmap=cmap_overlay, vmin=0, vmax=2, alpha=0.55)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(f"{target_name} -- {group_name} fit selection\n"
                 "(gray = excluded from fit, red = top-S/N spaxels used for sigma)")

    out_path = os.path.join(outdir, f"{target_name}_{group_name}_fit_selection.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Full-cube fit
# ----------------------------------------------------------------------
def fit_cube(cube_path, z, tie_doublets=True, dv_max_kms=300.0, outdir=".",
             target_name=None, window_pad=60.0,
             continuum_wave_min=6400.0, continuum_wave_max=6700.0,
             wave_range=None, include_continuum=True,
             mad_sigma=2.5, narrowband_pad=30.0, sigma_topk_frac=0.05, sigma_topk_min=1,
             continuum_poly_order=3, continuum_mask_halfwidth_sigma=4.0):
    """
    Fit all line complexes (LINE_GROUPS) in one LLAMAS channel cube, plus
    (optionally) a broadband continuum image (see compute_broadband_continuum).

    Which lines actually get fit is determined per-group by
    available_lines() against this cube's own (possibly truncated, see
    wave_range) wavelength coverage -- a group missing some of its members in
    this cube is still fit using whichever members ARE present, so a single
    target's line groups may end up assembled from more than one channel
    cube -- see fit_target, which orchestrates that across blue/green/red and
    merges the results.

    For each available group, fitting proceeds in three stages rather than
    a single free-for-all per-spaxel pass over the whole (ny, nx) grid:

    1. A fit-free narrowband image (compute_broadband_continuum, collapsed
       over group_wave_window(avail, z, narrowband_pad)) is masked with
       mad_background_mask(nb_img, sigma_multiplier=mad_sigma) -- the same
       masking astrometry_correction.py uses on its own line/continuum
       images before correlation, empirically confirmed to cleanly suppress
       this dataset's banding artifact where masks built from the FITTED
       amplitude do not. Spaxels outside this mask are skipped entirely --
       curve_fit is never even called there, not fit-then-discarded.
    2. The instrumental line width sigma is determined ONCE for the whole
       group: the narrowband image's own S/N picks the top sigma_topk_frac
       of masked spaxels (select_topk_by_snr), each is fit individually with
       sigma free (determine_group_sigma), and the median of those fits
       becomes sigma_fixed. This dataset's spectral resolution doesn't
       resolve any intrinsic line width, so sigma is purely instrumental and
       has no business varying spaxel to spaxel or being independently
       (weakly) constrained at every low-S/N pixel. A histogram of the
       per-spaxel pre-fit values (plot_sigma_histogram) and a 2D map of the
       mask/top-S/N-selection (plot_fit_selection_diagnostic) are written as
       diagnostics.
    3. Every masked spaxel (the original top-S/N ones included, for uniform
       treatment) is fit with sigma held FIXED at sigma_fixed (see
       build_model/fit_spaxel's sigma_fixed) -- genuinely fewer free
       parameters, not just a tight bound, which also stabilizes the
       density-sensitive-doublet ratio fit (RATIO_BOUNDS) at high S/N.

    Parameters
    ----------
    wave_range : (float or None, float or None), optional
        If given, restrict the fit (and hence available_lines' coverage
        check) to this [wave_min, wave_max] observed-frame Angstrom window
        before anything else -- e.g. a throughput_wave_range() result, to
        keep the fit away from a channel's low-throughput edges. Either side
        may be None to leave that side unbounded (the cube's own native
        min/max).
    include_continuum : bool
        Whether to compute and attach the broadband continuum image. Disable
        for a channel that doesn't cover continuum_wave_min/
        continuum_wave_max (e.g. a blue-channel-only fit used solely for
        [OII]) to avoid a spurious ValueError; fit_target computes the
        continuum image separately, once, from a chosen channel.
    mad_sigma : float
        sigma_multiplier passed to mad_background_mask for each group's
        fit-restriction mask.
    narrowband_pad : float
        Padding (Angstrom, each side) added to a group's bluest-to-reddest
        line span (group_wave_window) when building its narrowband image.
        Should not exceed window_pad/2 -- available_lines' own pad=
        window_pad/2 check already guarantees every line in avail sits that
        far inside the channel's throughput-truncated coverage, so keeping
        narrowband_pad within that bound guarantees the narrowband image
        (which reloads the raw, untruncated cube) never reaches into
        throughput-untrustworthy edge pixels the real fit wouldn't see.
    sigma_topk_frac, sigma_topk_min : float, int
        Passed to select_topk_by_snr for the sigma pre-fit's spaxel
        selection.

    Returns
    -------
    maps : dict[str, dict[str, ndarray]]
        maps[line_name] = {'flux': 2D array, 'flux_err': 2D array}
        Plus maps['_meta'][group_name] = {'sigma', 'z', 'chi2_reduced',
        'cont0', 'cont0_err', 'cont1', 'cont1_err'}, each a 2D array
        Plus maps['_continuum_img'] = {'flux': 2D array, 'flux_err': 2D array}
        (only if include_continuum)
        Plus maps['_metallicity'] = 2D array (see compute_metallicity_map)
    wcs : astropy.wcs.WCS (celestial part usable for plotting)
    """
    if target_name is None:
        target_name = os.path.basename(os.path.dirname(cube_path.rstrip("/"))) or "target"

    wave, flux, var, weight, wcs, header = load_cube(cube_path)
    if wave_range is not None:
        wlo, whi = wave_range
        wlo = -np.inf if wlo is None else wlo
        whi = np.inf if whi is None else whi
        sel = (wave >= wlo) & (wave <= whi)
        if not np.any(sel):
            raise ValueError(f"{cube_path}: wave_range {wave_range} excludes this cube's entire "
                              f"native coverage [{wave.min():.1f}, {wave.max():.1f}]")
        wave, flux, var, weight = wave[sel], flux[sel], var[sel], weight[sel]

    nwave, ny, nx = flux.shape
    wave_min, wave_max = wave.min(), wave.max()
    dz_max = dv_max_kms / C_KMS * (1 + z)

    cdelt3 = header["CDELT3"]
    sigma_init = 2 * cdelt3
    sigma_bounds = (0.5 * cdelt3, 10 * cdelt3)

    maps = {name: {"flux": np.full((ny, nx), np.nan),
                    "flux_err": np.full((ny, nx), np.nan)}
            for name in LINES}
    meta = {}

    for group_name, group_lines in LINE_GROUPS.items():
        avail = available_lines(group_lines, z, wave_min, wave_max, pad=window_pad / 2)
        print(f"[{group_name}] available lines at z={z}: {avail}")
        if not avail:
            continue

        nb_lo, nb_hi = group_wave_window(avail, z, narrowband_pad)
        try:
            nb_img, nb_img_err, _ = compute_broadband_continuum(cube_path, nb_lo, nb_hi)
        except ValueError:
            print(f"[{group_name}] no coverage for narrowband window "
                  f"[{nb_lo:.1f}, {nb_hi:.1f}], skipping group")
            continue
        fit_mask = mad_background_mask(nb_img, sigma_multiplier=mad_sigma)
        n_fit = int(np.count_nonzero(fit_mask))
        print(f"[{group_name}] fit-restriction mask: {n_fit}/{fit_mask.size} spaxels")
        if n_fit == 0:
            print(f"[{group_name}] mad_background_mask selected 0 spaxels, skipping group")
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            nb_sn = nb_img / nb_img_err
        topk_iy, topk_ix = select_topk_by_snr(fit_mask, nb_sn, topk_frac=sigma_topk_frac,
                                               min_topk=sigma_topk_min)
        sigma_fixed, sigma_values = determine_group_sigma(
            wave, flux, var, weight, avail, z, dz_max, sigma_init, sigma_bounds,
            tie_doublets, window_pad, topk_iy, topk_ix,
            poly_order=continuum_poly_order,
            continuum_mask_halfwidth_sigma=continuum_mask_halfwidth_sigma)
        if sigma_fixed is None:
            print(f"[{group_name}] no successful top-S/N pre-fits, skipping group")
            continue
        print(f"[{group_name}] sigma_fixed = {sigma_fixed:.4f} Angstrom "
              f"(median of {sigma_values.size} top-S/N spaxel fits)")
        plot_sigma_histogram(sigma_values, sigma_fixed, outdir, target_name, group_name)
        plot_fit_selection_diagnostic(nb_img, fit_mask, topk_iy, topk_ix, outdir, target_name, group_name)

        sigma_map = np.full((ny, nx), np.nan)
        z_map = np.full((ny, nx), np.nan)
        chi2_map = np.full((ny, nx), np.nan)
        cont0_map = np.full((ny, nx), np.nan)
        cont0_err_map = np.full((ny, nx), np.nan)
        cont1_map = np.full((ny, nx), np.nan)
        cont1_err_map = np.full((ny, nx), np.nan)

        fit_iy, fit_ix = np.nonzero(fit_mask)
        for k, (iy, ix) in enumerate(zip(fit_iy, fit_ix)):
            res = fit_spaxel(
                wave, flux[:, iy, ix], var[:, iy, ix], weight[:, iy, ix],
                avail, z, dz_max, sigma_init, sigma_bounds, tie_doublets,
                window_pad=window_pad, sigma_fixed=sigma_fixed,
                poly_order=continuum_poly_order,
                continuum_mask_halfwidth_sigma=continuum_mask_halfwidth_sigma,
            )
            if not res["success"]:
                continue
            for name in avail:
                maps[name]["flux"][iy, ix] = res["amp"][name]
                maps[name]["flux_err"][iy, ix] = res["amp_err"][name]
            sigma_map[iy, ix] = res["sigma"]
            z_map[iy, ix] = res["z"]
            chi2_map[iy, ix] = res["chi2_reduced"]
            cont0_map[iy, ix] = res["cont0"]
            cont0_err_map[iy, ix] = res["cont0_err"]
            cont1_map[iy, ix] = res["cont1"]
            cont1_err_map[iy, ix] = res["cont1_err"]
            if (k + 1) % 200 == 0 or (k + 1) == n_fit:
                print(f"  [{group_name}] spaxel {k + 1}/{n_fit} done", end="\r")
        print()

        meta[group_name] = {
            "sigma": sigma_map, "z": z_map, "chi2_reduced": chi2_map,
            "cont0": cont0_map, "cont0_err": cont0_err_map,
            "cont1": cont1_map, "cont1_err": cont1_err_map,
        }

    maps["_meta"] = meta
    if include_continuum:
        cont_img, cont_img_err, _ = compute_broadband_continuum(
            cube_path, continuum_wave_min, continuum_wave_max)
        maps["_continuum_img"] = {"flux": cont_img, "flux_err": cont_img_err}
    maps["_metallicity"] = compute_metallicity_map(
        maps["NII_6584"]["flux"], maps["Halpha"]["flux"],
        maps["NII_6584"]["flux_err"], maps["Halpha"]["flux_err"],
    )
    maps["_wcs"] = wcs
    maps["_target"] = target_name
    return maps


def merge_channel_maps(per_channel_maps):
    """
    Merge {channel: fit_cube(...)-shaped dict} (one entry per blue/green/red
    channel already fit via fit_target) into one maps dict spanning all of
    LINES.

    Per-line flux/flux_err are combined by inverse-variance weighting across
    any channels that both measured the same spaxel+line -- in practice, at
    most one channel does for a given target/redshift once throughput
    truncation removes the wavelength overlap between channels (see
    throughput_wave_range), but this stays correct even if it doesn't.
    Per-group _meta entries are kept separately, prefixed by channel, since
    fit diagnostics like sigma/z/chi2/continuum are channel-specific and
    shouldn't be blended.

    Does NOT set _continuum_img/_metallicity/_wcs/_target -- fit_target adds
    those after calling this.
    """
    channels = list(per_channel_maps)
    if not channels:
        raise ValueError("no channel maps to merge")

    sample_line = next(iter(LINES))
    ny, nx = per_channel_maps[channels[0]][sample_line]["flux"].shape

    merged = {name: {"flux": np.full((ny, nx), np.nan), "flux_err": np.full((ny, nx), np.nan)}
              for name in LINES}
    for name in LINES:
        num = np.zeros((ny, nx))
        den = np.zeros((ny, nx))
        any_valid = np.zeros((ny, nx), dtype=bool)
        for ch in channels:
            f = per_channel_maps[ch][name]["flux"]
            e = per_channel_maps[ch][name]["flux_err"]
            valid = np.isfinite(f) & np.isfinite(e) & (e > 0)
            ivar = np.where(valid, 1.0 / e ** 2, 0.0)
            num = num + np.where(valid, f * ivar, 0.0)
            den = den + ivar
            any_valid |= valid
        merged[name]["flux"][any_valid] = (num / np.where(den > 0, den, 1.0))[any_valid]
        merged[name]["flux_err"][any_valid] = np.sqrt(1.0 / den[any_valid])

    meta = {}
    for ch in channels:
        for group_name, gmaps in per_channel_maps[ch]["_meta"].items():
            meta[f"{ch}_{group_name}"] = gmaps
    merged["_meta"] = meta
    return merged


def fit_target(cube_paths, z, sensfunc_paths=None, truncate_frac=0.5,
                wave_range_overrides=None,
                tie_doublets=True, dv_max_kms=300.0, outdir=".", target_name=None,
                window_pad=60.0, continuum_channel=None,
                continuum_wave_min=6400.0, continuum_wave_max=6700.0,
                mad_sigma=2.5, narrowband_pad=30.0, sigma_topk_frac=0.05, sigma_topk_min=1,
                continuum_poly_order=3, continuum_mask_halfwidth_sigma=4.0):
    """
    Fit emission lines across one or more LLAMAS channel cubes (blue/green/
    red) for a single target and merge the results into one maps dict
    spanning every line in LINES. See fit_cube for how a single channel's fit
    works (including the fit-restriction mask and fixed-sigma mechanism);
    this just runs it once per channel present in cube_paths and merges via
    merge_channel_maps -- e.g. sii_doublet is naturally picked up from
    whichever channel (green or red) covers it at this redshift, and
    oii_doublet/hbeta_oiii/halpha_nii are similarly each assembled
    independently, each with its own z/sigma solution (see LINE_GROUPS).

    mad_sigma, narrowband_pad, sigma_topk_frac, sigma_topk_min are forwarded
    to every fit_cube(...) call -- see there for what each controls.

    Parameters
    ----------
    cube_paths : dict[str, str]
        {"blue": path, "green": path, "red": path} -- any subset; a missing
        channel is simply not fit.
    sensfunc_paths : dict[str, str], optional
        {"blue": path, "green": path, "red": path} PypeIt sensfunc FITS files
        (see throughput_wave_range) used to truncate each channel present in
        both dicts. A channel in cube_paths but missing here is fit over its
        full native wavelength range.
    truncate_frac : float or None
        Passed to throughput_wave_range as frac_of_peak. None disables
        truncation entirely (every channel uses its full native range).
    wave_range_overrides : dict[str, (float or None, float or None)], optional
        Per-channel (wave_min, wave_max) that overrides the throughput-derived
        bound on that side only -- e.g. {"green": (None, 7000.0)} keeps
        green's throughput-derived lower bound but replaces its upper bound
        with a fixed 7000 Angstrom regardless of where throughput crosses
        truncate_frac there. A None entry on either side means "no override,
        use the throughput-derived bound for that side" (or fully unbounded,
        if truncation is off/no sensfunc for that channel).
    continuum_channel : str, optional
        Which channel's cube to use for the broadband continuum image.
        Default: "green" if present, else the first channel in cube_paths.

    Returns
    -------
    maps : same shape as fit_cube's return value.
    """
    if not cube_paths:
        raise ValueError("cube_paths is empty -- need at least one channel")
    if target_name is None:
        any_path = next(iter(cube_paths.values()))
        target_name = os.path.basename(os.path.dirname(any_path.rstrip("/"))) or "target"

    per_channel_maps = {}
    for channel, cube_path in cube_paths.items():
        wave_range = None
        if truncate_frac is not None and sensfunc_paths and channel in sensfunc_paths:
            wave_range = throughput_wave_range(sensfunc_paths[channel], frac_of_peak=truncate_frac)

        if wave_range_overrides and channel in wave_range_overrides:
            lo_over, hi_over = wave_range_overrides[channel]
            lo = lo_over if lo_over is not None else (wave_range[0] if wave_range else None)
            hi = hi_over if hi_over is not None else (wave_range[1] if wave_range else None)
            wave_range = (lo, hi)

        if wave_range is not None:
            lo_str = "native" if wave_range[0] is None else f"{wave_range[0]:.1f}"
            hi_str = "native" if wave_range[1] is None else f"{wave_range[1]:.1f}"
            print(f"[{channel}] using wavelength range [{lo_str}, {hi_str}] Angstrom")

        per_channel_maps[channel] = fit_cube(
            cube_path, z, tie_doublets=tie_doublets, dv_max_kms=dv_max_kms,
            outdir=outdir, target_name=target_name, window_pad=window_pad,
            wave_range=wave_range, include_continuum=False,
            mad_sigma=mad_sigma, narrowband_pad=narrowband_pad,
            sigma_topk_frac=sigma_topk_frac, sigma_topk_min=sigma_topk_min,
            continuum_poly_order=continuum_poly_order,
            continuum_mask_halfwidth_sigma=continuum_mask_halfwidth_sigma,
        )

    merged = merge_channel_maps(per_channel_maps)

    continuum_channel = continuum_channel or ("green" if "green" in cube_paths else next(iter(cube_paths)))
    cont_img, cont_img_err, _ = compute_broadband_continuum(
        cube_paths[continuum_channel], continuum_wave_min, continuum_wave_max)
    merged["_continuum_img"] = {"flux": cont_img, "flux_err": cont_img_err}

    merged["_metallicity"] = compute_metallicity_map(
        merged["NII_6584"]["flux"], merged["Halpha"]["flux"],
        merged["NII_6584"]["flux_err"], merged["Halpha"]["flux_err"],
    )
    merged["_wcs"] = per_channel_maps[continuum_channel]["_wcs"]
    merged["_target"] = target_name
    return merged


# ----------------------------------------------------------------------
# Output: FITS + PNGs
# ----------------------------------------------------------------------
def save_fits(maps, outdir, target_name):
    os.makedirs(outdir, exist_ok=True)
    hdus = [fits.PrimaryHDU()]
    wcs2d = maps["_wcs"].celestial
    header2d = wcs2d.to_header()

    for name in LINES:
        if np.all(np.isnan(maps[name]["flux"])):
            continue
        hdus.append(fits.ImageHDU(maps[name]["flux"], header=header2d, name=f"{name}_FLUX"))
        hdus.append(fits.ImageHDU(maps[name]["flux_err"], header=header2d, name=f"{name}_ERR"))

    for group_name, gmaps in maps["_meta"].items():
        for key, arr in gmaps.items():
            hdus.append(fits.ImageHDU(arr, header=header2d, name=f"{group_name}_{key}".upper()[:68]))

    if "_continuum_img" in maps:
        hdus.append(fits.ImageHDU(maps["_continuum_img"]["flux"], header=header2d, name="CONTINUUM_IMG"))
        hdus.append(fits.ImageHDU(maps["_continuum_img"]["flux_err"], header=header2d, name="CONTINUUM_IMG_ERR"))

    if "_metallicity" in maps:
        hdus.append(fits.ImageHDU(maps["_metallicity"], header=header2d, name="METALLICITY"))

    out_path = os.path.join(outdir, f"{target_name}_linemaps.fits")
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    print(f"Wrote {out_path}")
    return out_path


def add_continuum_image(linemaps_path, cube_path, wave_min=6400.0, wave_max=6700.0):
    """
    Add/replace CONTINUUM_IMG/CONTINUUM_IMG_ERR extensions in an EXISTING
    *_linemaps.fits file, without re-running the (expensive) per-spaxel fit.
    """
    img, img_err, header2d = compute_broadband_continuum(cube_path, wave_min, wave_max)

    with fits.open(linemaps_path) as hdul:
        hdus = [hdu.copy() for hdu in hdul if hdu.name not in ("CONTINUUM_IMG", "CONTINUUM_IMG_ERR")]

    hdus.append(fits.ImageHDU(img, header=header2d, name="CONTINUUM_IMG"))
    hdus.append(fits.ImageHDU(img_err, header=header2d, name="CONTINUUM_IMG_ERR"))
    fits.HDUList(hdus).writeto(linemaps_path, overwrite=True)
    print(f"Updated {linemaps_path} with CONTINUUM_IMG/CONTINUUM_IMG_ERR")
    return linemaps_path


def add_metallicity_map(linemaps_path):
    """
    Add/replace the METALLICITY extension in an EXISTING *_linemaps.fits (or
    *_linemaps_corrected.fits) file from its NII_6584_FLUX/HALPHA_FLUX
    extensions (see compute_metallicity_map), without re-running the
    per-spaxel fit. Works on a corrected file too -- the metallicity value
    itself doesn't depend on WCS, only which pixel grid/orientation it's
    attached to.
    """
    with fits.open(linemaps_path) as hdul:
        nii_flux = hdul["NII_6584_FLUX"].data.astype(float)
        nii_err = hdul["NII_6584_ERR"].data.astype(float)
        halpha_flux = hdul["HALPHA_FLUX"].data.astype(float)
        halpha_err = hdul["HALPHA_ERR"].data.astype(float)
        header2d = hdul["HALPHA_FLUX"].header.copy()
        hdus = [hdu.copy() for hdu in hdul if hdu.name != "METALLICITY"]

    metallicity = compute_metallicity_map(nii_flux, halpha_flux, nii_err, halpha_err)
    hdus.append(fits.ImageHDU(metallicity, header=header2d, name="METALLICITY"))
    fits.HDUList(hdus).writeto(linemaps_path, overwrite=True)
    print(f"Updated {linemaps_path} with METALLICITY")
    return linemaps_path


def build_reference_mask(maps, reference_line=DEFAULT_REFERENCE_LINE, n_sigma=DEFAULT_MASK_SIGMA):
    """
    Boolean (ny, nx) "galaxy footprint" mask: True where reference_line is
    detected at S/N (flux/flux_err) >= n_sigma, False elsewhere.

    Halpha is the default reference line because it's normally the brightest,
    most robustly detected line feature, so thresholding on IT gives a much
    more reliable galaxy footprint than thresholding a fainter line directly.
    A fainter line near a channel's low-throughput edge (e.g. [SII] near the
    red channel's edge -- see throughput_wave_range) can show scattered,
    individually-S/N-passing "detections" from edge artifacting well outside
    the real galaxy extent (seen in FRB20240210A's [SII]6731 map). Callers
    use this mask to blank those spaxels on plots even though that line's own
    per-pixel S/N cut (see _display_and_norm's sn_min) would otherwise let
    them through.

    Returns None (i.e. "no mask available") if reference_line has no valid
    flux/flux_err in maps at all -- e.g. it fell outside every channel's
    wavelength coverage for this target/redshift -- so callers can fall back
    to unmasked plots rather than blanking everything.
    """
    ref = maps.get(reference_line)
    if ref is None:
        return None
    flux_map, err_map = ref.get("flux"), ref.get("flux_err")
    if flux_map is None or err_map is None or np.all(np.isnan(flux_map)):
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        sn = flux_map / err_map
    return sn >= n_sigma


def _display_and_norm(flux_map, err_map, sn_min, galaxy_mask=None):
    """
    Build a display array (S/N-masked, NaNs zero-filled for plotting) and a
    matching ImageNormalize, or (None, None) if nothing clears the S/N cut.

    galaxy_mask (see build_reference_mask), if given, is ANDed in on top of
    the per-line S/N cut -- it blanks spaxels outside the reference line's
    detected footprint even when this line's own S/N happens to clear sn_min
    there (e.g. edge-of-coverage artifacting).

    Stats/normalization are computed on the NaN-masked array (so zero-filled
    background pixels don't bias the color stretch); only the returned
    display array itself has NaNs replaced with 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        sn = flux_map / err_map
    masked = np.where(sn >= sn_min, flux_map, np.nan)
    if galaxy_mask is not None:
        masked = np.where(galaxy_mask, masked, np.nan)

    if np.all(np.isnan(masked)):
        return None, None

    _, med, std = sigma_clipped_stats(masked, sigma=3.0)
    norm = ImageNormalize(stretch=LinearStretch(), vmin=0, vmax=med + 5 * std)
    display = np.nan_to_num(masked, nan=0.0)
    return display, norm


def plot_maps(maps, outdir, target_name, sn_min=3.0, cmap="viridis", localization_csv=None,
              reference_line=DEFAULT_REFERENCE_LINE, mask_n_sigma=DEFAULT_MASK_SIGMA):
    """
    mask_n_sigma: passed to build_reference_mask to blank non-galaxy spaxels
    (see there); pass None to disable galaxy-footprint masking entirely.
    """
    os.makedirs(outdir, exist_ok=True)
    wcs2d = maps["_wcs"].celestial
    loc = load_localization(localization_csv, target_name)
    galaxy_mask = build_reference_mask(maps, reference_line, mask_n_sigma) if mask_n_sigma is not None else None

    for name in LINES:
        flux_map = maps[name]["flux"]
        err_map = maps[name]["flux_err"]
        if flux_map is None or np.all(np.isnan(flux_map)):
            continue

        display, norm = _display_and_norm(flux_map, err_map, sn_min, galaxy_mask=galaxy_mask)
        if display is None:
            print(f"[{name}] no spaxels above S/N >= {sn_min}, skipping PNG")
            continue

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": wcs2d})
        im = ax.imshow(display, origin="lower", norm=norm, cmap=cmap)
        plt.colorbar(im, ax=ax, label="Flux (counts)", pad=0.02, fraction=0.047)
        if loc is not None:
            add_localization_ellipse(ax, wcs2d, loc)
        ax.set_title(f"{target_name} -- {name}")
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.grid(color="lightgray", lw=0.5, ls="--")

        out_path = os.path.join(outdir, f"{target_name}_{name}_flux.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


# Panels for the combined summary figure: (line name, colormap)
PANEL_LINES = [
    ("OIII_5007", "Greens"),
    ("Halpha", "Reds"),
    ("NII_6584", "OrRd"),
    ("SII_6731", "Purples"),
]


def plot_combined_panel(maps, outdir, target_name, sn_min=3.0, localization_csv=None,
                         reference_line=DEFAULT_REFERENCE_LINE, mask_n_sigma=DEFAULT_MASK_SIGMA):
    """
    Single 1x4 figure with [OIII]5007, Halpha, [NII]6584, [SII]6731 side by side.

    mask_n_sigma: passed to build_reference_mask to blank non-galaxy spaxels
    (see there); pass None to disable galaxy-footprint masking entirely.
    """
    os.makedirs(outdir, exist_ok=True)
    wcs2d = maps["_wcs"].celestial
    loc = load_localization(localization_csv, target_name)
    galaxy_mask = build_reference_mask(maps, reference_line, mask_n_sigma) if mask_n_sigma is not None else None

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={"projection": wcs2d})
    for i, (ax, (name, cmap)) in enumerate(zip(axes, PANEL_LINES)):
        flux_map = maps[name]["flux"]
        err_map = maps[name]["flux_err"]
        if flux_map is None or np.all(np.isnan(flux_map)):
            print(f"[{name}] not available, leaving panel blank")
            ax.set_title(f"{name} (unavailable)")
            ax.set_xlabel("RA")
            if i != 0:
                ax.coords[1].set_ticklabel_visible(False)
            continue

        display, norm = _display_and_norm(flux_map, err_map, sn_min, galaxy_mask=galaxy_mask)
        if display is None:
            print(f"[{name}] no spaxels above S/N >= {sn_min}, panel will be all zero")
            display, norm = np.zeros_like(flux_map), None

        im = ax.imshow(display, origin="lower", norm=norm, cmap=cmap)
        plt.colorbar(im, ax=ax, label="Flux (counts)", pad=0.02, fraction=0.047)
        if loc is not None:
            add_localization_ellipse(ax, wcs2d, loc)
        ax.set_title(name)
        ax.set_xlabel("RA")
        if i == 0:
            ax.set_ylabel("Dec")
        else:
            ax.coords[1].set_ticklabel_visible(False)
        ax.grid(color="lightgray", lw=0.5, ls="--")

    fig.suptitle(target_name)
    out_path = os.path.join(outdir, f"{target_name}_panel_OIII_Ha_NII_SII.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metallicity_map(maps, outdir, target_name, localization_csv=None,
                          reference_line=DEFAULT_REFERENCE_LINE, mask_n_sigma=DEFAULT_MASK_SIGMA):
    """
    2D image of the Pettini & Pagel (2004) N2 metallicity, 12+log10(O/H)
    (see compute_metallicity_map). NaN spaxels are left as NaN (rendered
    transparent by imshow) rather than zero-filled like the flux maps -- 0 is
    not a meaningful metallicity value, so zero-filling would misleadingly
    paint "no measurement" spaxels as an extreme low-metallicity reading.

    mask_n_sigma: passed to build_reference_mask to blank non-galaxy spaxels
    (see there); pass None to disable galaxy-footprint masking entirely.
    """
    os.makedirs(outdir, exist_ok=True)
    metallicity = maps.get("_metallicity")
    if metallicity is None or np.all(np.isnan(metallicity)):
        print("[metallicity] no valid spaxels, skipping PNG")
        return None

    galaxy_mask = build_reference_mask(maps, reference_line, mask_n_sigma) if mask_n_sigma is not None else None
    if galaxy_mask is not None:
        metallicity = np.where(galaxy_mask, metallicity, np.nan)
        if np.all(np.isnan(metallicity)):
            print("[metallicity] no valid spaxels inside galaxy mask, skipping PNG")
            return None

    wcs2d = maps["_wcs"].celestial
    loc = load_localization(localization_csv, target_name)
    _, med, std = sigma_clipped_stats(metallicity, sigma=3.0)
    norm = ImageNormalize(stretch=LinearStretch(), vmin=med - 3 * std, vmax=med + 3 * std)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": wcs2d})
    im = ax.imshow(metallicity, origin="lower", norm=norm, cmap="viridis")
    plt.colorbar(im, ax=ax, label="12 + log10(O/H)", pad=0.02, fraction=0.047)
    if loc is not None:
        add_localization_ellipse(ax, wcs2d, loc)
    ax.set_title(f"{target_name} -- metallicity (Pettini & Pagel 2004 N2)")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.grid(color="lightgray", lw=0.5, ls="--")

    out_path = os.path.join(outdir, f"{target_name}_metallicity.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def estimate_ellipse_metallicity(maps, loc, n_sigma=2.0, sn_min=3.0):
    """
    Estimate the metallicity at the FRB location: the spaxels within its
    n_sigma localization ellipse (see localization_ellipse_mask).

    Normally this is the S/N-weighted average of the metallicity map's valid
    (non-NaN) spaxels inside the ellipse, weighted by
    min(NII_6584 S/N, Halpha S/N) per spaxel -- a ratio's reliability is set
    by whichever of its two lines is weaker there.

    If NO spaxel inside the ellipse clears the S/N cut used by
    compute_metallicity_map (e.g. the FRB sits in a region with no
    significant NII/Halpha detection at all -- confirmed for FRB20250227A),
    falls back to a 3-sigma LIMIT instead: the median NII_6584/Halpha flux
    uncertainty across the ellipse's spaxels, x3, plugged into the same N2
    formula -- the best available characterization of the metallicity in a
    region with no real line detection.

    Returns (value, label, is_limit) or None if the ellipse doesn't overlap
    the cube FOV at all, or no NII_6584/Halpha uncertainty is available
    there either.
    """
    wcs2d = maps["_wcs"].celestial
    metallicity = maps["_metallicity"]
    mask = localization_ellipse_mask(metallicity.shape, wcs2d, loc, n_sigma=n_sigma)
    if not np.any(mask):
        warnings.warn("FRB localization ellipse does not overlap the cube FOV; "
                       "skipping metallicity marker")
        return None

    nii_flux, nii_err = maps["NII_6584"]["flux"][mask], maps["NII_6584"]["flux_err"][mask]
    halpha_flux, halpha_err = maps["Halpha"]["flux"][mask], maps["Halpha"]["flux_err"][mask]
    z_in_ellipse = metallicity[mask]

    valid = np.isfinite(z_in_ellipse)
    if np.any(valid):
        with np.errstate(invalid="ignore", divide="ignore"):
            nii_sn = nii_flux / nii_err
            halpha_sn = halpha_flux / halpha_err
        weight = np.minimum(nii_sn, halpha_sn)[valid]
        z_value = float(np.average(z_in_ellipse[valid], weights=weight))
        return z_value, "S/N-weighted mean (FRB ellipse)", False

    nii_err_med = np.nanmedian(nii_err)
    halpha_err_med = np.nanmedian(halpha_err)
    if not (np.isfinite(nii_err_med) and np.isfinite(halpha_err_med) and halpha_err_med > 0):
        warnings.warn("no NII_6584/Halpha uncertainty available in the FRB ellipse either; "
                       "skipping metallicity marker")
        return None
    z_value = 8.90 + 0.57 * np.log10(3.0 * nii_err_med / (3.0 * halpha_err_med))
    return float(z_value), f"{sn_min:.0f}σ limit (FRB ellipse)", True


def plot_metallicity_histogram(maps, outdir, target_name, localization_csv=None,
                                reference_line=DEFAULT_REFERENCE_LINE, mask_n_sigma=DEFAULT_MASK_SIGMA):
    """
    1D histogram of per-spaxel metallicity values (finite spaxels only), plus
    a red dashed vertical line marking the metallicity estimate at the FRB
    location (see estimate_ellipse_metallicity) when the target is found in
    localization_csv.

    mask_n_sigma: passed to build_reference_mask to exclude non-galaxy
    spaxels from the histogram (see there); pass None to disable
    galaxy-footprint masking entirely. Does not affect the FRB-ellipse marker
    line, which is computed over the ellipse regardless of this mask.
    """
    os.makedirs(outdir, exist_ok=True)
    metallicity = maps.get("_metallicity")
    galaxy_mask = build_reference_mask(maps, reference_line, mask_n_sigma) if mask_n_sigma is not None else None
    if metallicity is not None and galaxy_mask is not None:
        metallicity = np.where(galaxy_mask, metallicity, np.nan)
    values = metallicity[np.isfinite(metallicity)] if metallicity is not None else np.array([])
    if values.size == 0:
        print("[metallicity] no valid spaxels, skipping histogram")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("12 + log10(O/H)")
    ax.set_ylabel("Number of spaxels")
    ax.set_title(f"{target_name} -- per-spaxel metallicity distribution (N={values.size})")

    loc = load_localization(localization_csv, target_name)
    if loc is not None:
        result = estimate_ellipse_metallicity(maps, loc)
        if result is not None:
            z_value, label, _is_limit = result
            ax.axvline(z_value, color="red", ls="--", lw=2, label=label)
            ax.legend()

    out_path = os.path.join(outdir, f"{target_name}_metallicity_hist.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def load_maps_from_fits(fits_path):
    """
    Reload flux/flux_err maps written by save_fits, without re-running the fit.
    NaNs are preserved exactly as stored (only plotting zero-fills them).
    """
    maps = {name: {"flux": None, "flux_err": None} for name in LINES}
    header2d = None
    with fits.open(fits_path) as hdul:
        for name in LINES:
            flux_ext, err_ext = f"{name}_FLUX", f"{name}_ERR"
            if flux_ext in hdul:
                maps[name]["flux"] = hdul[flux_ext].data.astype(float)
                maps[name]["flux_err"] = hdul[err_ext].data.astype(float)
                if header2d is None:
                    header2d = hdul[flux_ext].header
        if "METALLICITY" in hdul:
            maps["_metallicity"] = hdul["METALLICITY"].data.astype(float)
    maps["_wcs"] = WCS(header2d)
    return maps


