"""
Radial profile and gradient of a per-spaxel quantity (log O3N2, 12 + log(O/H), ...) in a disk galaxy, and how a point in
the disk (an FRB site) compares with what the gradient predicts at its radius.

Geometry: thin disk, cos i = q (the observed axis ratio), PA East of North. Everything is computed from sky offsets, so
flips and rotations in the WCS need no special handling. disk_radius does this for sky positions, deprojected_radius for
a whole pixel grid. isophote_geometry, kinematic_position_angle and kinemetry_rings measure q and PA.

Gradient: a straight line through the medians of radial bins (radial_profile, fit_gradient). Neighbouring spaxels are
correlated by the PSF, so a bootstrap over spaxels would understate the errors: block_bootstrap_gradient resamples
PSF-sized spatial blocks and redoes the medians and the fit.

No per-target values live here; see test_redux/radial_gradients_hosts.py.
"""
import numpy as np
from astropy import units as u
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

C_KMS = 2.99792458e5


def disk_radius(center, coord, pa_deg, axis_ratio):
    """
    Deprojected radius (arcsec) and disk-plane azimuth (rad, from the major axis at pa_deg) of the sky
    positions `coord` around `center`, for a thin disk of observed axis ratio `axis_ratio` (1 = no deprojection).
    """
    d_east, d_north = center.spherical_offsets_to(coord)
    d_east, d_north = d_east.to_value(u.arcsec), d_north.to_value(u.arcsec)
    pa = np.radians(pa_deg)
    along_major = d_east * np.sin(pa) + d_north * np.cos(pa)
    along_minor = (d_east * np.cos(pa) - d_north * np.sin(pa)) / axis_ratio
    return np.hypot(along_major, along_minor), np.arctan2(along_minor, along_major)


def deprojected_radius(wcs2d, center, pa_deg, axis_ratio, shape):
    """disk_radius over every spaxel of an image of `shape` with celestial WCS `wcs2d`."""
    yy, xx = np.indices(shape)
    return disk_radius(center, wcs2d.pixel_to_world(xx, yy), pa_deg, axis_ratio)


def continuum_peak(img, sigma_pix=3.0, region=None):
    """
    (x, y) pixel of the peak of `img` after Gaussian smoothing, NaNs filled with 0. `region`: boolean mask limiting the
    search (e.g. to exclude a foreground star). Unlike a flux-weighted centroid, this does not move when a bright knot,
    a neighbour or the field edge makes the light asymmetric about the nucleus.
    """
    smooth = gaussian_filter(np.where(np.isfinite(img), img, 0.0), sigma_pix)
    if region is not None:
        smooth = np.where(region, smooth, -np.inf)
    y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
    return int(x), int(y)


def _second_moment_shape(data, x0, y0, radius_pix):
    """(eps, PA in rad counter-clockwise from +x) of the positive flux within radius_pix of (x0, y0), from its moments."""
    yy, xx = np.indices(data.shape)
    dx, dy = xx - x0, yy - y0
    w = np.where((np.hypot(dx, dy) <= radius_pix) & (data > 0), data, 0.0)
    mxx, myy, mxy = np.sum(w * dx * dx), np.sum(w * dy * dy), np.sum(w * dx * dy)
    root = np.hypot(0.5 * (mxx - myy), mxy)
    major, minor = 0.5 * (mxx + myy) + root, 0.5 * (mxx + myy) - root
    return 1.0 - np.sqrt(minor / major), 0.5 * np.arctan2(2.0 * mxy, mxx - myy)


def isophote_geometry(img, wcs2d, x0, y0, sma_min, sma_max, sma0=None, step=0.1, maxgerr=1.0):
    """
    photutils ellipse-isophote fit around a fixed center (x0, y0) of an image, background-subtracted here. Semi-major
    axes in arcsec. Returns dict of arrays: sma (arcsec), eps, pa_deg (East of North, modulo 180) and snr (mean isophote
    intensity over the image's sigma-clipped background rms). Empty arrays if no isophote converges.

    The fit starts from the image's second moments. Started about 40 deg from the true PA, photutils stalls at its
    starting eps (stop code 2), and such isophotes look like measurements; only converged ones (stop code 0) are kept.
    """
    from photutils.isophote import Ellipse, EllipseGeometry

    pixscale = np.sqrt(abs(np.linalg.det(wcs2d.pixel_scale_matrix))) * 3600.0
    _mean, med, std = sigma_clipped_stats(img, sigma=3.0)
    data = np.where(np.isfinite(img), img - med, 0.0)
    sma_start = (sma0 if sma0 is not None else np.sqrt(sma_min * sma_max)) / pixscale
    eps, pa = _second_moment_shape(data, x0, y0, sma_start)
    geometry = EllipseGeometry(x0=float(x0), y0=float(y0), sma=sma_start, eps=float(np.clip(eps, 0.05, 0.8)), pa=float(pa))
    # An isophote's geometry can go to NaN in faint outskirts, which crashes fit_image: shrink the outer limit and retry.
    sma_limit = sma_max / pixscale
    while True:
        try:
            isolist = Ellipse(data, geometry=geometry).fit_image(sma0=sma_start, minsma=sma_min / pixscale, maxsma=sma_limit,
                                                                 step=step, fix_center=True, maxgerr=maxgerr)
            break
        except ValueError:
            sma_limit *= 0.9
            if sma_limit <= sma_start:
                return dict(sma=np.array([]), eps=np.array([]), pa_deg=np.array([]), snr=np.array([]))
    empty = dict(sma=np.array([]), eps=np.array([]), pa_deg=np.array([]), snr=np.array([]))
    if len(isolist) == 0:
        return empty
    converged = np.asarray(isolist.stop_code) == 0
    if not converged.any():
        return empty
    pa_pix = np.asarray(isolist.pa)[converged]
    # photutils' PA is counter-clockwise from +x in pixels; two points along it on the sky give the PA in sky terms.
    start = wcs2d.pixel_to_world(x0, y0)
    end = wcs2d.pixel_to_world(x0 + 5.0 * np.cos(pa_pix), y0 + 5.0 * np.sin(pa_pix))
    return dict(sma=np.asarray(isolist.sma)[converged] * pixscale, eps=np.asarray(isolist.eps)[converged],
                pa_deg=np.asarray(start.position_angle(end).deg) % 180.0,
                snr=np.asarray(isolist.intens)[converged] / std)


def axial_mean_std(pa_deg):
    """Circular mean and standard deviation (deg) of position angles known only modulo 180 deg."""
    doubled = np.radians(2.0 * np.asarray(pa_deg, dtype=float))
    c, s = np.mean(np.cos(doubled)), np.mean(np.sin(doubled))
    resultant = np.clip(np.hypot(c, s), 1e-12, 1.0)
    return float(np.degrees(np.arctan2(s, c) / 2.0) % 180.0), float(np.degrees(np.sqrt(-2.0 * np.log(resultant)) / 2.0))


def kinematic_position_angle(wcs2d, center, z_map, mask, n_iter=3, clip_sigma=3.0):
    """
    PA (deg, East of North, of the receding side) of a plane fit to the velocity field: v = v0 + a dE + b dN over the
    spaxels in `mask`, sigma-clipped. A solid-body approximation: it needs roughly symmetric coverage around the center
    (restrict `mask` to a circle inside the field if the field cuts the galaxy). Returns (pa_deg, gradient in km/s per
    arcsec, residual rms in km/s, spaxels used).
    """
    yy, xx = np.nonzero(mask & np.isfinite(z_map))
    d_east, d_north = center.spherical_offsets_to(wcs2d.pixel_to_world(xx, yy))
    z = z_map[yy, xx]
    z0 = np.median(z)
    v = C_KMS * (z - z0) / (1.0 + z0)
    design = np.column_stack([np.ones_like(v), d_east.to_value(u.arcsec), d_north.to_value(u.arcsec)])
    keep = np.ones_like(v, dtype=bool)
    for _ in range(n_iter):
        coeff, *_ = np.linalg.lstsq(design[keep], v[keep], rcond=None)
        resid = v - design @ coeff
        keep = np.abs(resid) < clip_sigma * np.std(resid[keep])
    return (float(np.degrees(np.arctan2(coeff[1], coeff[2])) % 360.0), float(np.hypot(coeff[1], coeff[2])),
            float(np.std(resid[keep])), int(keep.sum()))


def _ellipse_harmonics(fill, ok, x0, y0, sma_pix, pa_deg, q, sky_to_pix, n_samples, min_coverage):
    """Harmonic coefficients (A0, A1, B1, A2, B2, A3, B3) along one ellipse, or None if its coverage is below min_coverage."""
    psi = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    pa = np.radians(pa_deg)
    along_major, along_minor = sma_pix * np.cos(psi), q * sma_pix * np.sin(psi)
    d_east = along_major * np.sin(pa) + along_minor * np.cos(pa)
    d_north = along_major * np.cos(pa) - along_minor * np.sin(pa)
    dx, dy = sky_to_pix @ np.vstack([d_east, d_north])
    coords = np.vstack([y0 + dy, x0 + dx])
    good = map_coordinates(ok, coords, order=1, mode="constant", cval=0.0) > 0.999
    if good.mean() < min_coverage:
        return None
    values = map_coordinates(fill, coords[:, good], order=1)
    p = psi[good]
    design = np.column_stack([np.ones_like(p), np.sin(p), np.cos(p), np.sin(2 * p), np.cos(2 * p), np.sin(3 * p),
                              np.cos(3 * p)])
    return np.linalg.lstsq(design, values, rcond=None)[0]


def kinemetry_rings(velocity, wcs2d, x0, y0, sma_arcsec, pa_step=3.0, q_min=0.2, q_step=0.02, n_samples=64,
                    min_coverage=0.75):
    """
    Kinemetric PA and flattening of a velocity map (km/s, NaN where unmeasured) on rings of semi-major axis sma_arcsec
    around a fixed center, after Krajnovic et al. (2006): sample uniformly in eccentric anomaly psi (x = a cos psi,
    y = q a sin psi) with bilinear interpolation, expand to third harmonics (their eq. 7) and minimize
    chi2 = A1^2 + A2^2 + B2^2 + A3^2 + B3^2 (their eq. 8) on a (PA, q) grid, then a finer grid around the minimum.
    Unlike their second stage, the center is not fitted. For a thin disk q = cos i.

    Returns dict of arrays, one entry per ring, NaN where the circle of that radius has coverage below min_coverage:
    sma (arcsec), pa_deg (0-360, of the receding side), q, k1 (km/s, rotation amplitude) and resid (rms of the minimized
    terms, km/s).
    """
    ok = np.isfinite(velocity).astype(float)
    fill = np.where(np.isfinite(velocity), velocity, 0.0)
    # The WCS's orientation alone, unit scale: takes (East, North) offsets in pixel units to (x, y), keeping flips/rotation.
    pixscale = np.sqrt(abs(np.linalg.det(wcs2d.pixel_scale_matrix))) * 3600.0
    cd = wcs2d.pixel_scale_matrix * 3600.0 / pixscale
    sky_to_pix = np.linalg.inv(cd)

    def chi2(sma_pix, pa_deg, q):
        coeff = _ellipse_harmonics(fill, ok, x0, y0, sma_pix, pa_deg, q, sky_to_pix, n_samples, min_coverage)
        if coeff is None:
            return np.inf, None
        return float(np.sum(coeff[[1, 3, 4, 5, 6]] ** 2)), coeff

    def best_on(sma_pix, pa_grid, q_grid):
        best = (np.inf, np.nan, np.nan, None)
        for pa_deg in pa_grid:
            for q in q_grid:
                value, coeff = chi2(sma_pix, pa_deg, q)
                if value < best[0]:
                    best = (value, pa_deg, q, coeff)
        return best

    n_rings = len(sma_arcsec)
    out = dict(sma=np.asarray(sma_arcsec, dtype=float), pa_deg=np.full(n_rings, np.nan), q=np.full(n_rings, np.nan),
               k1=np.full(n_rings, np.nan), resid=np.full(n_rings, np.nan))
    for i, sma in enumerate(sma_arcsec):
        sma_pix = sma / pixscale
        # Coverage is judged on the circle, not on each candidate: a thin ellipse fits inside a truncated or gappy field
        # where a round one does not, so a per-candidate cut drives the fit to low q wherever coverage runs out.
        if _ellipse_harmonics(fill, ok, x0, y0, sma_pix, 0.0, 1.0, sky_to_pix, n_samples, min_coverage) is None:
            continue
        value, pa_deg, q, coeff = best_on(sma_pix, np.arange(0.0, 180.0, pa_step), np.arange(q_min, 1.0 + 1e-6, q_step))
        if coeff is None:
            continue
        value, pa_deg, q, coeff = best_on(sma_pix, np.arange(pa_deg - pa_step, pa_deg + pa_step + 1e-6, pa_step / 6.0),
                                          np.clip(np.arange(q - q_step, q + q_step + 1e-6, q_step / 4.0), q_min, 1.0))
        out["pa_deg"][i] = pa_deg % 360.0 if coeff[2] > 0 else (pa_deg + 180.0) % 360.0
        out["q"][i] = q
        out["k1"][i] = np.hypot(coeff[1], coeff[2])
        out["resid"][i] = np.sqrt(value / 5.0)
    return out


def radial_profile(values, radius, valid, footprint, bin_edges, field=None, pixel_area=None, axis_ratio=1.0):
    """
    Per-bin statistics of `values` over spaxels in `valid`, binned by `radius`. `footprint` marks the parent population
    (e.g. Halpha-detected spaxels) that valid_frac is measured against: where valid_frac is low, the median describes only
    the spaxels bright enough to pass the cuts. `field` (with `pixel_area` in radius-units^2 and `axis_ratio`) gives
    coverage: the fraction of the bin's deprojected annulus on the detector. Returns a dict of arrays, one entry per bin.
    """
    n_bins = len(bin_edges) - 1
    out = {key: np.full(n_bins, np.nan) for key in ("r_median", "median", "p16", "p84", "coverage")}
    out["n_valid"] = np.zeros(n_bins, dtype=int)
    out["n_footprint"] = np.zeros(n_bins, dtype=int)
    out["r_lo"], out["r_hi"] = np.asarray(bin_edges[:-1], dtype=float), np.asarray(bin_edges[1:], dtype=float)
    for i in range(n_bins):
        in_bin = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        selected = in_bin & valid
        out["n_valid"][i] = selected.sum()
        out["n_footprint"][i] = (in_bin & footprint).sum()
        if selected.any():
            out["r_median"][i] = np.median(radius[selected])
            out["p16"][i], out["median"][i], out["p84"][i] = np.percentile(values[selected], [16, 50, 84])
        if field is not None:
            ideal = np.pi * (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * axis_ratio / pixel_area
            out["coverage"][i] = (in_bin & field).sum() / ideal
    with np.errstate(invalid="ignore", divide="ignore"):
        out["valid_frac"] = out["n_valid"] / out["n_footprint"]
    return out


def fit_bins(profile, r_min, min_valid_frac, min_valid_spaxels):
    """
    Bins a gradient fit may use: from the first passing bin at or beyond r_min, outward to the last contiguous bin with
    valid_frac >= min_valid_frac and n_valid >= min_valid_spaxels. Boolean array, one entry per bin.
    """
    passes = (profile["valid_frac"] >= min_valid_frac) & (profile["n_valid"] >= min_valid_spaxels)
    use = np.zeros_like(passes)
    started = False
    for i in np.nonzero(profile["r_lo"] >= r_min)[0]:
        if passes[i]:
            use[i] = started = True
        elif started:
            break
    return use


def fit_gradient(r, y):
    """Unweighted straight line y = zero_point + slope r through the bin medians. Returns (slope, zero_point)."""
    good = np.isfinite(r) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan, np.nan
    slope, zero_point = np.polyfit(np.asarray(r)[good], np.asarray(y)[good], 1)
    return float(slope), float(zero_point)


def block_labels(shape, block_pix):
    """Integer label per spaxel of the square block (side block_pix) containing it."""
    yy, xx = np.indices(shape)
    side = max(int(np.ceil(block_pix)), 1)
    return (yy // side) * (shape[1] // side + 1) + (xx // side)


def block_bootstrap_gradient(values, radius, valid, bin_edges, use_bins, labels, n_boot=1000, seed=0):
    """
    Bootstrap the gradient by resampling spatial blocks (block_labels) of valid spaxels with replacement, then redoing
    the bin medians and fit_gradient over the bins in `use_bins`. A bin empty in a draw simply drops out of that draw.
    Returns (slopes, zero_points), each of length n_boot.
    """
    rng = np.random.default_rng(seed)
    edges_lo, edges_hi = np.asarray(bin_edges[:-1])[use_bins], np.asarray(bin_edges[1:])[use_bins]
    in_range = valid & (radius >= edges_lo.min()) & (radius < edges_hi.max())
    block_ids = np.unique(labels[in_range])
    blocks = [(values[in_range & (labels == b)], radius[in_range & (labels == b)]) for b in block_ids]
    slopes, zero_points = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        picks = rng.integers(0, len(blocks), len(blocks))
        v = np.concatenate([blocks[p][0] for p in picks])
        r = np.concatenate([blocks[p][1] for p in picks])
        r_bin, y_bin = [], []
        for lo, hi in zip(edges_lo, edges_hi):
            in_bin = (r >= lo) & (r < hi)
            if in_bin.any():
                r_bin.append(np.median(r[in_bin]))
                y_bin.append(np.median(v[in_bin]))
        slopes[k], zero_points[k] = fit_gradient(r_bin, y_bin)
    return slopes, zero_points


def expected_at(slopes, zero_points, r):
    """Median and half the 16-84 interval of the gradient's prediction at radius r, over bootstrap draws."""
    p16, p50, p84 = np.nanpercentile(zero_points + slopes * r, [16, 50, 84])
    return float(p50), float(0.5 * (p84 - p16))


def site_residual(site, site_err, expected, expected_err, azimuthal_scatter):
    """Site minus expected, its total uncertainty (the three terms in quadrature) and residual / uncertainty."""
    total = float(np.sqrt(site_err ** 2 + expected_err ** 2 + azimuthal_scatter ** 2))
    return float(site - expected), total, float((site - expected) / total)


def annulus_statistics(site, values, radius, valid, r_site, half_width, exclude=None):
    """
    Model-free comparison of `site` with valid spaxels within half_width of r_site, outside `exclude`: their number, mean
    and standard deviation, half their 16-84 interval (the azimuthal scatter at r_site), the fraction below `site`, and
    (site - mean) / std, the statistic of Galbany et al. (2016, Fig. 11). NaNs if the annulus is empty.
    """
    selected = valid & (np.abs(radius - r_site) <= half_width)
    if exclude is not None:
        selected &= ~exclude
    if not selected.any():
        return dict(n=0, mean=np.nan, std=np.nan, half_p16_p84=np.nan, fraction_below=np.nan, z_score=np.nan)
    v = values[selected]
    p16, p84 = np.percentile(v, [16, 84])
    mean, std = float(np.mean(v)), float(np.std(v))
    return dict(n=int(v.size), mean=mean, std=std, half_p16_p84=float(0.5 * (p84 - p16)),
                fraction_below=float(np.mean(v < site)), z_score=float((site - mean) / std) if std > 0 else np.nan)


def above_bpt_line(log_nii_halpha, log_oiii_hbeta, line="kewley01"):
    """
    True above a BPT demarcation: Kewley et al. (2001, "kewley01", maximum starburst) or Kauffmann et al. (2003,
    "kauffmann03"), including everything right of the curve's asymptote. False where either ratio is NaN.
    """
    asymptote, offset = {"kewley01": (0.47, 1.19), "kauffmann03": (0.05, 1.3)}[line]
    n2, o3 = np.asarray(log_nii_halpha, dtype=float), np.asarray(log_oiii_hbeta, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        above = (n2 >= asymptote) | (o3 > 0.61 / (n2 - asymptote) + offset)
    return above & np.isfinite(n2) & np.isfinite(o3)


def beam_smearing_flattening(halpha_img, radius, slope, fwhm_pix, bin_edges, use_bins):
    """
    Fraction of an injected gradient that survives the PSF. A log line ratio `slope` x radius is imposed on the
    Halpha-weighted numerator; numerator and denominator (Halpha) are convolved with a Gaussian of fwhm_pix; the ratio's
    slope is measured as for the data. Light-weighted smearing is why this needs the flux map. The observed Halpha map
    is already smeared by the PSF, so this is a lower limit on the flattening.
    """
    footprint = np.isfinite(halpha_img) & (halpha_img > 0)
    flux = np.where(footprint, halpha_img, 0.0)
    sigma = fwhm_pix * gaussian_fwhm_to_sigma
    half = int(np.ceil(4.0 * sigma))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    numerator = fftconvolve(flux * 10.0 ** (slope * radius), kernel, mode="same")
    denominator = fftconvolve(flux, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smeared = np.log10(numerator / denominator)
    profile = radial_profile(smeared, radius, footprint & np.isfinite(smeared), footprint, bin_edges)
    measured, _ = fit_gradient(profile["r_median"][use_bins], profile["median"][use_bins])
    return measured / slope
