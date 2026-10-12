"""
PSF size of a 2D map (emission-line map or collapsed cube) measured from the
compact sources in it: stars, QSOs, or compact star-forming clumps.

Each source is fit with an elliptical, pixel-integrated Gaussian
(photutils GaussianPRF) on top of a tilted plane, which absorbs the smooth
diffuse emission a clump sits on. A resolved clump's fitted FWHM is its
intrinsic size convolved with the PSF, so it is only an upper limit on the
PSF; summarize_psf() turns a set of clump fits into that upper limit and a
best estimate. The formal fit errors are too small on maps whose pixels are
correlated (e.g. by a cube's resampling kernel), so per-source uncertainties
come from inject_recover() on the real map instead.
"""
import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Planar2D
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.utils.exceptions import AstropyUserWarning
from photutils.detection import find_peaks
from photutils.profiles import RadialProfile
from photutils.psf import CircularGaussianPRF, GaussianPRF
from scipy import ndimage

FWHM_BOUNDS_PIX = (1.0, 20.0)
BOUND_TOL = 1e-3
N_FIT_PARAMS = 9  # flux, x_0, y_0, x_fwhm, y_fwhm, theta, slope_x, slope_y, intercept


def _cutout_slices(shape, x, y, box_pix):
    half = int(box_pix) // 2
    xc, yc = int(round(x)), int(round(y))
    return (slice(max(yc - half, 0), min(yc + half + 1, shape[0])),
            slice(max(xc - half, 0), min(xc + half + 1, shape[1])))


def fit_gaussian_psf(flux, err, x, y, box_pix, fwhm_guess_pix, mask=None, max_shift_pix=2.0,
                     support_sn=3.0):
    """
    Fit one source at image position (x, y) inside a box_pix-wide cutout.
    FWHMs are in pixels; fwhm is the geometric mean sqrt(x_fwhm*y_fwhm) and q
    the minor/major axis ratio. support_pix counts the connected pixels at
    S/N >= support_sn (above the fitted plane) containing the fitted centre.
    """
    sy, sx = _cutout_slices(flux.shape, x, y, box_pix)
    data = np.asarray(flux[sy, sx], dtype=float)
    sig = np.asarray(err[sy, sx], dtype=float)
    yy, xx = np.mgrid[sy, sx]
    good = np.isfinite(data) & np.isfinite(sig) & (sig > 0)
    if mask is not None:
        good &= ~np.asarray(mask[sy, sx], dtype=bool)
    out = dict(x_init=float(x), y_init=float(y), success=False, at_bound=True, n_good=int(good.sum()),
               origin=(sy.start, sx.start))
    if good.sum() <= N_FIT_PARAMS + 5:
        out["message"] = "too few good pixels"
        return out

    edge = np.zeros_like(good)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    bg0 = float(np.median(data[good & edge])) if np.any(good & edge) else float(np.median(data[good]))
    iy, ix = int(round(y)) - sy.start, int(round(x)) - sx.start
    peak0 = data[iy, ix] - bg0 if good[iy, ix] else np.max(data[good]) - bg0
    sigma0 = fwhm_guess_pix * gaussian_fwhm_to_sigma
    flux0 = max(float(peak0) * 2 * np.pi * sigma0 ** 2, 1e-6)

    g = GaussianPRF(flux=flux0, x_0=x, y_0=y, x_fwhm=fwhm_guess_pix, y_fwhm=fwhm_guess_pix, theta=0.0)
    for name in ("x_fwhm", "y_fwhm", "theta"):
        getattr(g, name).fixed = False  # photutils 3 fixes these by default
    g.x_fwhm.bounds = FWHM_BOUNDS_PIX
    g.y_fwhm.bounds = FWHM_BOUNDS_PIX
    g.theta.bounds = (-90.0, 90.0)
    g.x_0.bounds = (x - max_shift_pix, x + max_shift_pix)
    g.y_0.bounds = (y - max_shift_pix, y + max_shift_pix)
    g.flux.bounds = (0.0, None)
    model = g + Planar2D(slope_x=0.0, slope_y=0.0, intercept=bg0)

    fitter = TRFLSQFitter(calc_uncertainties=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        fit = fitter(model, xx[good], yy[good], data[good], weights=1.0 / sig[good], maxiter=2000)

    p = dict(zip(fit.param_names, fit.parameters))
    xf, yf = p["x_fwhm_0"], p["y_fwhm_0"]
    x0, y0 = p["x_0_0"], p["y_0_0"]
    fwhm = float(np.sqrt(xf * yf))

    try:
        stds = dict(zip(fit.param_names, fit.stds.stds))
        fwhm_err_formal = 0.5 * fwhm * np.hypot(stds["x_fwhm_0"] / xf, stds["y_fwhm_0"] / yf)
    except (AttributeError, TypeError, KeyError):
        fwhm_err_formal = np.nan

    model_img = fit(xx, yy)
    plane_img = fit[1](xx, yy)
    resid = data - model_img
    chi2 = float(np.sum((resid[good] / sig[good]) ** 2))

    jc = min(max(int(round(x0)) - sx.start, 0), data.shape[1] - 1)
    ic = min(max(int(round(y0)) - sy.start, 0), data.shape[0] - 1)
    err_c = sig[ic, jc] if good[ic, jc] else float(np.median(sig[good]))
    amp = p["flux_0"] / (2 * np.pi * (xf * gaussian_fwhm_to_sigma) * (yf * gaussian_fwhm_to_sigma))

    labels, _ = ndimage.label(good & ((data - plane_img) / np.where(good, sig, 1.0) >= support_sn))
    support_pix = int(np.sum(labels == labels[ic, jc])) if labels[ic, jc] > 0 else 0

    at_bound = bool(
        min(xf, yf) <= FWHM_BOUNDS_PIX[0] + BOUND_TOL or max(xf, yf) >= FWHM_BOUNDS_PIX[1] - BOUND_TOL
        or abs(x0 - x) >= max_shift_pix - BOUND_TOL or abs(y0 - y) >= max_shift_pix - BOUND_TOL)

    out.update(
        success=bool(fitter.fit_info.get("success", False)) and np.all(np.isfinite(fit.parameters)),
        message=str(fitter.fit_info.get("message", "")), at_bound=at_bound,
        x=float(x0), y=float(y0), x_fwhm=float(xf), y_fwhm=float(yf), theta=float(p["theta_0"]),
        fwhm=fwhm, fwhm_err_formal=float(fwhm_err_formal), q=float(min(xf, yf) / max(xf, yf)),
        flux=float(p["flux_0"]), amp=float(amp), peak_sn=float(amp / err_c),
        chi2_red=chi2 / max(int(good.sum()) - N_FIT_PARAMS, 1), support_pix=support_pix,
        data=np.where(good, data, np.nan), model=model_img, plane=plane_img,
        resid=np.where(good, resid, np.nan), err=np.where(good, sig, np.nan),
    )
    return out


def radial_profile_fwhm(fit_result, max_radius_pix=None):
    """Gaussian and Moffat FWHM (pixels) of the plane-subtracted cutout's azimuthal profile."""
    img = fit_result["data"] - fit_result["plane"]
    img = np.where(np.isfinite(img), img, 0.0)
    xycen = (fit_result["x"] - fit_result["origin"][1], fit_result["y"] - fit_result["origin"][0])
    rmax = max_radius_pix or (min(img.shape) // 2)
    profile = RadialProfile(img, xycen, np.arange(0, rmax + 1, 1.0))
    return dict(gaussian=float(profile.gaussian_fwhm), moffat=float(profile.moffat_fwhm))


def find_clump_candidates(flux, err, fwhm_guess_pix, sn_min=10.0, border_pix=None):
    """
    Local maxima (within a ~fwhm_guess box) whose own pixel S/N >= sn_min;
    returns an (N, 2) array of (x, y). Shape is vetted later by the fit --
    DAOStarFinder's roundness pre-filter was found to discard most bright
    knots sitting on diffuse emission.
    """
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    box = int(round(fwhm_guess_pix)) | 1
    tab = find_peaks(np.where(good, flux, 0.0), threshold=np.where(good, sn_min * err, np.inf),
                     box_size=box, border_width=box if border_pix is None else border_pix, mask=~good)
    if tab is None:
        return np.empty((0, 2))
    return np.column_stack([np.asarray(tab["x_peak"], float), np.asarray(tab["y_peak"], float)])


def select_psf_clumps(fits, q_min=0.8, sn_min=15.0, min_support_pix=9, isolation_fwhm=1.0):
    """
    Flag each fit dict in place ('accepted', 'reject_reasons') and return the
    accepted ones. No cut is made on FWHM itself.
    """
    xy = np.array([[f["x_init"], f["y_init"]] for f in fits]) if fits else np.empty((0, 2))
    for i, f in enumerate(fits):
        reasons = []
        if not f["success"]:
            reasons.append("fit failed")
        else:
            if f["at_bound"]:
                reasons.append("parameter at bound")
            if f["q"] < q_min:
                reasons.append(f"q={f['q']:.2f}<{q_min}")
            if f["peak_sn"] < sn_min:
                reasons.append(f"peak S/N={f['peak_sn']:.1f}<{sn_min}")
            if f["support_pix"] < min_support_pix:
                reasons.append(f"support={f['support_pix']}<{min_support_pix} px")
            d = np.hypot(xy[:, 0] - f["x"], xy[:, 1] - f["y"])
            d[i] = np.inf
            if np.any(d < isolation_fwhm * f["fwhm"]):
                reasons.append("neighbour within FWHM")
        f["reject_reasons"] = reasons
        f["accepted"] = not reasons
    return [f for f in fits if f["accepted"]]


def inject_recover(flux, err, fwhm_pix, peak_sn, box_pix, fwhm_guess_pix, footprint, avoid_xy,
                   avoid_radius_pix, n_trials=40, rng=None):
    """
    Add one round Gaussian source (fwhm_pix, peak = peak_sn x local ERR) at a
    time to the real map, at random footprint positions at least
    avoid_radius_pix from every avoid_xy position, and re-fit it with
    fit_gaussian_psf. Returns the fractional FWHM bias (median) and scatter
    (1.4826 x MAD) of recovered/true.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = flux.shape
    yy, xx = np.indices(flux.shape)
    ok = footprint & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    half = int(box_pix) // 2 + 2
    ok[:half, :] = ok[-half:, :] = False
    ok[:, :half] = ok[:, -half:] = False
    for ax, ay in np.atleast_2d(avoid_xy):
        ok &= np.hypot(xx - ax, yy - ay) >= avoid_radius_pix
    candidates = np.flatnonzero(ok)
    result = dict(fwhm_true=float(fwhm_pix), peak_sn=float(peak_sn), n_trials=0, n_ok=0,
                  bias=np.nan, scatter=np.nan, recovered=np.array([]))
    if candidates.size == 0:
        return result

    picks = rng.choice(candidates, size=min(n_trials, candidates.size), replace=False)
    sigma = fwhm_pix * gaussian_fwhm_to_sigma
    recovered = np.full(picks.size, np.nan)
    for k, idx in enumerate(picks):
        yc, xc = divmod(int(idx), nx)
        x0, y0 = xc + rng.uniform(-0.5, 0.5), yc + rng.uniform(-0.5, 0.5)
        src = CircularGaussianPRF(flux=peak_sn * err[yc, xc] * 2 * np.pi * sigma ** 2,
                                  x_0=x0, y_0=y0, fwhm=fwhm_pix)
        r = fit_gaussian_psf(flux + src(xx, yy), err, x0 + rng.normal(0, 0.3), y0 + rng.normal(0, 0.3),
                             box_pix, fwhm_guess_pix)
        if r["success"] and not r["at_bound"]:
            recovered[k] = r["fwhm"]

    ratio = recovered[np.isfinite(recovered)] / fwhm_pix
    result.update(n_trials=int(picks.size), n_ok=int(ratio.size), recovered=recovered)
    if ratio.size >= 5:
        result.update(bias=float(np.median(ratio) - 1.0),
                      scatter=float(1.4826 * np.median(np.abs(ratio - np.median(ratio)))))
    return result


def summarize_psf(accepted, fwhm_key="fwhm", err_key="fwhm_err"):
    """
    Most compact accepted source (strict upper limit on the PSF) and the
    inverse-variance-weighted mean FWHM of the sources that agree with it
    within 2 sigma (best estimate).
    """
    empty = dict(n_accepted=0, i_min=-1, fwhm_min=np.nan, fwhm_min_err=np.nan,
                 fwhm_best=np.nan, fwhm_best_err=np.nan, n_consistent=0)
    if not accepted:
        return empty
    f = np.array([a[fwhm_key] for a in accepted], dtype=float)
    e = np.array([a[err_key] for a in accepted], dtype=float)
    usable = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if not np.any(usable):
        return dict(empty, n_accepted=len(accepted))
    i = int(np.flatnonzero(usable)[np.argmin(f[usable])])
    consistent = usable & (np.abs(f - f[i]) <= 2 * np.sqrt(e ** 2 + e[i] ** 2))
    w = 1.0 / e[consistent] ** 2
    return dict(n_accepted=len(accepted), i_min=i, fwhm_min=float(f[i]), fwhm_min_err=float(e[i]),
                fwhm_best=float(np.sum(w * f[consistent]) / np.sum(w)),
                fwhm_best_err=float(1.0 / np.sqrt(np.sum(w))), n_consistent=int(consistent.sum()))
