"""
PSF-kernel-weighted mean of a surface-brightness map, with an empirical error from
the same kernel placed on blank sky.

Per-spaxel fit maps are noisy and correlated between neighbours (cube resampling
kernel, fibre footprint), and a fit whose line amplitude is bounded at zero never
returns a negative flux, so blank sky reads as a positive floor instead of zero.
A formal error propagated as if the spaxels were independent underestimates the
scatter several-fold, and an S/N cut applied before averaging biases the mean high.
`kernel_weighted_mean` therefore averages the raw, unthresholded map, and
`sky_kernel_means` gives the distribution of the same average over every position
where the whole kernel fits on the map, from which the caller picks a blank-sky
region for the floor and the kernel-to-kernel scatter.
"""
import numpy as np
from scipy.ndimage import convolve


def gaussian_kernel_stamp(fwhm_pix, truncate_fwhm=2.0):
    """Circular Gaussian (peak 1) truncated at `truncate_fwhm` x FWHM, on an odd square stamp."""
    radius = truncate_fwhm * fwhm_pix
    half = int(np.ceil(radius))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = xx ** 2 + yy ** 2
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.where(r2 <= radius ** 2, np.exp(-0.5 * r2 / sigma ** 2), 0.0)


def kernel_weighted_mean(img, weight, err=None):
    """
    Mean of `img` under `weight` (same shape) over its finite pixels, with the share of the kernel
    weight that lies on finite pixels. If `err` is given, also the formal error treating pixels as
    independent (a lower bound on the true error). Returns (mean, finite_weight_fraction, formal_err).
    """
    sel = weight > 0
    fin = sel & np.isfinite(img)
    if not np.any(fin):
        return np.nan, 0.0, np.nan
    w = weight[fin]
    mean = float(np.average(img[fin], weights=w))
    frac = float(w.sum() / weight[sel].sum())
    formal = np.nan if err is None else float(np.sqrt(np.sum((err[fin] * w) ** 2)) / w.sum())
    return mean, frac, formal


def sky_kernel_means(img, stamp, min_finite_frac=0.999):
    """
    Map of the stamp-weighted mean of `img` centred on every pixel, NaN wherever less than
    `min_finite_frac` of the stamp weight falls on finite pixels (kernel off the edge, or on NaNs).
    The stamp is symmetric, so convolution and correlation agree.
    """
    finite = np.isfinite(img)
    num = convolve(np.where(finite, img, 0.0), stamp, mode="constant", cval=0.0)
    den = convolve(finite.astype(float), stamp, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den >= min_finite_frac * stamp.sum(), out, np.nan)


def robust_sky_stats(values):
    """(median, 1.4826 x MAD) of the finite `values`: the floor and kernel-to-kernel scatter."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med)))
