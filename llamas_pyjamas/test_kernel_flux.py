"""Checks for kernel_flux: kernel-mean recovery, the zero-bounded sky floor, and the correlated-noise scatter."""
import numpy as np
from scipy.ndimage import gaussian_filter

import kernel_flux as kf


def test_stamp_truncation_and_peak():
    s = kf.gaussian_kernel_stamp(6.0)
    assert s.shape == (25, 25) and s[12, 12] == 1.0
    assert s[12, 12 + 12] == 0.0 or s[12, 12 + 12] > 0  # radius 12 px sits on the boundary
    assert s[0, 0] == 0.0
    assert np.isclose(s.sum(), 2 * np.pi * (6.0 / 2.3548) ** 2, rtol=0.02)


def test_kernel_weighted_mean_recovers_uniform_field_and_reports_missing_weight():
    img = np.full((60, 60), 3.0)
    stamp = kf.gaussian_kernel_stamp(5.0)
    weight = np.zeros_like(img)
    weight[30 - 10:30 + 11, 30 - 10:30 + 11] = stamp
    mean, frac, formal = kf.kernel_weighted_mean(img, weight, err=np.full_like(img, 0.2))
    assert np.isclose(mean, 3.0) and np.isclose(frac, 1.0)
    assert formal < 0.2 / 5  # many weighted pixels: well below the per-pixel error
    img[:, :30] = np.nan  # half the kernel on NaNs
    mean, frac, _ = kf.kernel_weighted_mean(img, weight)
    assert np.isclose(mean, 3.0) and 0.4 < frac < 0.6


def test_sky_floor_and_correlated_scatter():
    rng = np.random.default_rng(1)
    sigma_pix, corr = 1.0, 2.5
    noise = gaussian_filter(rng.normal(0, 1, (400, 400)), corr)
    noise *= sigma_pix / noise.std()
    img = np.clip(noise, 0, None)  # zero-bounded amplitude fit: sky is a positive floor
    stamp = kf.gaussian_kernel_stamp(6.0)
    sky = kf.sky_kernel_means(img, stamp)
    med, scatter = kf.robust_sky_stats(sky[60:-60, 60:-60])
    assert med > 0.2  # floor survives averaging (rectified noise mean is 0.4 sigma_pix)
    formal = sigma_pix * np.sqrt(np.sum(stamp ** 2)) / stamp.sum()
    assert scatter > 2 * formal  # correlated noise: independent-pixel error is far too small
    # kernel_weighted_mean at a position equals the convolution map there, and a uniform source adds exactly its flux.
    weight = np.zeros_like(img)
    h = stamp.shape[0] // 2
    weight[200 - h:200 + h + 1, 200 - h:200 + h + 1] = stamp
    mean, _, _ = kf.kernel_weighted_mean(img, weight)
    assert np.isclose(mean, sky[200, 200])
    assert np.isclose(kf.sky_kernel_means(img + 5.0, stamp)[200, 200], sky[200, 200] + 5.0)


def test_sky_kernel_means_excludes_incomplete_kernels():
    img = np.ones((50, 50))
    img[25, 25] = np.nan
    stamp = kf.gaussian_kernel_stamp(4.0)
    sky = kf.sky_kernel_means(img, stamp)
    assert np.isnan(sky[0, 0]) and np.isnan(sky[25, 25])
    assert np.isclose(sky[25, 10], 1.0)
