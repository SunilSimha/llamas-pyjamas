"""Unit tests for llamas_pyjamas.map_psf (PSF size from compact sources in a 2D map)."""

import numpy as np
import pytest
from photutils.psf import GaussianPRF
from scipy import ndimage

from llamas_pyjamas import map_psf

FWHM_GUESS = 6.7  # 2" at 0.3"/px, as used on the LLAMAS maps
BOX = 17


def _correlated_noise(shape, sigma_pix, std, rng):
    noise = ndimage.gaussian_filter(rng.normal(size=shape), sigma_pix)
    return noise * (std / noise.std())


def _scene(fwhm, q, rng, peak_sn=50.0, noise_std=1.0, shape=(61, 61)):
    yy, xx = np.indices(shape)
    x0, y0 = 30.3, 29.6
    x_fwhm, y_fwhm = fwhm / np.sqrt(q), fwhm * np.sqrt(q)
    sx, sy = x_fwhm / 2.3548, y_fwhm / 2.3548
    src = GaussianPRF(flux=peak_sn * noise_std * 2 * np.pi * sx * sy, x_0=x0, y_0=y0,
                      x_fwhm=x_fwhm, y_fwhm=y_fwhm, theta=25.0)
    plane = 3.0 + 0.05 * xx - 0.03 * yy
    img = src(xx, yy) + plane + _correlated_noise(shape, 2.5, noise_std, rng)
    return img, np.full(shape, noise_std), (x0, y0)


@pytest.mark.parametrize("fwhm", [5.0, 7.0, 9.0])
@pytest.mark.parametrize("q", [1.0, 0.7])
def test_fit_recovers_fwhm_and_axis_ratio(fwhm, q):
    rng = np.random.default_rng(42)
    ratios, qs = [], []
    for _ in range(10):
        img, err, (x0, y0) = _scene(fwhm, q, rng)
        r = map_psf.fit_gaussian_psf(img, err, x0 + 0.4, y0 - 0.3, BOX, FWHM_GUESS)
        assert r["success"] and not r["at_bound"]
        ratios.append(r["fwhm"] / fwhm)
        qs.append(r["q"])
    assert abs(np.median(ratios) - 1.0) < 0.02
    assert abs(np.median(qs) - q) < 0.05


def test_single_pixel_spike_is_rejected_without_fwhm_cut():
    rng = np.random.default_rng(1)
    img, err, (x0, y0) = _scene(7.0, 1.0, rng, shape=(81, 81))
    img[60, 20] += 400.0  # isolated one-pixel fit-divergence-like spike
    xy = map_psf.find_clump_candidates(img - np.median(img), err, FWHM_GUESS)
    fits = [map_psf.fit_gaussian_psf(img, err, x, y, BOX, FWHM_GUESS) for x, y in xy]
    accepted = map_psf.select_psf_clumps(fits)
    assert all(np.hypot(a["x"] - 20, a["y"] - 60) > 3 for a in accepted)
    assert any(np.hypot(a["x"] - x0, a["y"] - y0) < 1 for a in accepted)


def test_inject_recover_unbiased_on_smooth_map():
    rng = np.random.default_rng(7)
    shape = (81, 81)
    yy, xx = np.indices(shape)
    img = 3.0 + 0.02 * xx + _correlated_noise(shape, 2.5, 1.0, rng)
    res = map_psf.inject_recover(img, np.ones(shape), fwhm_pix=7.0, peak_sn=40.0, box_pix=BOX,
                                 fwhm_guess_pix=FWHM_GUESS, footprint=np.ones(shape, bool),
                                 avoid_xy=np.empty((0, 2)), avoid_radius_pix=0.0, n_trials=30, rng=rng)
    assert res["n_ok"] >= 25
    assert abs(res["bias"]) < 0.03


def test_summarize_psf_uses_sources_consistent_with_minimum():
    accepted = [dict(fwhm=6.0, fwhm_err=0.2), dict(fwhm=6.3, fwhm_err=0.2), dict(fwhm=8.0, fwhm_err=0.2)]
    s = map_psf.summarize_psf(accepted)
    assert s["fwhm_min"] == 6.0 and s["n_consistent"] == 2
    assert s["fwhm_best"] == pytest.approx(6.15)
