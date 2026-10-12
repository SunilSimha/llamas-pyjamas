"""Checks for synthetic_photometry: constant-f_nu recovery, agreement with sedpy, and additivity across segments."""
import numpy as np

import synthetic_photometry as sp


def _flat_fnu_cube(wave, fnu_cgs, ny=2, nx=3):
    f_lambda = fnu_cgs * sp.C_ANGSTROM_PER_S / wave ** 2 / sp.CUBE_FLUX_UNIT
    flux = np.broadcast_to(f_lambda[:, None, None], (wave.size, ny, nx)).copy()
    return flux, np.ones_like(flux)


def test_flat_fnu_recovers_ab_mag_and_matches_sedpy():
    filt = sp.load_filter("decam_g")
    wave = np.arange(3300.0, 6000.0, 0.8)
    fnu = 10 ** (-0.4 * (20.0 + sp.AB_ZEROPOINT_CGS))  # AB = 20
    flux, var = _flat_fnu_cube(wave, fnu)
    img, _ = sp.band_fnu_image([(wave, flux, var, wave.min(), wave.max() + 1)], filt)
    assert np.allclose(sp.fnu_to_ab(img), 20.0, atol=2e-3)
    img2, _ = sp.band_fnu_image([(wave, flux, var, wave.min(), wave.max() + 1)], filt, scale=0.25)
    assert np.allclose(img2, 0.25 * img)
    sedpy_mag = filt.ab_mag(wave, flux[:, 0, 0] * sp.CUBE_FLUX_UNIT)
    assert abs(sedpy_mag - 20.0) < 2e-3


def test_two_segments_equal_one():
    filt = sp.load_filter("decam_g")
    wave = np.arange(3300.0, 6000.0, 0.8)
    flux, var = _flat_fnu_cube(wave, 1e-28)
    one, _ = sp.band_fnu_image([(wave, flux, var, 3300.0, 6001.0)], filt)
    two, _ = sp.band_fnu_image([(wave, flux, var, 3300.0, 4700.0), (wave, flux, var, 4700.0, 6001.0)], filt)
    assert np.allclose(one, two, rtol=1e-6)


def test_spaxel_flux_scale():
    assert abs(sp.spaxel_flux_scale({"PIXSIZE": 0.3, "FIBPITCH": 0.75}) - 0.09 / 0.48714) < 1e-4


def test_coverage():
    filt = sp.load_filter("decam_g")
    assert sp.band_coverage([(0, 1e5)], filt) == 1.0
    assert 0.3 < sp.band_coverage([(4553.0, 7116.0)], filt) < 0.8


if __name__ == "__main__":
    test_flat_fnu_recovers_ab_mag_and_matches_sedpy()
    test_two_segments_equal_one()
    test_coverage()
    test_spaxel_flux_scale()
    print("all passed")
