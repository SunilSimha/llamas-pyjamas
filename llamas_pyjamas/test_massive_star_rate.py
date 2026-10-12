"""Checks for massive_star_rate: IMF normalisation against Beniamini+2019, and the SB -> surface-luminosity geometry."""
import numpy as np
from astropy.cosmology import Planck18

import massive_star_rate as msr


def test_beniamini_milky_way_rate():
    # Beniamini et al. 2019 sec. 2.1: 1.65 Msun/yr with a Kroupa (2001) IMF -> 17 +/- 2 stars (M > 8 Msun) per kyr.
    for m_lo in (0.01, 0.08, 0.1):
        rate = 1.65 * msr.massive_stars_per_msun(msr.kroupa2001_imf, m_lo=m_lo) * 1e3
        assert 15.0 <= rate <= 19.0
    assert np.isclose(msr.massive_stars_per_msun(), 0.01113, rtol=2e-3)


def test_salpeter_number_per_mass():
    # Analytic: N(8-100)/M(0.1-100) for dN/dm ~ m^-2.35.
    a = 2.35
    n = (8 ** (1 - a) - 100 ** (1 - a)) / (a - 1)
    m = (100 ** (2 - a) - 0.1 ** (2 - a)) / (2 - a)
    assert np.isclose(msr.massive_stars_per_msun(msr.salpeter_imf), n / m, rtol=1e-6)


def test_surface_luminosity_matches_distance_route():
    # Same patch via 4 pi d_L^2 F / (Omega D_A^2) at a finite redshift.
    z, sb = 0.04, 1e-17
    omega_sr = 1.0 / msr.ARCSEC2_PER_SR
    d_l = Planck18.luminosity_distance(z).to("cm").value
    d_a_kpc = Planck18.angular_diameter_distance(z).to("kpc").value
    direct = 4 * np.pi * d_l ** 2 * sb * 1.0 / (omega_sr * d_a_kpc ** 2)
    assert np.isclose(msr.halpha_surface_luminosity(sb, z), direct, rtol=1e-6)


def test_log_offset_is_additive():
    off = msr.log_massive_star_rate_offset()
    for sb17 in (0.3, 4.0, 150.0):
        assert np.isclose(np.log10(msr.massive_star_rate_surface_density(sb17 * 1e-17)), np.log10(sb17) + off)
