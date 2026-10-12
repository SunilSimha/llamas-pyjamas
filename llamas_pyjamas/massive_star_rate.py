"""
Halpha surface brightness -> star-formation-rate surface density -> formation
rate of massive (>= 8 Msun) stars per projected kpc^2.

    L_A        = 4 pi (1+z)^4 Sigma_Halpha                 (erg s^-1 kpc^-2)
    Sigma_SFR  = L_A / C_Halpha                            (Msun yr^-1 kpc^-2)
    Ndot_>=8   = Sigma_SFR * n_8 * 1e3                     (kpc^-2 kyr^-1)

with Sigma_Halpha in erg s^-1 cm^-2 sr^-1 times kpc_cm^2 (the per-kpc^2 surface
luminosity is distance independent apart from the (1+z)^4 dimming), C_Halpha the
Kennicutt & Evans (2012) Table 1 calibration (log C = 41.27, Kroupa-type IMF,
0.1-100 Msun; Murphy et al. 2011), and n_8 the number of stars with
M >= 8 Msun formed per Msun of stars formed, for the chosen IMF over the same
0.1-100 Msun range the SFR refers to. The default IMF is Kroupa (2001), as in
Beniamini et al. (2019, sec. 2.1), whose Milky Way SFR of 1.65 Msun/yr gives
17 +/- 2 such stars per kyr.

No dust correction and no inclination correction are applied here.
"""
import numpy as np
from scipy.integrate import quad

KPC_CM = 3.0856775814913673e21
ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
LOG_C_HALPHA_KE12 = 41.27      # Kennicutt & Evans 2012, Table 1: log SFR = log L(Halpha) - 41.27
C_HALPHA_K98_SALPETER = 7.9e-42  # Kennicutt 1998 (Salpeter 0.1-100 Msun), for comparison only


def kroupa2001_imf(m):
    """Kroupa (2001) dN/dm (unnormalised): slope -0.3 below 0.08, -1.3 to 0.5, -2.3 above (continuous)."""
    m = float(m)
    if m < 0.08:
        return (m / 0.08) ** -0.3 * (0.08 / 0.5) ** -1.3
    if m < 0.5:
        return (m / 0.5) ** -1.3
    return (m / 0.5) ** -2.3


def kroupa_weidner2003_imf(m):
    """IMF behind the Kennicutt & Evans (2012) calibrations: slope -1.3 over 0.1-1, -2.35 over 1-100 Msun."""
    m = float(m)
    return m ** -1.3 if m < 1.0 else m ** -2.35


def salpeter_imf(m):
    return float(m) ** -2.35


def massive_stars_per_msun(imf=kroupa2001_imf, m_threshold=8.0, m_lo=0.1, m_hi=100.0):
    """Number of stars with m_threshold <= M <= m_hi per Msun of stars formed over [m_lo, m_hi]."""
    breaks = [b for b in (0.08, 0.5, 1.0) if m_lo < b < m_hi]
    mass = quad(lambda m: m * imf(m), m_lo, m_hi, points=breaks or None, limit=200)[0]
    number = quad(imf, m_threshold, m_hi, limit=200)[0]
    return number / mass


def halpha_surface_luminosity(sb_arcsec2, z=0.0):
    """erg s^-1 kpc^-2 (projected) from an observed Halpha SB in erg s^-1 cm^-2 arcsec^-2.
    L/A = 4 pi Sigma d_L^2 / D_A^2 = 4 pi (1+z)^4 Sigma, with Sigma per steradian."""
    return 4.0 * np.pi * (1.0 + z) ** 4 * np.asarray(sb_arcsec2) * ARCSEC2_PER_SR * KPC_CM ** 2


def sfr_surface_density(sb_arcsec2, z=0.0, log_c=LOG_C_HALPHA_KE12):
    """Msun yr^-1 kpc^-2 from an observed Halpha SB in erg s^-1 cm^-2 arcsec^-2."""
    return halpha_surface_luminosity(sb_arcsec2, z) / 10.0 ** log_c


def massive_star_rate_surface_density(sb_arcsec2, z=0.0, log_c=LOG_C_HALPHA_KE12, **imf_kwargs):
    """Formation rate of M >= 8 Msun stars (or imf_kwargs' m_threshold) in kpc^-2 kyr^-1."""
    return sfr_surface_density(sb_arcsec2, z, log_c) * massive_stars_per_msun(**imf_kwargs) * 1e3


def log_massive_star_rate_offset(sb_unit=1e-17, z=0.0, log_c=LOG_C_HALPHA_KE12, **imf_kwargs):
    """Additive constant: log10(Ndot / kpc^-2 kyr^-1) = log10(Sigma / sb_unit) + this."""
    return float(np.log10(massive_star_rate_surface_density(sb_unit, z, log_c, **imf_kwargs)))
