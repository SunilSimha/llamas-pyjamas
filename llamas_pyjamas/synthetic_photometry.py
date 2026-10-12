"""
Synthetic broad-band photometry from flux-calibrated LLAMAS cubes.

A cube's f_lambda spectrum in every spaxel is integrated through a filter
response (sedpy) to give a band-averaged AB flux density per spaxel; summing
those over a set of spaxels and converting gives an AB magnitude for that
region.

Units: a cube pixel is a weighted MEAN of the fibre fluxes around it
(simple_cube_constructor normalises the kernel weights to 1), and the flux
calibration sums native fibre flux, so a pixel holds the flux per FIBRE
footprint (hexagonal, (sqrt(3)/2) * pitch^2 = 0.487 arcsec^2 at the 0.75"
pitch), not per 0.3" pixel (0.09 arcsec^2). Summing pixels over a region
therefore overcounts its flux by fibre area / pixel area (5.4x); pass
scale=spaxel_flux_scale(header) to band_fnu_image to correct for it. The integral is additive in wavelength, so a band that straddles two
channels (e.g. g, which spans the blue and green cubes) is built by handing
band_fnu_image() one segment per channel, each restricted to its own
wavelength range.

Filter response is treated as a photon-counting response (as sedpy's filters
are), so the band-averaged flux density is

    f_nu = int f_lambda * lambda * T dlambda / (c * int T dlambda / lambda)

which reduces to the constant f_nu for an f_nu = const spectrum.
"""
import numpy as np
from astropy.io import fits

C_ANGSTROM_PER_S = 2.99792458e18
AB_ZEROPOINT_CGS = 48.60  # AB = -2.5 log10(f_nu [erg/s/cm2/Hz]) - 48.60
CUBE_FLUX_UNIT = 1e-17  # the cubes are in units of 1e-17 erg/s/cm2/Angstrom


def load_filter(name):
    """sedpy Filter by name (e.g. 'decam_g', 'panstarrs_g'); needs astro-sedpy (pip install astro-sedpy)."""
    from sedpy import observate
    return observate.Filter(name)


def cube_wavelength(header):
    """Wavelength axis (Angstrom) of a cube from its CRVAL3/CRPIX3/PC3_3*CDELT3 (stored in metres)."""
    n = header["NAXIS3"]
    step = header.get("PC3_3", 1.0) * header.get("CDELT3", 1.0)
    return (header["CRVAL3"] + (np.arange(n) + 1 - header["CRPIX3"]) * step) * 1e10


def load_cube(path):
    """(wavelength [A], flux, variance) in 1e-17 erg/s/cm2/A per spaxel, arrays shaped (n_wave, ny, nx)."""
    with fits.open(path) as hdul:
        wave = cube_wavelength(hdul[0].header)
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float64)
        var = np.asarray(hdul["VAR"].data, dtype=np.float64)
    return wave, flux, var


def load_spaxel_flux_scale(path):
    """spaxel_flux_scale() of a cube file."""
    return spaxel_flux_scale(fits.getheader(path, 0))


def _trapezoid_weights(x):
    """Weights w such that sum(w * y) is the trapezoid integral of y over x."""
    w = np.zeros_like(x, dtype=np.float64)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def fiber_area_arcsec2(fiber_pitch_arcsec):
    """Area of one hexagonal-packed fibre footprint: (sqrt(3)/2) * pitch^2."""
    return 0.5 * np.sqrt(3.0) * fiber_pitch_arcsec ** 2


def spaxel_flux_scale(header):
    """
    Factor that turns a cube pixel's value (flux per fibre footprint, see the
    module docstring) into that pixel's own share of the flux: pixel area /
    fibre area, from the cube header's PIXSIZE and FIBPITCH.
    """
    return header["PIXSIZE"] ** 2 / fiber_area_arcsec2(header["FIBPITCH"])


def band_normalization(filt):
    """c * int T dlambda / lambda over the whole filter, in Angstrom/s (the dlambda/lambda part is dimensionless)."""
    return C_ANGSTROM_PER_S * np.trapezoid(filt.transmission / filt.wavelength, filt.wavelength)


def band_coverage(wave_ranges, filt):
    """Fraction of the filter's int T dlambda/lambda lying inside the union of (lo, hi) wavelength ranges."""
    lam, resp = filt.wavelength, filt.transmission
    inside = np.zeros(lam.size, dtype=bool)
    for lo, hi in wave_ranges:
        inside |= (lam >= lo) & (lam <= hi)
    integrand = resp / lam
    return float(np.sum(integrand[inside] * _trapezoid_weights(lam)[inside]) / np.sum(integrand * _trapezoid_weights(lam)))


def band_fnu_image(segments, filt, scale=1.0):
    """
    Per-spaxel band flux density, plus its formal variance.

    segments: list of (wave, flux, var, lo, hi) -- the cube (see load_cube)
    restricted to lo <= wave < hi (hi inclusive for the last one is not
    special-cased: pass hi = wave.max() + 1 to keep the last pixel). Wavelength
    ranges of different segments must not overlap, or that part of the band
    is counted twice.

    scale: multiplies the flux (and scale**2 the variance) -- pass
    spaxel_flux_scale(header) to get each pixel's own flux rather than the
    flux per fibre footprint.

    Returns (fnu, fnu_var) in erg/s/cm2/Hz, shape (ny, nx). Flux outside the
    covered ranges contributes nothing (see band_coverage to know how much
    of the filter that is). The variance sums VAR over wavelength with
    squared weights and so ignores spectral and spatial covariance.
    """
    norm = band_normalization(filt)
    num = None
    num_var = None
    for wave, flux, var, lo, hi in segments:
        keep = (wave >= lo) & (wave < hi)
        if not np.any(keep):
            continue
        wk = wave[keep]
        resp = np.interp(wk, filt.wavelength, filt.transmission, left=0.0, right=0.0)
        weights = resp * wk * _trapezoid_weights(wk)  # T * lambda * dlambda
        f = flux[keep]
        v = var[keep]
        n_seg = np.einsum("l,lyx->yx", weights, np.nan_to_num(f))
        v_seg = np.einsum("l,lyx->yx", weights ** 2, np.nan_to_num(v))
        num = n_seg if num is None else num + n_seg
        num_var = v_seg if num_var is None else num_var + v_seg
    if num is None:
        raise ValueError("no cube wavelengths fall inside the requested segments")
    unit = scale * CUBE_FLUX_UNIT / norm
    return num * unit, num_var * unit ** 2


def fnu_to_ab(fnu):
    """AB magnitude of a flux density in erg/s/cm2/Hz (NaN for non-positive flux)."""
    fnu = np.asarray(fnu, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fnu > 0, -2.5 * np.log10(fnu) - AB_ZEROPOINT_CGS, np.nan)


def aperture_ab_mag(fnu_image, fnu_var_image, mask):
    """(AB mag, 1-sigma mag error) of the summed flux over the spaxels where mask is True."""
    total = float(np.sum(fnu_image[mask]))
    sigma = float(np.sqrt(np.sum(fnu_var_image[mask])))
    mag = float(fnu_to_ab(total))
    return mag, (2.5 / np.log(10)) * sigma / total if total > 0 else np.nan


def nanomaggies_to_ab(flux_nmgy):
    """AB magnitude of a Legacy Survey / SDSS-style flux in nanomaggies (AB = 22.5 - 2.5 log10 flux)."""
    flux_nmgy = np.asarray(flux_nmgy, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flux_nmgy > 0, 22.5 - 2.5 * np.log10(flux_nmgy), np.nan)
