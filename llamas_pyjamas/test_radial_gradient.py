"""Checks for radial_gradient: geometry under flipped/rotated WCSs, gradient recovery, block bootstrap, leave-site-out."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

import radial_gradient as rg

SHAPE = (120, 120)
PIXSCALE = 0.3  # arcsec
CENTER_PIX = (60.0, 55.0)  # x, y


def make_wcs(flip_x=True, rotate_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [51.6, -54.6]
    w.wcs.crpix = [CENTER_PIX[0] + 1.0, CENTER_PIX[1] + 1.0]  # FITS is 1-based
    theta = np.radians(rotate_deg)
    scale = PIXSCALE / 3600.0
    sign = -1.0 if flip_x else 1.0
    w.wcs.cd = scale * np.array([[sign * np.cos(theta), -np.sin(theta)], [sign * np.sin(theta), np.cos(theta)]])
    return w


def center_of(w):
    return w.pixel_to_world(*CENTER_PIX)


def disk_grid(w, pa_deg, q):
    return rg.deprojected_radius(w, center_of(w), pa_deg, q, SHAPE)


def exponential_disk(radius, scale=8.0):
    return 100.0 * np.exp(-radius / scale)


def test_disk_radius_along_major_and_minor_axes():
    center = SkyCoord(51.6 * u.deg, -54.6 * u.deg)
    pa, q = 30.0, 0.5
    along_major = center.directional_offset_by(pa * u.deg, 10.0 * u.arcsec)
    along_minor = center.directional_offset_by((pa + 90.0) * u.deg, 5.0 * u.arcsec)
    assert np.isclose(rg.disk_radius(center, along_major, pa, q)[0], 10.0, rtol=1e-3)
    assert np.isclose(rg.disk_radius(center, along_minor, pa, q)[0], 10.0, rtol=1e-3)


@pytest.mark.parametrize("flip_x,rotate_deg", [(True, 0.0), (False, 0.0), (True, 35.0), (False, 120.0)])
def test_deprojected_radius_does_not_depend_on_wcs_orientation(flip_x, rotate_deg):
    w = make_wcs(flip_x, rotate_deg)
    radius, _az = disk_grid(w, 40.0, 0.6)
    assert radius[int(CENTER_PIX[1]), int(CENTER_PIX[0])] < 1e-6
    target = center_of(w).directional_offset_by(40.0 * u.deg, 12.0 * u.arcsec)
    x, y = w.world_to_pixel(target)
    assert abs(radius[int(round(float(y))), int(round(float(x)))] - 12.0) < PIXSCALE


def test_continuum_peak_ignores_bright_offset_light():
    w = make_wcs()
    radius, _ = disk_grid(w, 0.0, 1.0)
    img = exponential_disk(radius)
    yy, xx = np.indices(SHAPE)
    img = img + 20.0 * np.exp(-0.5 * ((xx - 100.0) ** 2 + (yy - 55.0) ** 2) / 15.0 ** 2)  # a broad, offset knot
    x, y = rg.continuum_peak(img)
    assert abs(x - CENTER_PIX[0]) <= 2 and abs(y - CENTER_PIX[1]) <= 2


@pytest.mark.parametrize("flip_x,rotate_deg", [(True, 0.0), (True, 35.0), (False, 120.0)])
def test_isophotes_recover_input_pa_and_axis_ratio(flip_x, rotate_deg):
    w = make_wcs(flip_x, rotate_deg)
    pa, q = 65.0, 0.6
    radius, _ = disk_grid(w, pa, q)
    rng = np.random.default_rng(1)
    img = exponential_disk(radius) + rng.normal(0.0, 0.02, SHAPE)
    iso = rg.isophote_geometry(img, w, *CENTER_PIX, sma_min=2.0, sma_max=12.0)
    assert iso["sma"].size > 3
    mean_pa, _std = rg.axial_mean_std(iso["pa_deg"])
    assert abs(((mean_pa - pa) + 90.0) % 180.0 - 90.0) < 3.0
    assert np.isclose(np.median(1.0 - iso["eps"]), q, atol=0.03)


def test_axial_mean_handles_wraparound():
    mean, std = rg.axial_mean_std([178.0, 2.0])
    assert min(mean, 180.0 - mean) < 1e-6
    assert std < 3.0


@pytest.mark.parametrize("flip_x,rotate_deg", [(True, 0.0), (False, 120.0)])
def test_kinematic_pa_recovers_receding_side(flip_x, rotate_deg):
    w = make_wcs(flip_x, rotate_deg)
    pa = 250.0
    yy, xx = np.indices(SHAPE)
    d_east, d_north = center_of(w).spherical_offsets_to(w.pixel_to_world(xx, yy))
    along = d_east.to_value(u.arcsec) * np.sin(np.radians(pa)) + d_north.to_value(u.arcsec) * np.cos(np.radians(pa))
    z_map = 0.05 + 8.0 * along / rg.C_KMS * 1.05
    fitted, gradient, rms, _n = rg.kinematic_position_angle(w, center_of(w), z_map, np.ones(SHAPE, dtype=bool))
    assert abs(fitted - pa) < 0.5
    assert np.isclose(gradient, 8.0, rtol=0.01)
    assert rms < 1.0


def rotating_disk(w, pa, q, v_max=150.0, r_turn=4.0):
    """Line-of-sight velocity of a thin disk with an arctan rotation curve; pa is the receding side."""
    radius, azimuth = disk_grid(w, pa, q)
    return v_max * (2.0 / np.pi) * np.arctan(radius / r_turn) * np.sqrt(1.0 - q ** 2) * np.cos(azimuth)


@pytest.mark.parametrize("flip_x,rotate_deg", [(True, 0.0), (False, 120.0)])
def test_kinemetry_recovers_pa_and_flattening(flip_x, rotate_deg):
    w = make_wcs(flip_x, rotate_deg)
    pa, q = 250.0, 0.6
    rings = rg.kinemetry_rings(rotating_disk(w, pa, q), w, *CENTER_PIX, sma_arcsec=[4.0, 8.0, 12.0], pa_step=6.0, q_step=0.04)
    assert np.all(np.abs(((rings["pa_deg"] - pa) + 180.0) % 360.0 - 180.0) < 3.0)
    assert np.all(np.abs(rings["q"] - q) < 0.05)
    assert np.all(rings["k1"] > 0.0)


def test_kinemetry_rejects_a_ring_off_the_field():
    w = make_wcs()
    velocity = rotating_disk(w, 90.0, 0.8)
    velocity[:, 75:] = np.nan
    rings = rg.kinemetry_rings(velocity, w, *CENTER_PIX, sma_arcsec=[2.0, 12.0], pa_step=10.0, q_step=0.1)
    assert np.isfinite(rings["q"][0])
    assert np.isnan(rings["q"][1])


def linear_map(radius, slope=-0.02, zero_point=8.8):
    return zero_point + slope * radius


def test_profile_and_fit_recover_a_linear_gradient():
    w = make_wcs()
    radius, _ = disk_grid(w, 30.0, 0.7)
    rng = np.random.default_rng(2)
    values = linear_map(radius) + rng.normal(0.0, 0.03, SHAPE)
    valid = radius < 20.0
    edges = np.arange(0.0, 21.0, 2.0)
    profile = rg.radial_profile(values, radius, valid, valid, edges)
    slope, zero_point = rg.fit_gradient(profile["r_median"], profile["median"])
    assert np.isclose(slope, -0.02, atol=1e-3)
    assert np.isclose(zero_point, 8.8, atol=0.01)


def test_coverage_counts_the_field_inside_each_annulus():
    w = make_wcs()
    radius, _ = disk_grid(w, 30.0, 0.7)
    field = np.ones(SHAPE, dtype=bool)
    edges = np.array([4.0, 8.0])
    profile = rg.radial_profile(radius, radius, field, field, edges, field=field, pixel_area=PIXSCALE ** 2, axis_ratio=0.7)
    assert np.isclose(profile["coverage"][0], 1.0, atol=0.02)
    half_field = np.indices(SHAPE)[1] < CENTER_PIX[0]
    profile = rg.radial_profile(radius, radius, field, field, edges, field=half_field, pixel_area=PIXSCALE ** 2,
                                axis_ratio=0.7)
    assert np.isclose(profile["coverage"][0], 0.5, atol=0.03)


def test_fit_bins_stops_at_the_first_failing_bin():
    profile = dict(r_lo=np.array([0.0, 2.0, 4.0, 6.0, 8.0]), valid_frac=np.array([1.0, 0.9, 0.2, 0.9, 0.9]),
                   n_valid=np.full(5, 100))
    assert rg.fit_bins(profile, 2.0, 0.3, 10).tolist() == [False, True, False, False, False]


def test_fit_bins_skips_leading_failures():
    profile = dict(r_lo=np.array([0.0, 2.0, 4.0, 6.0]), valid_frac=np.array([1.0, 0.1, 0.9, 0.9]), n_valid=np.full(4, 100))
    assert rg.fit_bins(profile, 2.0, 0.3, 10).tolist() == [False, False, True, True]


def test_block_bootstrap_widens_with_correlated_noise():
    w = make_wcs()
    radius, _ = disk_grid(w, 30.0, 1.0)
    valid = radius < 16.0
    edges = np.arange(2.0, 16.1, 2.0)
    use = np.ones(len(edges) - 1, dtype=bool)
    rng = np.random.default_rng(3)
    white = rng.normal(0.0, 0.03, SHAPE)
    correlated = gaussian_filter(rng.normal(0.0, 1.0, SHAPE), 4.0)
    correlated *= 0.03 / np.std(correlated)
    labels = rg.block_labels(SHAPE, 8)
    spreads = []
    for noise in (white, correlated):
        slopes, _ = rg.block_bootstrap_gradient(linear_map(radius) + noise, radius, valid, edges, use, labels, n_boot=200)
        spreads.append(np.std(slopes))
    assert spreads[1] > 1.5 * spreads[0]


def test_leaving_the_site_out_recovers_a_local_excess():
    w = make_wcs()
    radius, _ = disk_grid(w, 30.0, 1.0)
    yy, xx = np.indices(SHAPE)
    site_x, site_y = 85.0, 55.0
    near_site = np.hypot(xx - site_x, yy - site_y) < 10.0
    values = linear_map(radius) + 0.3 * near_site
    valid = radius < 16.0
    edges = np.arange(2.0, 16.1, 2.0)
    r_site = radius[int(site_y), int(site_x)]

    def residual(mask):
        profile = rg.radial_profile(values, radius, mask, mask, edges)
        slope, zero_point = rg.fit_gradient(profile["r_median"], profile["median"])
        return values[int(site_y), int(site_x)] - (zero_point + slope * r_site)

    assert np.isclose(residual(valid & ~near_site), 0.3, atol=0.01)


def test_annulus_statistics_and_scatter():
    radius = np.linspace(0.0, 10.0, 1001)
    values = np.linspace(0.0, 1.0, 1001)
    stats = rg.annulus_statistics(0.6, values, radius, np.ones(1001, dtype=bool), 5.0, 1.0)
    assert stats["n"] == 201
    assert np.isclose(stats["mean"], 0.5, atol=1e-6)
    assert np.isclose(stats["fraction_below"], 1.0, atol=0.01)
    assert np.isclose(stats["half_p16_p84"], 0.068, atol=0.005)
    assert np.isclose(stats["z_score"], 0.1 / stats["std"])
    empty = rg.annulus_statistics(0.5, values, radius, np.zeros(1001, dtype=bool), 5.0, 1.0)
    assert empty["n"] == 0 and np.isnan(empty["z_score"])


def test_bpt_lines_classify_known_points():
    # A star-forming point, one between the lines (composite), and an AGN-like one, plus one right of both asymptotes.
    n2 = np.array([-1.0, -0.2, 0.2, 0.6, np.nan])
    o3 = np.array([0.0, 0.2, 1.0, -0.5, 0.0])
    assert rg.above_bpt_line(n2, o3, "kauffmann03").tolist() == [False, True, True, True, False]
    assert rg.above_bpt_line(n2, o3, "kewley01").tolist() == [False, False, True, True, False]


def test_site_residual_combines_terms_in_quadrature():
    residual, total, significance = rg.site_residual(8.7, 0.03, 8.6, 0.0, 0.04)
    assert np.isclose(residual, 0.1)
    assert np.isclose(total, 0.05)
    assert np.isclose(significance, 2.0)


def test_beam_smearing_is_small_for_a_small_kernel_and_flattens_for_a_large_one():
    w = make_wcs()
    radius, _ = disk_grid(w, 30.0, 1.0)
    halpha = exponential_disk(radius, scale=6.0)
    halpha[radius > 18.0] = np.nan
    edges = np.arange(2.0, 14.1, 2.0)
    use = np.ones(len(edges) - 1, dtype=bool)
    small = rg.beam_smearing_flattening(halpha, radius, -0.02, 1.0, edges, use)
    large = rg.beam_smearing_flattening(halpha, radius, -0.02, 20.0, edges, use)
    assert small > 0.95
    assert large < small
