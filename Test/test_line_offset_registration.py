"""
Tests for the line-to-line registration section of astrometry_correction.py
(measure_line_offset, resample_*_with_offset, and the linemaps / cube-slab
writers). Synthetic data only -- no data files needed.

The point of several of these is to pin down conventions that are easy to get
backwards (sign of the shift, East vs West on an East-left grid) by checking
them numerically rather than trusting the docstrings.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from llamas_pyjamas.astrometry_correction import (
    measure_line_offset, resample_map_with_offset, resample_cube_with_offset,
    write_offset_corrected_linemaps, write_offset_cube_slabs, read_offset_cube_slabs,
)

PIX_ARCSEC = 0.3
SHAPE = (90, 96)  # (ny, nx), deliberately not square


def make_wcs(shape=SHAPE, crpix_offset=(0.0, 0.0)):
    """Standard North-up / East-left celestial WCS (RA increases toward -x)."""
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crpix = [nx / 2.0 + crpix_offset[0], ny / 2.0 + crpix_offset[1]]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.cdelt = [-PIX_ARCSEC / 3600.0, PIX_ARCSEC / 3600.0]
    return w


def blobs(centres, amps, sigma=3.0, shape=SHAPE):
    """Sum of circular Gaussians -- a clumpy 'galaxy'. centres are (x, y) in pixels."""
    yy, xx = np.indices(shape, dtype=float)
    img = np.zeros(shape)
    for (cx, cy), a in zip(centres, amps):
        img += a * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
    return img


CENTRES = [(40.0, 42.0), (56.0, 50.0), (34.0, 58.0)]
AMPS = [1.0, 0.6, 0.35]


def shifted_map(dx, dy, amps=AMPS):
    return blobs([(cx + dx, cy + dy) for cx, cy in CENTRES], amps)


@pytest.mark.parametrize("dx,dy", [(-2.6, 1.3), (1.7, -0.8), (0.0, 0.0), (-2.9, -0.2)])
def test_measure_recovers_subpixel_shift_and_sign(dx, dy):
    wcs = make_wcs()
    ref = blobs(CENTRES, AMPS)
    line = shifted_map(dx, dy)  # the line's features sit at (reference + shift)
    res = measure_line_offset(line, wcs, ref, wcs)
    assert res["shift_x_pix"] == pytest.approx(dx, abs=0.05)
    assert res["shift_y_pix"] == pytest.approx(dy, abs=0.05)
    assert res["agree"]


def test_east_north_sign_on_east_left_grid():
    wcs = make_wcs()
    ref = blobs(CENTRES, AMPS)
    # -x is East on this grid, +y is North.
    res = measure_line_offset(shifted_map(-2.9, 0.0), wcs, ref, wcs)
    assert res["d_east_arcsec"] == pytest.approx(2.9 * PIX_ARCSEC, abs=0.03)
    assert res["d_north_arcsec"] == pytest.approx(0.0, abs=0.03)
    res = measure_line_offset(shifted_map(0.0, 2.0), wcs, ref, wcs)
    assert res["d_north_arcsec"] == pytest.approx(2.0 * PIX_ARCSEC, abs=0.03)


def test_windowed_search_excludes_a_mismatched_clump_alias_and_flags_it():
    """
    When the line's brightest clump is a different one than the reference's (like [OIII] vs Hbeta
    peaking in different knots), correlating brightest-to-brightest is a strong alias. A search
    window smaller than the clump separation excludes it; the plain global solution is still
    reported, and the disagreement is flagged rather than silently resolved.
    """
    wcs = make_wcs()
    ref = blobs(CENTRES, AMPS)
    line = shifted_map(-2.0, 1.0, amps=[0.3, 1.0, 0.2])  # true shift (-2, +1); clumps ~16 px apart
    tight = measure_line_offset(line, wcs, ref, wcs, search_radius_arcsec=3.0)  # 10 px < 16 px
    assert tight["shift_x_pix"] == pytest.approx(-2.0, abs=0.6)
    assert tight["shift_y_pix"] == pytest.approx(1.0, abs=0.6)
    assert abs(tight["global_shift_x_pix"] - tight["shift_x_pix"]) > 5.0  # the alias, still reported
    assert not tight["agree"]
    assert any("differ" in w for w in tight["warnings"])

    # A window LARGER than the clump separation admits the alias -- documented, not hidden:
    wide = measure_line_offset(line, wcs, ref, wcs, search_radius_arcsec=5.0)  # 16.7 px > 16 px
    assert abs(wide["shift_x_pix"]) < 5.0 / PIX_ARCSEC  # still bounded by the window


def test_noise_mc_error_scales_with_the_noise_level():
    from llamas_pyjamas.astrometry_correction import estimate_offset_uncertainty
    wcs = make_wcs()
    ref, line = blobs(CENTRES, AMPS), shifted_map(-2.6, 1.0)
    sig = {}
    for level in (0.02, 0.15):
        err = np.full(SHAPE, level)
        sig[level] = estimate_offset_uncertainty(line, wcs, ref, wcs, line_err=err, ref_err=err, n_mc=40, seed=3)
    assert sig[0.15]["mc_sigma_east_arcsec"] > 2.0 * sig[0.02]["mc_sigma_east_arcsec"]
    assert sig[0.02]["mc_sigma_east_arcsec"] < 0.05  # clean, well-constrained data
    assert sig[0.02]["n_mc_ok"] == 40


def test_jackknife_is_larger_when_regions_disagree_about_the_shift():
    """A uniform shift is insensitive to which region is dropped; clumps that shift differently are not."""
    from llamas_pyjamas.astrometry_correction import estimate_offset_uncertainty
    wcs = make_wcs()
    ref = blobs(CENTRES, AMPS)
    uniform = shifted_map(-2.6, 1.0)
    mixed = blobs([(CENTRES[0][0] - 5.0, CENTRES[0][1]), (CENTRES[1][0], CENTRES[1][1]),
                   (CENTRES[2][0] + 3.0, CENTRES[2][1])], AMPS)
    ju = estimate_offset_uncertainty(uniform, wcs, ref, wcs, n_mc=0, block_arcsec=4.0)
    jm = estimate_offset_uncertainty(mixed, wcs, ref, wcs, n_mc=0, block_arcsec=4.0)
    assert ju["n_blocks"] >= 3 and jm["n_blocks"] >= 3
    assert jm["jackknife_sigma_east_arcsec"] > 3.0 * ju["jackknife_sigma_east_arcsec"]
    assert np.isnan(ju["mc_sigma_east_arcsec"])  # MC skipped without ERR maps / n_mc=0
    assert ju["sigma_east_arcsec"] == pytest.approx(ju["jackknife_sigma_east_arcsec"])


def test_compute_global_false_skips_the_global_solution():
    wcs = make_wcs()
    res = measure_line_offset(shifted_map(-2.0, 0.5), wcs, blobs(CENTRES, AMPS), wcs, compute_global=False)
    assert np.isnan(res["global_shift_x_pix"]) and res["agree"]
    assert res["shift_x_pix"] == pytest.approx(-2.0, abs=0.05)


@pytest.mark.parametrize("shape", [(40, 52), (48, 48)])
@pytest.mark.parametrize("name", ["identity", "flip_y", "flip_x", "rot180", "transpose", "rot90", "rot270", "anti_transpose"])
def test_find_dihedral_match_recovers_each_transform(name, shape):
    from llamas_pyjamas.astrometry_correction import apply_dihedral, find_dihedral_match
    rng = np.random.default_rng(5)
    raw = rng.normal(size=shape) + blobs([(10, 12), (30, 20)], [3.0, 2.0], shape=shape)  # asymmetric content
    reference = apply_dihedral(raw, name)
    found, r, scores = find_dihedral_match(raw, reference)
    assert r == pytest.approx(1.0)
    assert np.array_equal(apply_dihedral(raw, found), reference)  # equivalent transform even if two share a name-space


def test_dihedral_acts_identically_on_a_cube_and_its_planes():
    from llamas_pyjamas.astrometry_correction import apply_dihedral
    cube = np.random.default_rng(6).normal(size=(4, 20, 26))
    for name in ("transpose", "anti_transpose", "rot90", "flip_x"):
        out = apply_dihedral(cube, name)
        assert all(np.array_equal(out[i], apply_dihedral(cube[i], name)) for i in range(4))


def test_find_dihedral_match_rejects_symmetric_and_unrelated_images():
    from llamas_pyjamas.astrometry_correction import find_dihedral_match
    yy, xx = np.indices((41, 41))
    symmetric = np.exp(-0.5 * ((xx - 20) ** 2 + (yy - 20) ** 2) / 25.0)  # one centred round blob
    with pytest.raises(ValueError, match="ambiguous"):
        find_dihedral_match(symmetric, symmetric)
    noise_a = np.random.default_rng(7).normal(size=(30, 30))
    noise_b = np.random.default_rng(8).normal(size=(30, 30))
    with pytest.raises(ValueError, match="best dihedral match"):
        find_dihedral_match(noise_a, noise_b)


def test_round_trip_residual_is_small():
    wcs = make_wcs()
    ref = blobs(CENTRES, AMPS)
    dx, dy = -2.92, -0.19
    line = shifted_map(dx, dy)
    err = np.full(SHAPE, 0.1)
    m = measure_line_offset(line, wcs, ref, wcs)
    f, e = resample_map_with_offset(line, err, wcs, wcs, SHAPE, m["shift_x_pix"], m["shift_y_pix"])
    valid = np.isfinite(f)
    f0 = np.where(valid, f, 0.0)
    again = measure_line_offset(f0, wcs, ref, wcs)
    assert abs(again["shift_x_pix"]) < 0.05
    assert abs(again["shift_y_pix"]) < 0.05


def test_flux_is_conserved_for_an_interior_source():
    wcs = make_wcs()
    line = shifted_map(0.0, 0.0)
    f, e = resample_map_with_offset(line, np.ones(SHAPE), wcs, wcs, SHAPE, -2.37, 0.61)
    assert np.nansum(f) == pytest.approx(line.sum(), rel=1e-6)


def test_err_is_interpolated_linearly_not_in_quadrature():
    wcs = make_wcs()
    err = np.full(SHAPE, 2.0)
    f, e = resample_map_with_offset(np.ones(SHAPE), err, wcs, wcs, SHAPE, 0.5, 0.5)
    # Quadrature of four 0.25-weighted equal errors would give 1.0; linear gives 2.0.
    assert np.nanmedian(e) == pytest.approx(2.0)


def test_edge_rule_is_nearest_pixel_and_does_not_attenuate():
    """Valid columns are 0..59. new(p) = src(p + shift): the footprint edge follows the ROUNDED shift."""
    wcs = make_wcs()
    img = np.full(SHAPE, 7.0)
    img[:, 60:] = np.nan
    err = np.full(SHAPE, 2.0)
    # shift 0.4: pixel 59 samples 59.4 -> weight 0.6 on valid col 59, 0.4 on NaN col 60 -> kept (0.6 >= 0.5)
    f, e = resample_map_with_offset(img, err, wcs, wcs, SHAPE, 0.4, 0.0)
    assert np.isfinite(f[:, 59]).all() and np.isnan(f[:, 60]).all()
    # ...and renormalised, NOT attenuated to 0.6*7: a constant map stays constant, errors likewise.
    assert np.allclose(f[:, :60], 7.0) and np.allclose(e[:, :60], 2.0)
    # shift 0.6: pixel 59 samples 59.6 -> only 0.4 of the weight is valid -> dropped; pixel 58 kept.
    f2, _ = resample_map_with_offset(img, err, wcs, wcs, SHAPE, 0.6, 0.0)
    assert np.isnan(f2[:, 59]).all() and np.isfinite(f2[:, 58]).all()
    # An integer shift has no partial-weight neighbour: the edge moves by exactly that many pixels.
    f3, _ = resample_map_with_offset(img, err, wcs, wcs, SHAPE, 1.0, 0.0)
    assert np.isfinite(f3[:, 58]).all() and np.isnan(f3[:, 59]).all()


def test_small_footprint_flux_is_not_lost_to_edge_erosion():
    """The regression that motivated the rule: a small strip footprint under a sub-pixel reprojection."""
    wcs = make_wcs()
    img = np.zeros(SHAPE)
    img[30:50, 20:28] = 1.0            # an 8-px-wide strip, like FRB20250316A's
    img[~(img > 0)] = np.nan
    src_wcs = make_wcs(crpix_offset=(0.3, 0.3))  # deterministic ~0.3 px WCS difference
    f, _ = resample_map_with_offset(img, np.ones(SHAPE), src_wcs, wcs, SHAPE, 0.0, 0.0)
    assert np.isfinite(f).sum() >= 0.9 * np.isfinite(img).sum()   # strict NaN lost ~30%+ of a strip this narrow
    assert np.nanmean(f) == pytest.approx(1.0)


def test_out_of_bounds_is_nan():
    wcs = make_wcs()
    f, _ = resample_map_with_offset(np.ones(SHAPE), np.ones(SHAPE), wcs, wcs, SHAPE, 3.0, 0.0)
    assert np.isnan(f[:, -3:]).all()
    assert np.isfinite(f[:, :-4]).all()


def test_wcs_aware_resampling_matches_the_pure_pixel_shift():
    """src grid whose WCS is offset by an integer number of pixels from the reference grid."""
    ref_wcs = make_wcs()
    src_wcs = make_wcs(crpix_offset=(2.0, 0.0))  # world coord at src pixel x == that of ref pixel x-2
    img = blobs(CENTRES, AMPS)
    f, _ = resample_map_with_offset(img, np.ones(SHAPE), src_wcs, ref_wcs, SHAPE, 0.0, 0.0)
    # ref pixel p maps to src pixel p + 2  ->  f[:, p] == img[:, p + 2]
    assert np.allclose(f[:, :-4], img[:, 2:-2], atol=1e-6, equal_nan=False)


def test_identical_wcs_skips_the_wcs_round_trip_exactly():
    wcs = make_wcs()
    img = blobs(CENTRES, AMPS)
    f, _ = resample_map_with_offset(img, np.ones(SHAPE), wcs, wcs, SHAPE, 0.0, 0.0)
    assert np.array_equal(f, img)


def test_cube_matches_map_plane_by_plane():
    wcs = make_wcs()
    nw = 5
    rng = np.random.default_rng(1)
    flux = np.stack([blobs(CENTRES, AMPS) * (i + 1) for i in range(nw)])
    var = rng.uniform(0.5, 1.5, size=flux.shape)
    weight = np.ones(flux.shape)
    fc, vc, wc = resample_cube_with_offset(flux, var, weight, wcs, wcs, SHAPE, -1.4, 0.7)
    for i in range(nw):
        fm, em = resample_map_with_offset(flux[i], np.sqrt(var[i]), wcs, wcs, SHAPE, -1.4, 0.7)
        assert np.allclose(fc[i], fm, equal_nan=True)
        assert np.allclose(vc[i], em ** 2, equal_nan=True)
    assert np.nanmax(np.abs(wc - 1.0)) < 1e-9


def _linemaps_file(path, wcs):
    """A minimal linemaps-style file: shifted-candidate lines, reference lines, and extras that must be dropped."""
    hdus = [fits.PrimaryHDU()]
    rng = np.random.default_rng(0)
    for name in ["HBETA_FLUX", "HBETA_ERR", "OIII_5007_FLUX", "OIII_5007_ERR", "OIII_4959_FLUX", "OIII_4959_ERR",
                 "HALPHA_FLUX", "HALPHA_ERR", "NII_6584_FLUX", "NII_6584_ERR", "NII_6548_FLUX", "NII_6548_ERR",
                 "OII_3727_FLUX", "SII_6716_FLUX", "GREEN_HBETA_OIII_SIGMA", "CONTINUUM_IMG", "CONTINUUM_IMG_ERR",
                 "METALLICITY"]:
        h = wcs.to_header()
        h["WCSFLIP"] = "None"
        h["WCSROT"] = 0
        h["BUNIT"] = "1e-17 erg s-1 cm-2"
        data = blobs(CENTRES, AMPS) if name.endswith("FLUX") else rng.uniform(0.5, 1.0, SHAPE)
        hdus.append(fits.ImageHDU(data.astype(np.float32), header=h, name=name))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_linemaps_writer_keeps_only_the_two_complexes(tmp_path):
    wcs = make_wcs()
    src, out = tmp_path / "src.fits", tmp_path / "src_offsetcorr.fits"
    _linemaps_file(src, wcs)
    written = write_offset_corrected_linemaps(
        src, out, -2.9, -0.2, wcs, SHAPE,
        shifted_lines=("HBETA", "OIII_4959", "OIII_5007"),
        reference_lines=("HALPHA", "NII_6548", "NII_6584"),
        passthrough=("CONTINUUM_IMG", "CONTINUUM_IMG_ERR"),
        d_east_arcsec=0.87, d_north_arcsec=0.05)
    expected = {f"{ln}_{s}" for ln in ("HBETA", "OIII_4959", "OIII_5007", "HALPHA", "NII_6548", "NII_6584")
                for s in ("FLUX", "ERR")} | {"CONTINUUM_IMG", "CONTINUUM_IMG_ERR"}
    assert set(written) == expected
    with fits.open(out) as new, fits.open(src) as old:
        assert {h.name for h in new[1:]} == expected
        for gone in ("OII_3727_FLUX", "SII_6716_FLUX", "GREEN_HBETA_OIII_SIGMA", "METALLICITY"):
            assert gone not in new
        # Unshifted extensions: data byte-identical to the source.
        for name in ("HALPHA_FLUX", "NII_6584_ERR", "CONTINUUM_IMG"):
            assert np.array_equal(new[name].data, old[name].data)
            assert new[name].header["OFFCORR"] is False
        # Shifted ones differ, carry provenance, and keep the orientation/unit cards consumers read.
        assert not np.array_equal(new["HBETA_FLUX"].data, old["HBETA_FLUX"].data)
        h = new["HBETA_FLUX"].header
        assert h["OFFCORR"] is True and h["OFFDXPIX"] == pytest.approx(-2.9)
        assert h["OFFDE_AS"] == pytest.approx(0.87) and h["OFFSRC"] == "src.fits"
        assert h["WCSFLIP"] == "None" and h["WCSROT"] == 0 and h["BUNIT"].startswith("1e-17")
        assert new[0].header["OFFLINES"] == "HBETA,OIII_4959,OIII_5007"


def test_linemaps_writer_stamps_extra_cards_and_zero_shift_is_an_identity_copy(tmp_path):
    """A measured-but-not-applied offset is recorded in the headers, and applying 0 changes no data."""
    wcs = make_wcs()
    src, out = tmp_path / "src.fits", tmp_path / "src_offsetcorr.fits"
    _linemaps_file(src, wcs)
    cards = {"OFFAPPLY": (False, "Measured offset applied?"), "OFFMEAE": (0.31, "measured E"),
             "OFFWHY": ("not significant (1.6 sigma < 5)", "why")}
    write_offset_corrected_linemaps(src, out, 0.0, 0.0, wcs, SHAPE,
                                    shifted_lines=("HBETA", "OIII_5007"), reference_lines=("HALPHA",),
                                    passthrough=("CONTINUUM_IMG",), d_east_arcsec=0.0, d_north_arcsec=0.0,
                                    extra_cards=cards)
    with fits.open(out) as new, fits.open(src) as old:
        for name in ("HBETA_FLUX", "HBETA_ERR", "HALPHA_FLUX", "CONTINUUM_IMG"):
            h = new[name].header
            assert h["OFFAPPLY"] is False and h["OFFMEAE"] == pytest.approx(0.31) and h["OFFWHY"].startswith("not sig")
            assert np.array_equal(new[name].data, old[name].data.astype(np.float64), equal_nan=True)
        assert new[0].header["OFFAPPLY"] is False and new[0].header["OFFLINES"] == "HBETA,OIII_5007"


def test_linemaps_writer_skips_absent_names(tmp_path):
    """FRB20250316A's blue file has only Hbeta/[OIII] -- reference/passthrough names must be skipped silently."""
    wcs = make_wcs()
    src, out = tmp_path / "blue.fits", tmp_path / "blue_offsetcorr.fits"
    hdus = [fits.PrimaryHDU()]
    for name in ["HBETA_FLUX", "HBETA_ERR", "OIII_5007_FLUX", "OIII_5007_ERR", "OIII_4959_FLUX", "OIII_4959_ERR"]:
        hdus.append(fits.ImageHDU(blobs(CENTRES, AMPS).astype(np.float32), header=wcs.to_header(), name=name))
    fits.HDUList(hdus).writeto(src)
    written = write_offset_corrected_linemaps(src, out, 1.0, 0.5, wcs, SHAPE,
                                              shifted_lines=("HBETA", "OIII_4959", "OIII_5007"),
                                              reference_lines=("HALPHA",), passthrough=())
    assert len(written) == 6


def test_slab_file_round_trip(tmp_path):
    wcs = make_wcs()
    rng = np.random.default_rng(2)
    slabs = []
    for name, w0, shifted in (("HB_OIII", 4915.0, True), ("HA_NII", 6696.0, False)):
        wave = w0 + 1.2 * np.arange(12)
        slabs.append(dict(name=name, wave=wave, flux=rng.normal(size=(12,) + SHAPE),
                          var=rng.uniform(1, 2, size=(12,) + SHAPE), weight=np.ones((12,) + SHAPE),
                          wcs2d=wcs, bunit="1e-17 erg s-1 cm-2 Angstrom-1", shifted=shifted))
    path = tmp_path / "slabs.fits"
    write_offset_cube_slabs(slabs, path, {"target": "FRBTEST", "z": 0.03, "offdxpix": -2.9})
    back = read_offset_cube_slabs(path)
    assert back["meta"]["TARGET"] == "FRBTEST"
    for s in slabs:
        b = back[s["name"]]
        assert np.array_equal(b["wave"], s["wave"])
        assert np.allclose(b["flux"], s["flux"].astype(np.float32))
        assert b["shifted"] is s["shifted"]
        assert b["bunit"].startswith("1e-17")
        # Same geometry, though written as a CD matrix rather than PC+CDELT.
        assert np.allclose(b["wcs2d"].pixel_scale_matrix, wcs.pixel_scale_matrix, rtol=1e-9, atol=0)
        assert np.allclose(b["wcs2d"].wcs.crpix, wcs.wcs.crpix) and np.allclose(b["wcs2d"].wcs.crval, wcs.wcs.crval)


def test_equivalent_wcs_representations_count_as_the_same_grid():
    from llamas_pyjamas.astrometry_correction import _same_celestial_grid
    a = make_wcs()
    b = WCS(naxis=2)
    b.wcs.ctype, b.wcs.crpix, b.wcs.crval = list(a.wcs.ctype), list(a.wcs.crpix), list(a.wcs.crval)
    b.wcs.cd = a.pixel_scale_matrix  # same geometry, CD-matrix representation
    assert _same_celestial_grid(a, b, SHAPE, SHAPE)
    assert not _same_celestial_grid(a, make_wcs(crpix_offset=(0.5, 0.0)), SHAPE, SHAPE)
    assert not _same_celestial_grid(a, a, SHAPE, (SHAPE[0] + 1, SHAPE[1]))
