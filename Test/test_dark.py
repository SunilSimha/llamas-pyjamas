"""Tests for the Dark module (llamas_pyjamas.Dark).

Builds tiny in-memory MEF frames (matching the (COLOR, BENCH, SIDE)-keyed
convention used by the Bias module) so no external FITS fixtures are
required.
"""
import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Dark import DarkReadModeError
from llamas_pyjamas.Dark.darkLlamas import (
    DarkLlamas,
    apply_dark_correction,
    apply_dark_to_hdulist,
)

SHAPE = (20, 20)
DETECTORS = [("Green", 1, "A"), ("Red", 2, "B")]
BIAS_LEVEL = {"Green": 500.0, "Red": 600.0}
DARK_RATE = {"Green": 2.0, "Red": 3.5}  # DN/s, ground truth injected into fixtures


def make_bias(tmp_path, name="master_bias.fits", read_mode="SLOW"):
    ph = fits.PrimaryHDU()
    ph.header["READ-MDE"] = read_mode
    hdus = [ph]
    for color, bench, side in DETECTORS:
        ext = fits.ImageHDU(data=np.full(SHAPE, BIAS_LEVEL[color], dtype=np.float32))
        ext.header["COLOR"] = color
        ext.header["BENCH"] = bench
        ext.header["SIDE"] = side
        hdus.append(ext)
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


def make_dark(tmp_path, name, *, rexptime, read_mode="SLOW"):
    ph = fits.PrimaryHDU()
    ph.header["READ-MDE"] = read_mode
    ph.header["REXPTIME"] = rexptime
    hdus = [ph]
    for color, bench, side in DETECTORS:
        level = BIAS_LEVEL[color] + DARK_RATE[color] * rexptime
        ext = fits.ImageHDU(data=np.full(SHAPE, level, dtype=np.float32))
        ext.header["COLOR"] = color
        ext.header["BENCH"] = bench
        ext.header["SIDE"] = side
        hdus.append(ext)
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


def test_master_dark_recovers_injected_rate(tmp_path):
    bias_path = make_bias(tmp_path)
    dark_paths = [
        make_dark(tmp_path, "dark_5s.fits", rexptime=5.0),
        make_dark(tmp_path, "dark_50s.fits", rexptime=50.0),
        make_dark(tmp_path, "dark_100s.fits", rexptime=100.0),
    ]

    dark_llamas = DarkLlamas(dark_paths)
    out_path = dark_llamas.master_dark(bias_path)

    with fits.open(out_path) as hdul:
        assert len(hdul) == 1 + len(DETECTORS)
        for color, bench, side in DETECTORS:
            match = None
            for hdu in hdul[1:]:
                if (hdu.header["COLOR"] == color and hdu.header["BENCH"] == bench
                        and hdu.header["SIDE"] == side):
                    match = hdu
                    break
            assert match is not None
            assert match.header["COMBINED"] is True
            assert match.header["DARKUNIT"] == "DN/s"
            assert match.header["NDARKFRM"] == len(dark_paths)
            assert np.allclose(match.data, DARK_RATE[color], atol=1e-6)


def test_master_dark_readmode_mismatch_raises(tmp_path):
    bias_path = make_bias(tmp_path, read_mode="SLOW")
    dark_paths = [make_dark(tmp_path, "dark_fast.fits", rexptime=5.0, read_mode="FAST")]

    dark_llamas = DarkLlamas(dark_paths)
    with pytest.raises(DarkReadModeError):
        dark_llamas.master_dark(bias_path)


def test_master_dark_missing_keys_raises(tmp_path):
    bias_path = make_bias(tmp_path)

    ph = fits.PrimaryHDU()
    ph.header["READ-MDE"] = "SLOW"
    ph.header["REXPTIME"] = 5.0
    ext = fits.ImageHDU(data=np.zeros(SHAPE, dtype=np.float32))
    # Deliberately omit COLOR/BENCH/SIDE
    dark_path = tmp_path / "bad_dark.fits"
    fits.HDUList([ph, ext]).writeto(dark_path, overwrite=True)

    dark_llamas = DarkLlamas([str(dark_path)])
    with pytest.raises(KeyError):
        dark_llamas.master_dark(bias_path)


def test_apply_dark_correction_arithmetic():
    dark_hdu = fits.ImageHDU(data=np.full(SHAPE, 2.0, dtype=np.float32))
    data = np.full(SHAPE, 520.0, dtype=np.float32)
    corrected = apply_dark_correction(data, exptime=10.0, dark_hdu=dark_hdu)
    assert np.allclose(corrected, 500.0)


def test_apply_dark_to_hdulist(tmp_path):
    bias_path = make_bias(tmp_path)
    dark_paths = [
        make_dark(tmp_path, "dark_5s.fits", rexptime=5.0),
        make_dark(tmp_path, "dark_50s.fits", rexptime=50.0),
    ]
    dark_llamas = DarkLlamas(dark_paths)
    master_dark_path = dark_llamas.master_dark(bias_path)

    exptime = 30.0
    ph = fits.PrimaryHDU()
    ph.header["REXPTIME"] = exptime
    hdus = [ph]
    for color, bench, side in DETECTORS:
        level = BIAS_LEVEL[color] + DARK_RATE[color] * exptime
        ext = fits.ImageHDU(data=np.full(SHAPE, level, dtype=np.float32))
        ext.header["COLOR"] = color
        ext.header["BENCH"] = bench
        ext.header["SIDE"] = side
        hdus.append(ext)
    science_hdul = fits.HDUList(hdus)

    with fits.open(master_dark_path) as master_dark_hdul:
        apply_dark_to_hdulist(science_hdul, master_dark_hdul)

    for i, (color, bench, side) in enumerate(DETECTORS, start=1):
        assert np.allclose(science_hdul[i].data, BIAS_LEVEL[color], atol=1e-3)
        assert science_hdul[i].header["DARKSUB"] is True
        assert science_hdul[i].header["DARKEXP"] == exptime
