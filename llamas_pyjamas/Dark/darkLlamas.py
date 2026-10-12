"""
This module provides functionality to create a master dark-current-rate frame
from a directory of raw dark images or a list of files, and to apply that
correction to a target frame. Modeled directly on llamas_pyjamas.Bias.llamasBias.

Classes:
    DarkLlamas: A class to handle dark-frame processing and creation of a
        master dark-current-rate frame.

Functions:
    apply_dark_correction: Subtract a scaled master-dark rate map from a
        single detector's data array.
    apply_dark_to_hdulist: Apply apply_dark_correction across every matching
        extension of a science HDUList.

Usage:
    dark_llamas = DarkLlamas('/path/to/dark/images')
    master_dark_path = dark_llamas.master_dark('/path/to/slow_master_bias.fits')
"""
import os
import numpy as np
from astropy.io import fits

from llamas_pyjamas.config import CALIB_DIR
from llamas_pyjamas.Dark import DarkNotFoundError, DarkReadModeError
from llamas_pyjamas.Bias.biasChecking import _match_bias_hdu as _match_hdu


class DarkLlamas:
    def __init__(self, input_data) -> None:
        """
        Initializes the DarkLlamas object with either a directory path, a file
        path, or a list of file paths.

        Parameters
        ----------
        input_data : str or list
            If a string, it is treated as either a directory (from which all
            .fits files are grabbed) or a full file path. If a list, it
            should contain file paths to .fits files.
        """
        from llamas_pyjamas.config import CALIB_DIR  # ensure CALIB_DIR is available

        # Case when a list is provided
        if isinstance(input_data, list):
            # Keep only files ending with '.fits'
            self.files = [file for file in input_data if str(file).endswith('.fits')]
            if not self.files:
                raise ValueError("No .fits files found in the provided list.")
            # Use the directory of the first file if available, else default to CALIB_DIR
            self.dark_path = os.path.dirname(self.files[0]) if os.path.dirname(self.files[0]) else CALIB_DIR

        # Case when a string is provided
        elif isinstance(input_data, str):
            # If it's an existing file, use it directly
            if os.path.isfile(input_data):
                self.files = [input_data]
                self.dark_path = os.path.dirname(input_data)
            # If it's an existing directory, search for .fits files
            elif os.path.isdir(input_data):
                self.dark_path = input_data
                self.files = [os.path.join(input_data, f) for f in os.listdir(input_data) if f.endswith('.fits')]
            else:
                # If the string is just a filename without path, assume it resides in CALIB_DIR
                potential = os.path.join(CALIB_DIR, input_data)
                if os.path.isfile(potential):
                    self.files = [potential]
                    self.dark_path = os.path.dirname(potential)
                else:
                    raise ValueError("Provided input_data does not exist as a file or a directory.")
            if not self.files:
                raise ValueError("No .fits files found in the provided input_data.")

        else:
            raise TypeError("input_data must be a string (file or directory path) or a list of file paths.")

    def master_dark(self, master_bias_file: str) -> str:
        """
        Create a master dark-current-rate frame from a list of raw dark images.

        Each raw dark FITS file is assumed to have a primary HDU (carrying the
        exposure time in the 'REXPTIME' keyword, and optionally 'READ-MDE')
        and one or more extensions, where each extension contains a dark
        image from a detector, identified by the 'COLOR', 'BENCH', and 'SIDE'
        header keywords.

        For each file and each extension:
          1. The matching detector extension is located in ``master_bias_file``
             and subtracted from the raw dark data.
          2. The bias-subtracted data is divided by the frame's REXPTIME to
             produce a per-second dark-current-rate map (DN/s).
        All per-second maps for a given detector (across all input files) are
        then median-combined (via nanmedian) into that detector's master dark
        rate map.

        The resulting FITS file mirrors the structure produced by
        BiasLlamas.master_bias(): a primary HDU (using the primary header from
        the first dark file) plus one ImageHDU per detector, and is written to
        'combined_dark.fits' in self.dark_path.

        Parameters
        ----------
        master_bias_file : str
            Path to a combined master bias FITS file (e.g. slow_master_bias.fits
            or fast_master_bias.fits) covering the same read-mode as the raw
            dark frames.

        Returns
        -------
        str
            The path to the written master dark FITS file.

        Raises
        ------
        DarkNotFoundError
            If master_bias_file cannot be opened, or no matching bias
            extension can be found for a given dark extension.
        DarkReadModeError
            If a dark file's READ-MDE does not match the master bias's
            READ-MDE.
        KeyError
            If a dark extension is missing COLOR/BENCH/SIDE headers.
        """
        try:
            bias_hdul = fits.open(master_bias_file)
        except (OSError, FileNotFoundError) as e:
            raise DarkNotFoundError(master_bias_file) from e

        try:
            bias_readmode = bias_hdul[0].header.get('READ-MDE', None)
            if bias_readmode is not None:
                bias_readmode = str(bias_readmode).strip().upper()

            # Open the first file to get the primary header as reference
            first_file = os.path.join(self.dark_path, self.files[0]) if not os.path.isabs(self.files[0]) else self.files[0]
            with fits.open(first_file) as hdulist_first:
                primary_hdr = hdulist_first[0].header.copy()

            combined_hdus = []
            # Create primary HDU using the primary header from the first file
            primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            combined_hdus.append(primary_hdu)

            # Dictionary to group per-second dark-rate maps by (COLOR, BENCH, SIDE)
            groups = {}

            # Loop through each dark file
            for file in self.files:
                file_path = os.path.join(self.dark_path, file) if not os.path.isabs(file) else file
                with fits.open(file_path) as hdul:
                    file_readmode = hdul[0].header.get('READ-MDE', None)
                    if file_readmode is not None and bias_readmode is not None:
                        file_readmode = str(file_readmode).strip().upper()
                        if file_readmode != bias_readmode:
                            raise DarkReadModeError(file_readmode, bias_readmode, master_bias_file)

                    try:
                        rexptime = float(hdul[0].header['REXPTIME'])
                    except KeyError as e:
                        raise KeyError(f"File {file} missing required primary header key: {e}")

                    # Loop through each extension (skip primary, i.e. index 0)
                    for ext in range(1, len(hdul)):
                        hdr = hdul[ext].header.copy()
                        try:
                            color, bench, side = hdr["COLOR"], hdr["BENCH"], hdr["SIDE"]
                        except KeyError as e:
                            raise KeyError(f"Extension in file {file} missing required key: {e}")

                        bias_hdu = _match_hdu(bias_hdul, bench, side, color)
                        if bias_hdu is None:
                            raise DarkNotFoundError(
                                master_bias_file,
                                message=(
                                    f"No matching bias extension for detector "
                                    f"(COLOR={color}, BENCH={bench}, SIDE={side}) "
                                    f"in '{master_bias_file}'."
                                ),
                            )

                        rate = (hdul[ext].data.astype(float) - bias_hdu.data.astype(float)) / rexptime

                        key = (color, bench, side)
                        if key not in groups:
                            groups[key] = {"data": [], "header": hdr}
                        groups[key]["data"].append(rate)

            # Process each group by computing the median-combined dark rate
            for (color, bench, side), group in groups.items():
                data_array = np.array(group["data"])
                combined_data = np.nanmedian(data_array, axis=0)
                combined_header = group["header"]
                combined_header["COMBINED"] = True
                combined_header["COLOR"] = color
                combined_header["BENCH"] = bench
                combined_header["SIDE"] = side
                combined_header["DARKUNIT"] = ('DN/s', 'Units of this master dark (rate, not counts)')
                combined_header["NDARKFRM"] = (len(group["data"]), 'Number of raw dark frames combined')
                image_hdu = fits.ImageHDU(data=combined_data, header=combined_header, name=f"COL_{color}_BNCH_{bench}")
                combined_hdus.append(image_hdu)

            hdul_out = fits.HDUList(combined_hdus)
            out_filename = os.path.join(self.dark_path, 'combined_dark.fits')
            hdul_out.writeto(out_filename, overwrite=True)

            return out_filename
        finally:
            bias_hdul.close()


def apply_dark_correction(data: np.ndarray, exptime: float, dark_hdu) -> np.ndarray:
    """
    Subtract a scaled master-dark rate map from a single detector's data.

    Parameters
    ----------
    data : numpy.ndarray
        The raw (or bias-subtracted) 2-D detector data.
    exptime : float
        Exposure time (seconds) of the target frame.
    dark_hdu : astropy.io.fits.ImageHDU
        The matching detector extension of a master dark FITS produced by
        DarkLlamas.master_dark() (data in DN/s).

    Returns
    -------
    numpy.ndarray
        ``data - dark_hdu.data * exptime``.
    """
    return data.astype(float) - dark_hdu.data.astype(float) * exptime


def apply_dark_to_hdulist(science_hdul, master_dark_hdul, exptime: float = None):
    """
    Apply apply_dark_correction across every detector extension of a science
    HDUList, matching each extension to the corresponding master dark
    extension by (COLOR, BENCH, SIDE).

    Parameters
    ----------
    science_hdul : astropy.io.fits.HDUList
        The target frame to correct. Modified in place.
    master_dark_hdul : astropy.io.fits.HDUList
        A master dark FITS as produced by DarkLlamas.master_dark().
    exptime : float, optional
        Exposure time (seconds) of the target frame. If not provided, read
        from ``science_hdul[0].header['REXPTIME']``.

    Returns
    -------
    astropy.io.fits.HDUList
        The same ``science_hdul`` object, with each detector extension's
        data dark-subtracted and header keywords DARKSUB/DARKLVL/DARKEXP set.

    Raises
    ------
    DarkNotFoundError
        If no matching master-dark extension is found for a science extension.
    """
    if exptime is None:
        exptime = float(science_hdul[0].header['REXPTIME'])

    for ext in range(1, len(science_hdul)):
        hdr = science_hdul[ext].header
        try:
            color, bench, side = hdr["COLOR"], hdr["BENCH"], hdr["SIDE"]
        except KeyError as e:
            raise KeyError(f"Science extension {ext} missing required key: {e}")

        dark_hdu = _match_hdu(master_dark_hdul, bench, side, color)
        if dark_hdu is None:
            raise DarkNotFoundError(
                message=(
                    f"No matching master-dark extension for detector "
                    f"(COLOR={color}, BENCH={bench}, SIDE={side})."
                ),
                path='<master_dark_hdul>',
            )

        corrected = apply_dark_correction(science_hdul[ext].data, exptime, dark_hdu)
        science_hdul[ext].data = corrected
        hdr['DARKSUB'] = (True, 'True = master dark was subtracted')
        hdr['DARKLVL'] = (float(np.nanmedian(dark_hdu.data.astype(float) * exptime)), 'Median dark level subtracted (DN)')
        hdr['DARKEXP'] = (exptime, 'Exposure time (s) used to scale the master dark')

    return science_hdul
