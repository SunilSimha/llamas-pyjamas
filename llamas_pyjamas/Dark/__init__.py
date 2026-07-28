"""
llamas_pyjamas.Dark
====================
Dark-current calibration package for LLAMAS IFU pipeline.

Public API
----------
DarkNotFoundError  -- raised when a required dark/bias extension cannot be
                       located or a master dark FITS file cannot be read.
DarkReadModeError  -- raised when dark read-mode does not match the master
                       bias (or target frame) it is being combined/applied with.

The DarkLlamas class (master dark-current-rate stacking) and the
apply_dark_correction / apply_dark_to_hdulist functions live in
darkLlamas.py and are not re-exported here, to avoid circular imports with
astropy during Ray worker init (same rationale as llamas_pyjamas.Bias).
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DarkNotFoundError(FileNotFoundError):
    """
    Raised when a master-dark FITS file is absent/unreadable, or when no
    dark extension matching a requested (bench, side, color) can be found.

    Attributes
    ----------
    path : str
        The file path that could not be opened, or the master-dark path
        being searched when a matching extension was not found.
    """
    def __init__(self, path: str, message: str = None):
        self.path = path
        if message is None:
            message = (
                f"Master dark not found or could not be read: '{path}'. "
                f"Check that the file exists and is a valid 24-extension LLAMAS dark FITS."
            )
        super().__init__(message)


class DarkReadModeError(ValueError):
    """
    Raised when the read-mode encoded in a raw dark frame's primary header
    does not match the read-mode of the master bias being subtracted from
    it (during master_dark()), or the read-mode of the master dark does not
    match the target frame it is being applied to.

    Attributes
    ----------
    requested_mode : str
    dark_mode      : str
    dark_path      : str
    """
    def __init__(self, requested_mode: str, dark_mode: str, dark_path: str):
        self.requested_mode = requested_mode
        self.dark_mode = dark_mode
        self.dark_path = dark_path
        super().__init__(
            f"Read-mode mismatch: frame needs '{requested_mode}' "
            f"but dark file '{dark_path}' has READ-MDE='{dark_mode}'."
        )
