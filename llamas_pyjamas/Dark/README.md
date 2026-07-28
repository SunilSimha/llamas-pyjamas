# Dark Module

This module handles dark-current frame processing and correction for the LLAMAS instrument.

## Core Functionality

The dark module performs:
- Reading and validation of raw dark frames
- Bias-subtracting and exposure-time-normalizing raw darks into per-second rate maps
- Median combination of those maps into a master dark-current-rate frame
- Dark-current correction of target frames (master dark scaled by exposure time)

## Key Files

### `darkLlamas.py`
Main dark processing module containing:
- `DarkLlamas` class: reads a directory/file/list of raw dark FITS, and via
  `master_dark(master_bias_file)` bias-subtracts each frame, divides by its
  `REXPTIME`, and median-combines the results per detector into a master
  dark-current-rate FITS (DN/s), in the same structure as `combined_bias.fits`.
- `apply_dark_correction(data, exptime, dark_hdu)`: subtracts
  `dark_hdu.data * exptime` from a single detector's data.
- `apply_dark_to_hdulist(science_hdul, master_dark_hdul, exptime=None)`:
  applies `apply_dark_correction` across every matching detector extension
  of a science HDUList.

## Usage

```python
from llamas_pyjamas.Dark.darkLlamas import DarkLlamas, apply_dark_to_hdulist
from astropy.io import fits

# Build a master dark-current-rate frame from raw darks (same read-mode as the bias)
dark = DarkLlamas(dark_files)
master_dark_path = dark.master_dark('/path/to/slow_master_bias.fits')

# Apply dark correction to a bias-subtracted target frame
science_hdul = fits.open('science.fits', mode='update')
master_dark_hdul = fits.open(master_dark_path)
apply_dark_to_hdulist(science_hdul, master_dark_hdul)
science_hdul.flush()
```

The dark module is typically run after bias correction, using the same
`(COLOR, BENCH, SIDE)`-keyed extension matching as the Bias module.
