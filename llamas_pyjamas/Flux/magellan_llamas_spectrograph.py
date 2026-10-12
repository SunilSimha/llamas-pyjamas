"""
Minimal PypeIt Spectrograph subclass for Magellan/LLAMAS.

LLAMAS has no spectrograph class in the installed PypeIt package (it is a
plain PyPI dependency here, not an instrument-aware fork), so
``load_spectrograph('magellan_llamas')`` fails outright. This class exists
only to satisfy the parts of PypeIt's flux-calibration machinery
(``SensFunc``, ``get_sensfunc_factor``) that fluxcal_cubes.py drives -- it
does not support running PypeIt's own raw-frame reduction on LLAMAS data.

Importing this module registers the class with PypeIt: spectrograph lookup
(``pypeit.spectrographs.util.load_spectrograph`` /
``pypeit.spectrographs.spectrograph_classes``) walks all currently-imported
subclasses of ``Spectrograph`` at call time rather than reading a static
registry, so no changes to the installed PypeIt package are needed.
"""

from pypeit import telescopes
from pypeit.spectrographs import spectrograph


class MagellanLLAMASSpectrograph(spectrograph.Spectrograph):
    """Child to handle Magellan/LLAMAS specific code."""

    name = 'magellan_llamas'
    ndet = 1
    telescope = telescopes.MagellanTelescopePar()
    camera = 'LLAMAS'
    header_name = 'LLAMAS'
    pypeline = 'SlicerIFU'

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        Cards verified against a real raw frame
        (science/LLAMAS_2025-12-17_06-17-02.7_SCI22_mef.fits) and a real
        per-channel RSS file -- both share the same primary-header lineage.
        """
        self.meta = {}
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJECT', default='unknown')
        # CHANNEL ('red'/'green'/'blue') exists on per-channel RSS/cube
        # files; the raw multi-extension frame has no single top-level
        # value since it interleaves all three, but fluxcal_cubes.py only
        # ever points this spectrograph at per-channel files.
        self.meta['dispname'] = dict(ext=0, card='CHANNEL', default='default')
        self.meta['decker'] = dict(ext=0, card=None, default='default')
        self.meta['binning'] = dict(ext=0, card=None, default='1,1')
        # No plain MJD/EXPTIME cards exist anywhere in the header chain --
        # only MJD-OBS and REXPTIME (requested exptime; SEXPTIME/CEXPTIME
        # are never populated).
        self.meta['mjd'] = dict(ext=0, card='MJD-OBS')
        self.meta['exptime'] = dict(ext=0, card='REXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')

    def configuration_keys(self):
        return ['dispname']

    def raw_header_cards(self):
        return []

    @classmethod
    def default_pypeit_par(cls):
        par = super().default_pypeit_par()
        par['rdx']['spectrograph'] = cls.name
        # R10000 is the lowest-resolution telluric PCA grid PypeIt ships;
        # correct choice for LLAMAS's ~3000 resolving power, which is below
        # every available grid's native R (same choice keck_lris.py makes
        # for its comparable resolution).
        par['sensfunc']['IR']['telgridfile'] = 'TellPCA_3000_26000_R10000.fits'
        return par
