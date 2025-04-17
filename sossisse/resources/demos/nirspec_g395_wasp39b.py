#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2025-03-25 at 09:57

@author: cook
"""
from sossisse.core import constants

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'soss_trappist1b.py'

# copy of CDict
Cdict = constants.CDict.copy(source=__NAME__)

# =============================================================================
# Define info (must contain "title")
# =============================================================================
INFO = dict()
INFO['title'] = 'JWST.NIRSPEC.G395 WASP 39b transit'
INFO['object'] = 'WASP 39b'
INFO['instrument'] = 'JWST.NIRSPEC.G395'
INFO['description'] = 'WASP 39b transit observed XXX'
INFO['date'] = '2025-03-25'
INFO['author'] = 'Loic Albert'
# url for data download
URL = 'https://www.astro.umontreal.ca/~cook/pogo-data/'
# download: key = file, value = local path
DOWNLOAD = dict()
DOWNLOAD['GENERAL.FILES'] = 'PATHS.RAWPATH'
DOWNLOAD['GENERAL.BKGFILE'] = 'PATHS.CALIBPATH'
DOWNLOAD['GENERAL.FLATFILE'] = 'PATHS.CALIBPATH'
DOWNLOAD['GENERAL.POS_FILE'] = 'PATHS.CALIBPATH'

# =============================================================================
# Switches to turn large parts of the code on/off
# =============================================================================
cgroup = 'RUN'

# Switch for turning on/off the linear reconstruction step
Cdict.set('LINEAR_RECON', group=cgroup, source=__NAME__,
          value=True)

# Switch for turning on/off the spectral extraction
Cdict.set('SPECTRAL_EXTRACTION', group=cgroup, source=__NAME__,
          value=True)

# =============================================================================
# Definition of inputs related to the data
# =============================================================================
cgroup = 'INPUTS'

# Instrument mode
Cdict.set('INSTRUMENTMODE', group=cgroup, source=__NAME__,
          value='JWST.NIRISS.G395')

# Name of the object (must match the object directory name)
Cdict.set('OBJECTNAME', group=cgroup, source=__NAME__,
          value='Wasp39b')

# =============================================================================
# Definition of general inputs
# =============================================================================
cgroup = 'GENERAL'

# List of files to be processed in form:
# FILES:
#  - file1
#  - file2
#  - file3
Cdict.set('FILES', group=cgroup, source=__NAME__,
          value=['jw01366003001_04101_00001-seg001_nrs2_rateints.fits',
                 'jw01366003001_04101_00001-seg002_nrs2_rateints.fits',
                 'jw01366003001_04101_00001-seg003_nrs2_rateints.fits'])

# background file --> leave as None if the is no background available for
#    the mode - if empty DO_BACKGROUND is set to False regardless of the value
Cdict.set('BKGFILE', group=cgroup, source=__NAME__,
          value=None)

# do background correction - must have BKGFILE defined to do this
Cdict.set('DO_BACKGROUND', value=False, group=cgroup, source=__NAME__)

# flat field file --> leave as '' if there is no flat field available
#     for the mode
Cdict.set('FLATFILE', group=cgroup, source=__NAME__,
          value=None)

# trace position file
Cdict.set('POS_FILE', group=cgroup, source=__NAME__,
          value=None)

# Wavelength calibration file
Cdict.set('WAVE_FILE', group=cgroup, source=__NAME__,
          value='jw01366003001_04101_00001-seg001_nrs2_x1dints.fits')

# Wavelength calibration file type
Cdict.set('WAVE_FILE_TYPE', group=cgroup, source=__NAME__,
          value='ext1d')


# =============================================================================
# Definition of inputs related to the position within data cube timeseries
# =============================================================================
cgroup = 'WLC.INPUTS'

# Integrations used to construct a model PSF.
#     This should be a list of [start, end] lists
#     Ideally, these would be out-of-transit (if any) and without flares.
#     If you observed constantly a variable object (brown dwarf, phase curve),
#     you should enter the first to last as frame IDs, except significant
#     flares.
#     Note that on a first pass of data reduction, you may not know the
#     optimal parameters. You can enter a guess and fine-tune in a second step.
#     As a starting point, BASELINE_INTS can be [0, NINTS-1] where NINTS can
#     be found in your fits header.
#     Note the 1st frame is zero (pythonic numbering)
#     Note that leaving this blank sets the baseline from 0 to NINTS-1
#     E.g.
#         BASELINE_INTS:
#               - [0, 89]
#               - [111, 230]
Cdict.set('BASELINE_INTS', group=cgroup, source=__NAME__,
          value=None)

# Whether there is a transit/eclipse in the data.
# If this is set to False TRANSIT_INTS is ignored
# If this is set to True and TRANSIT_INTS is None a graph will help you
#    decide where the transit should be
# If this is set to True and TRANSIT_INTS is set the code will process transits
#    and treat the data as such
Cdict.set('HAS_TRANSIT', group=cgroup, source=__NAME__,
          value=True)


# If there are transit(s)/eclipse(s) in the data, enter the frames
#     corresponding to either the 1st, 2nd, 3rd and 4th contact [length 4 list]
#     or the 1st to 4th contact [length 2 list].
#     These should not overlap with the BASELINE_INTS,
#     but there may be some domain that is neither because of flares
#     (i.e., not within a transit but not within baseline).
#     You can enter a guess and fine-tune in a second step.
#     Note the 1st frame is zero (pythonic numbering)
#     Note that leaving this blank assumes there is no transit/eclipse
#     e.g. TRANSIT_INTS:
#           - [90, 97, 103, 110]
Cdict.set('TRANSIT_INTS', group=cgroup, source=__NAME__,
          value = [[160, 200, 300, 340]])


# =============================================================================
# Definition of inputs related to handling of the data within each frame
# =============================================================================
cgroup = 'WLC.GENERAL'

# Use fancy centering of the trace (sets RECENTER_TRACE_POSITION to False)
Cdict.set('USE_FANCY_CENTERING', group=cgroup, source=__NAME__,
          value=False)

# Trace extraction width. Set to 0 to use the full image
Cdict.set('TRACE_WIDTH_EXTRACTION', group=cgroup, source=__NAME__,
          value=7)

# define the width for masking the white light curve trace
#         Default
Cdict.set('TRACE_WIDTH_MASKING', group=cgroup, source=__NAME__,
          value=5)


# =============================================================================
# Definition for white light curve
# =============================================================================
cgroup = 'SPEC_EXT'

# Define the resolution to bin to
Cdict.set('RESOLUTION_BIN', group=cgroup, source=__NAME__,
          value=5)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
