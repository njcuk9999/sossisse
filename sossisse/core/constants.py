#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant definitions

Created on 2024-08-13

@author: cook
"""
from sossisse.core import base
from sossisse.core import base_classes


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.constants'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Get constants class
Const = base_classes.Const
# Define the title for the yaml
TITLE = f"""
#############################################################################
  SOSSICE parameter definitions
##############################################################################

    Version = {__version__}
    Date    = {__date__}

If using a different version it is recommended to run "run_setup" to generate
a new yaml file.

"""

# =============================================================================
# Define constants in a dictionary
# =============================================================================
CDICT = dict()

# =============================================================================
# Notes from adding constant
# =============================================================================
# CDICT['KEY'] = Const('KEY', value=None, 
#                      dtype=DTYPE, dtypeio=DTYPEO,
#                      minimum=MINIMUM, maximum=MAXIMUM, length=LENGTH,
#                      options=OPTIONS, 
#                      required=REQUIRED, 
#                      comment=COMMENT, active=ACTIVE,
#                      modes=MODES)
#
# where:
#   KEY = the key to use in the yaml file
#   DTYPE = the type of the value (str, int, float, bool, list)
#   DTYPEI = the type of value in the list (if DTYPE = list)
#   MINIMUM = the minimum value (if DTYPE = int or float) [Default = None]
#   MAXIMUM = the maximum value (if DTYPE = int or float) [Default = None]
#   LENGTH = the length of the list (if DTYPE = list) [Default = None]
#   OPTIONS = a list of options to choose from (if not None) [Default = None]
#   REQUIRED = whether the key is required in the yaml file [Default = False]
#   COMMENT = a comment to add to the yaml file [Default = None] if None not 
#              added to yaml file
#   ACTIVE = whether the key is active (i.e. used in the code) [Default = False]
#   MODES = a list of modes that this key is active for [Default = None]

# =============================================================================
# Definition of inputs related to the data
# =============================================================================
# Define the data directory
comment = """\n\n
=============================================================================
 Definition of inputs related to the data
=============================================================================
 The data directory (required)
"""
CDICT['SOSSIOPATH'] = Const('SOSSIOPATH', value=None, dtype=str,
                            required=True, comment=comment,
                            active=True)
# -----------------------------------------------------------------------------
# A unique identifier for this data set
comment = """
Set the SOSSISSE ID (SID) for using the same directory as before
    if left as None the code will work out whether this yaml is found before
    or whether we need to create a new SID
"""
CDICT['SID'] = Const('SID', value=None, dtype=str, comment=comment, active=True)
# -----------------------------------------------------------------------------
# Log level (DEBUG, INFO, WARNING, ERROR, NONE)
comment = """
Log level (DEBUG, INFO, WARNING, ERROR, NONE)
"""
CDICT['LOG_LEVEL'] = Const('LOG_LEVEL', value='INFO', dtype=str, 
                           comment=comment, active=True)
# -----------------------------------------------------------------------------
# Define the name of the object (must match the object directory name)
comment = """
Name of the object (must match the object directory name)
"""
CDICT['OBJECTNAME'] = Const('OBJECTNAME', value=None, dtype=str,
                            required=True, comment=comment, active=True)
# -----------------------------------------------------------------------------
# Instrument mode i.e. JWST.NIRISS.SOSS or JWST.NIRISS.PRISM
comment = """
Instrument mode i.e. JWST.NIRISS.SOSS or JWST.NIRISS.PRISM
"""
CDICT['INSTRUMENTMODE'] = Const('INSTRUMENTMODE', value=None, dtype=str,
                                required=True, comment=comment, active=True)
# -----------------------------------------------------------------------------
# A suffix to identify this setup (e.g. a specific visit)
comment = """
A suffix to identify this setup (e.g. a specific visit)
"""
CDICT['SUFFIX'] = Const('SUFFIX', value='', dtype=str, comment=comment, 
                        active=True)
# -----------------------------------------------------------------------------
# Define the parameter file that seeded this run 
#     (not in yaml file: comment=None))
CDICT['PARAM_FILE'] = Const('PARAM_FILE', value=None, dtype=str, comment=None)
# -----------------------------------------------------------------------------
# Define whether user wants all constants in yaml file created (doesn't go in
#     yaml file itself: comment = None)
CDICT['ALL_CONSTANTS'] = Const('ALL_CONSTANTS', value=False, dtype=bool,
                               comment=None)
# -----------------------------------------------------------------------------
# Special splash
CDICT['SSPLASH'] = Const('SSPLASH', value=False, dtype=bool, comment=None)
# -----------------------------------------------------------------------------
# list of files to be processed
comment = """
list of files to be processed
   in form:
   e.g. FILES:
    - file1
    - file2
    - file3
"""
CDICT['FILES'] = Const('FILES', value=None, dtype=list, dtypei=str,
                       required=True, comment=comment, active=True)
# -----------------------------------------------------------------------------
# background file --> leave as None if the is no background available for
#                     the mode
comment = """
background file --> leave as None if the is no background available for
   the mode
"""
CDICT['BKGFILE'] = Const('BKGFILE', value=None, dtype=str, comment=comment, 
                         active=True)
# -----------------------------------------------------------------------------
# flat field file --> leave as None if there is no flat field available
#                     for the mode
comment = """
flat field file --> leave as '' if there is no flat field available
    for the mode
"""
CDICT['FLATFILE'] = Const('FLATFILE', value=None, dtype=str, comment=comment, 
                          active=True)
# -----------------------------------------------------------------------------
# trace position file
comment = """
trace position file
"""
CDICT['POS_FILE'] = Const('POS_FILE', value=None, dtype=str, comment=comment, 
                          active=True)
# -----------------------------------------------------------------------------
# wavelength calibration file (just for SOSS/FGS)
comment = """
wavelength calibration file 
"""
CDICT['WAVE_FILE'] = Const('WAVE_FILE', value=None, dtype=str, comment=comment,
                           modes='JWST.NIRSPEC.PRISM', active=True)
# -----------------------------------------------------------------------------
# allow for temporary files to speed the process if you run the code more
#    than once
comment = """
allow for temporary files to speed the process if you run the code more
   than once
"""
CDICT['ALLOW_TEMPORARY'] = Const('ALLOW_TEMPORARY', value=True, dtype=bool, 
                                 comment=comment, active=True)
# -----------------------------------------------------------------------------
# Use temporary files (False we overwrite the files for a fresh start even
#   if they exist on disk)
comment = """
Use temporary files (False we overwrite the files for a fresh start even
  if they exist on disk)
"""
CDICT['USE_TEMPORARY'] = Const('USE_TEMPORARY', value=True, dtype=bool, 
                               comment=comment, active=True)
# -----------------------------------------------------------------------------
# Save results at the end
comment = """
Save results at the end
"""
CDICT['SAVE_RESULTS'] = Const('SAVE_RESULTS', value=True, dtype=bool, 
                              comment=comment, active=True)

# -----------------------------------------------------------------------------
# Switch for turning on/off the white light curve step
comment = """
Switch for turning on/off the white light curve step
"""
CDICT['WHITE_LIGHT_CURVE'] = Const('WHITE_LIGHT_CURVE', value=True, dtype=bool,
                                   comment=comment, active=True)

# -----------------------------------------------------------------------------
# Switch for turning on/off the spectral extraction step
comment = """
Switch for turning on/off the spectral extraction step
"""
CDICT['SPECTRAL_EXTRACTION'] = Const('SPECTRAL_EXTRACTION', value=True,
                                     dtype=bool, comment=comment, active=True)

# =============================================================================
# Definition of paths (normally created at run time but can be overridden)
# =============================================================================
# all data for this instrument mode will be stored under this directory
comment = """\n\n
=============================================================================
 Definition of paths (normally created at run time but can be overridden)
=============================================================================
all data for this instrument mode will be stored under this directory
"""
CDICT['MODEPATH'] = Const('MODEPATH', value=None, dtype=str, comment=comment,
                          active=False)
# -----------------------------------------------------------------------------
# the calibration path is where we store all calibration files
comment = """
the calibration path is where we store all calibration files
"""
CDICT['CALIBPATH'] = Const('CALIBPATH', value=None, dtype=str, comment=comment,
                          active=False)
# -----------------------------------------------------------------------------
# the raw path is where we store all the raw data
comment = """
the raw path is where we store all the raw data
"""
CDICT['RAWPATH'] = Const('RAWPATH', value=None, dtype=str, comment=comment,
                          active=False)
# -----------------------------------------------------------------------------
# the object path is where we store all the object data
comment = """
the object path is where we store all the object data
"""
CDICT['OBJECTPATH'] = Const('OBJECTPATH', value=None, dtype=str, 
                            comment=comment,  active=False)
# -----------------------------------------------------------------------------
# the temp path is where we store temporary versions of the raw data
comment = """
the temp path is where we store temporary versions of the raw data
"""
CDICT['TEMP_PATH'] = Const('TEMP_PATH', value=None, dtype=str, comment=comment,
                          active=False)
# -----------------------------------------------------------------------------
# the plot path
comment = """
the plot path
"""
CDICT['PLOT_PATH'] = Const('PLOT_PATH', value=None, dtype=str, comment=comment,
                          active=False)
# -----------------------------------------------------------------------------
# other data is stored in this path
comment = """
other data is stored in this path
"""
CDICT['OTHER_PATH'] = Const('OTHER_PATH', value=None, dtype=str, 
                            comment=comment, active=False)
# -----------------------------------------------------------------------------
# the fits path
comment = """
the fits path
"""
CDICT['FITS_PATH'] = Const('FITS_PATH', value=None, dtype=str, comment=comment,
                          active=False)

# =============================================================================
# Definition of inputs related to plots
# =============================================================================
# if you want to see the plots at the end of the code, add your login here
# this should be the string as returned by "os.getlogin()"
# or use SHOW_PLOTS to always show plots
comment = """\n\n
=============================================================================
 Definition of inputs related to plots
=============================================================================

if you want to see the plots at the end of the code, add your login here
this should be the string as returned by "os.getlogin()"
or use SHOW_PLOTS to always show plots

e.g. USER_SHOW_PLOTS:
     - cook
"""
CDICT['USER_SHOW_PLOTS'] = Const('USER_SHOW_PLOTS', value=[], dtype=list,
                                 dtypei=str, comment=comment, active=True)
# -----------------------------------------------------------------------------
# Or you can use the following to always show plots regardless of the username
comment = """
Or you can use the following to always show plots regardless of the username
"""
CDICT['SHOW_PLOTS'] = Const('SHOW_PLOTS', value=False, dtype=bool, 
                            comment=comment, active=True)
# -----------------------------------------------------------------------------
# output(s) type of figure
comment = """
output(s) type of figure
"""
CDICT['FIGURE_TYPES'] = Const('FIGURE_TYPES', value=['png', 'pdf'],
                              dtype=list, dtypei=str,
                              options=['png', 'pdf'], comment=comment,
                              active=True)
# -----------------------------------------------------------------------------
# Define the spectrum plot y limits in parts per million (ppm)
#  Formally spectrum_ylim_ppm
comment = """
Define the spectrum plot y limits in parts per million (ppm)
"""
CDICT['PLOT_SPECTRUIM_YLIM'] = Const('PLOT_SPECTRUIM_YLIM',
                                     value=[-6000, 12000],
                                     dtype=list, dtypei=float, comment=comment,
                                     active=True)

# =============================================================================
# Definition of inputs related to the position within data cube timeseries
# =============================================================================
# DQ flags that we should use (list)
comment = """\n\n
=============================================================================
Definition of inputs related to the position within data cube timeseries
=============================================================================
DQ flags that we should use (list)
    e.g.
    - 0
    - 2
"""
CDICT['VALID_DQ'] = Const('VALID_DQ', value=None, dtype=list, dtypei=int,
                          minimum=0, comment=comment, active=True)
# -----------------------------------------------------------------------------
# Define the Nth frame for 1st contact [it1], 2nd contact [it2], ... through
#    to the 4th contact [itn]
# formally it
comment = """
define the Nth frame for 1st contact [it1],
     2nd contact [it2] ... through 4th contact
     e.g. CONTACT_FRAMES:
        - 90
        - 97
        - 103
        - 110
"""
CDICT['CONTACT_FRAMES'] = Const('CONTACT_FRAMES', value=[90, 97, 103, 110],
                                dtype=list, dtypei=int,
                                minimum=0, length=4, comment=comment,
                                active=True)
# -----------------------------------------------------------------------------
# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values
#
# reject domain:
# - [0, 600]
comment = """
# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values:
#
# REJECT_DOMAIN:
# - [0, 600]
#
# if you want to reject two bits of domain, 0-600 and 3000-3200
# just use
# REJECT_DOMAIN:
# - [0, 600]
# - [3000, 3200]
"""
CDICT['REJECT_DOMAIN'] = Const('REJECT_DOMAIN', value=None, dtype=list,
                               dtypei=list, comment=comment, active=True)
# -----------------------------------------------------------------------------
# If PRISM data or saturated, you can perform a CDS between these two readouts
#cds_id:
comment = """
If FILES are CDS data or saturated, you can perform a CDS between these two
e.g. 
CDS_IDS:
- 0   # first
- 2   # last
"""
CDICT['CDS_IDS'] = Const('CDS_IDS', value=None, dtype=list, dtypei=int,
                         minimum=0, comment=comment, active=True)
# -----------------------------------------------------------------------------
# If input is a CDS file you must define the read out noise
comment = """
If input is a CDS file you must define the read out noise
"""
CDICT['CDS_RON'] = Const('CDS_RON', value=None, dtype=float, minimum=0.0, 
                         comment=comment, active=True)


# =============================================================================
# Definition of inputs to the linear model
# =============================================================================
# fit the dx -- along the dispersion
comment = """\n\n
=============================================================================
 Definition of inputs to the linear model
=============================================================================
fit the dx -- along the dispersion
"""
CDICT['FIT_DX'] = Const('FIT_DX', value=True, dtype=bool, comment=comment, 
                        active=True)
# -----------------------------------------------------------------------------
# fit the dy -- along the cross-dispersion
comment = """
fit the dy -- along the cross-dispersion
"""
CDICT['FIT_DY'] = Const('FIT_DY', value=True, dtype=bool, comment=comment, 
                        active=True)
# -----------------------------------------------------------------------------
# fit the before - after morphological change
#   Formally before_after
comment = """
fit the before - after morphological change
"""
CDICT['FIT_BEFORE_AFTER'] = Const('FIT_BEFORE_AFTER', value=False, dtype=bool, 
                                  comment=comment, active=True)
# -----------------------------------------------------------------------------
# fit the rotation in the linear reconstruction of the trace
comment = """
fit the rotation in the linear reconstruction of the trace
"""
CDICT['FIT_ROTATION'] = Const('FIT_ROTATION', value=True, dtype=bool, 
                              comment=comment, active=True)
# -----------------------------------------------------------------------------
# fit the zero point offset in the linear model for the trace
#  should not be used at the time as the quadratic term
#  Formally zero_point_offset
comment = """
fit the zero point offset in the linear model for the trace
    should not be used at the time as the quadratic term
"""
CDICT['FIT_ZERO_POINT_OFFSET'] = Const('FIT_ZERO_POINT_OFFSET', value=True,
                                       dtype=bool, comment=comment, active=True)
# -----------------------------------------------------------------------------
# fit a flux^2 dependency. This is mostly meant as a test of the quality
# of the non-linearity correction. Normally this term should be de-correlated
# with the amplitude term. Set to false for actual science analysis
# should not be used at the same time as the zero point offset
# formally quadratic_term
comment = """
fit a flux^2 dependency. This is mostly meant as a test of the quality
   of the non-linearity correction. Normally this term should be de-correlated
   with the amplitude term. Set to false for actual science analysis
   should not be used at the same time as the zero point offset
"""
CDICT['FIT_QUAD_TERM'] = Const('FIT_QUAD_TERM', value=False, dtype=bool, 
                              comment=comment, active=True)
# -----------------------------------------------------------------------------
# fit the 2nd derivative in y, good to find glitches!
#  Formally ddy
comment = """
fit the 2nd derivative in y, good to find glitches!
"""
CDICT['FIT_DDY'] = Const('FIT_DDY', value=False, dtype=bool, 
                              comment=comment, active=True)
# -----------------------------------------------------------------------------
# fit with a PCA
comment = """
fit with a PCA
"""
CDICT['FIT_PCA'] = Const('FIT_PCA', value=False, dtype=bool, 
                              comment=comment, active=True)
# -----------------------------------------------------------------------------
# Number of PCA components to use
#  Formally n_pca
comment = """
Number of PCA components to use
"""
CDICT['FIT_N_PCA'] = Const('N_PCA', value=0, dtype=int, minimum=0, 
                           comment=comment, active=True)
# -----------------------------------------------------------------------------
# Bin the input data cube in time
# formally time_bin
comment = """
Bin the input data cube in time
"""
CDICT['DATA_BIN_TIME'] = Const('DATA_BIN_TIME', value=False, dtype=bool, 
                           comment=comment, active=True)
# -----------------------------------------------------------------------------
# Number of frames in each bin
comment = """
Number of frames in each bin
"""
CDICT['DATA_BIN_NUMBER'] = Const('DATA_BIN_NUMBER', value=1, dtype=int,
                                 minimum=1, comment=comment, active=True)

# =============================================================================
# Definition of inputs related to handling of the data within each frame
# =============================================================================
# Define whether to use pixel level de-trending
comment = """\n\n
=============================================================================
Definition of inputs related to handling of the data within each frame
=============================================================================
Define which orders to use
   e.g. TRACE_ORDERS:
        - 1
        - 2
"""
CDICT['TRACE_ORDERS'] = Const('TRACE_ORDERS', value=[1, 2], dtype=list,
                              dtypei=int, minimum=1, comment=comment, 
                              active=True)
# -----------------------------------------------------------------------------
# wavelength domain for the white light curve
#   For SOSS if this is defined we only get order 1
comment = """
wavelength domain for the white light curve
    e.g. WLC_DOMAIN:
     - 1.2
     - 1.6
"""
CDICT['WLC_DOMAIN'] = Const('WLC_DOMAIN', value=None, dtype=list,
                            dtypei=float, length=2, comment=comment, 
                            active=True)
# -----------------------------------------------------------------------------
# median of out-of-transit values for reference trace construction.
# If set to false, then we have the median of the entire timeseries
# Formally ootmed
comment = """
median of out-of-transit values for reference trace construction.
    If set to false, then we have the median of the entire timeseries
"""
CDICT['MEDIAN_OOT'] = Const('MEDIAN_OOT', value=True, dtype=bool, 
                            comment=comment, active=True)
# -----------------------------------------------------------------------------
# The number of pixels in the x direction to offset the trace by
comment = """
The number of pixels in the x direction to offset the trace by
"""
CDICT['X_TRACE_OFFSET'] = Const('X_TRACE_OFFSET', value=0, dtype=int, 
                                comment=comment, active=True)
# -----------------------------------------------------------------------------
# The number of pixels in the y direction to offset the trace by
comment = """
The number of pixels in the y direction to offset the trace by
"""
CDICT['Y_TRACE_OFFSET'] = Const('Y_TRACE_OFFSET', value=0, dtype=int, 
                                comment=comment, active=True)
# -----------------------------------------------------------------------------
# Whether to mask order zero
# Formally mask_order_0
comment = """
Whether to mask order zero
"""
CDICT['MASK_ORDER_ZERO'] = Const('MASK_ORDER_ZERO', value=True, dtype=bool,
                                 comment=comment, active=True, 
                                 modes='JWST.NIRISS.SOSS,'
                                       'JWST.NIRISS.FGS')
# -----------------------------------------------------------------------------
# Whether to recenter the trace position
comment = """
Whether to recenter the trace position
"""
CDICT['RECENTER_TRACE_POSITION'] = Const('RECENTER_TRACE_POSITION', value=True,
                                         dtype=bool, comment=comment, 
                                         active=True, 
                                         modes='JWST.NIRISS.SOSS,'
                                               'JWST.NIRISS.FGS')
# -----------------------------------------------------------------------------
# out of transit polynomial level correction
comment = """
out of transit polynomial level correction
"""
CDICT['TRANSIT_BASELINE_POLYORD'] = Const('TRANSIT_BASELINE_POLYORD', value=2,
                                          dtype=int, minimum=0, 
                                          comment=comment, active=True)
# -----------------------------------------------------------------------------
# out-of-trace baseline polynomial order
comment = """
out-of-trace baseline polynomial order
"""
CDICT['TRACE_BASELINE_POLYORD'] = Const('TRACE_BASELINE_POLYORD', value=2,
                                        dtype=int, minimum=0, 
                                        comment=comment, active=True)
# -----------------------------------------------------------------------------
# degree of the polynomial for the 1/f correction
# degree_1f_corr = 0 -> just a constant through the 256 pix spatial
# degree_1f_corr = 1 -> slope ... and so on
comment = """
degree of the polynomial for the 1/f correction
   degree_1f_corr = 0 -> just a constant through the 256 pix spatial
   degree_1f_corr = 1 -> slope ... and so on
"""
CDICT['DEGREE_1F_CORR'] = Const('DEGREE_1F_CORR', value=0, dtype=int, 
                                minimum=0, comment=comment, active=True)
# -----------------------------------------------------------------------------
# Trace extraction width. Set to 0 to use the full image
comment = """
Trace extraction width. Set to 0 to use the full image
"""
CDICT['TRACE_WIDTH_EXTRACTION'] = Const('TRACE_WIDTH_EXTRACTION', value=40,
                                        dtype=int, minimum=0, comment=comment, 
                                        active=True)
# -----------------------------------------------------------------------------
# used for masking and white light curve
comment = """
used for masking and WLC
"""
CDICT['TRACE_WIDTH_MASKING'] = Const('TRACE_WIDTH_MASKING', value=40, dtype=int,
                                     minimum=0, comment=comment, active=True)
# -----------------------------------------------------------------------------
# do remove trend from out-of-transit
comment = """
do remove trend from out-of-transit
"""
CDICT['REMOVE_TREND'] = Const('REMOVE_TREND', value=True, dtype=bool, 
                              comment=comment, active=True)
# -----------------------------------------------------------------------------
# define how the "white" transit depth is computed/assigned
#   "compute": Compute transit depth using median OOT relative flux from WLC,
#       and mean in-transit relative flux from WLC
#   OR
#   "known": provide the number in TDEPTH
comment = """
define how the "white" transit depth is computed/assigned
   "compute": Compute transit depth using median OOT relative flux from WLC,
       and mean in-transit relative flux from WLC
   OR
   "known": provide the number in TDEPTH
"""
CDICT['TDEPTH_MODE'] = Const('TDEPTH_MODE', value='compute', dtype=str,
                             options=['compute', 'known'], comment=comment, 
                             active=True)
# -----------------------------------------------------------------------------
# define the "white" transit depth if known
comment = """
define the "white" transit depth if known
"""
CDICT['TDEPTH'] = Const('TDEPTH', value=None, dtype=float, comment=comment,
                        active=True)
# -----------------------------------------------------------------------------
# Define the resolution to bin to
comment = """
Define the resolution to bin to
"""
CDICT['RESOLUTION_BIN'] = Const('RESOLUTION_BIN', value=20, dtype=int,
                                minimum=0, comment=comment, active=True)
# -----------------------------------------------------------------------------
# define the area around which we will optimize the background
#   this should be a length 4 list (x start, x end, y start, y end)
comment = """
define the area around which we will optimize the background
   this should be a length 4 list (x start, x end, y start, y end)
"""
CDICT['BACKGROUND_GLITCH_BOX'] = Const('BACKGROUND_GLITCH_BOX',
                                       value=[650, 750, 200, 240],
                                       dtype=list, dtypei=int,
                                       length=4, comment=comment,
                                       active=True)
# -----------------------------------------------------------------------------
# define the area around which the background will be optimized
#    should be a list (start, end, step)
comment = """
define the area around which the background will be optimized
    should be a list (start, end, step)
"""
CDICT['BACKGROUND_SHIFTS'] = Const('BACKGROUND_SHIFTS', value=[-5, 5, 0.2],
                                   dtype=list, dtypei=float, length=3,
                                   comment=comment, active=False)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
