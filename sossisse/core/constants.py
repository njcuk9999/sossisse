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

# =============================================================================
# Define constants in a dictionary
# =============================================================================
CDICT = dict()

# =============================================================================
# Definition of inputs related to the data
# =============================================================================
# Define the data directory
CDICT['SOSSIOPATH'] = Const('SOSSIOPATH', value=None, dtype=str,
                            required=True)

# Define the name of the object (must match the object directory name)
CDICT['OBJECTNAME'] = Const('OBJECTNAME', value=None, dtype=str,
                            required=True)

# Instrument mode i.e. JWST.NIRISS.SOSS or JWST.NIRISS.PRISM
CDICT['INSTRUMENTMODE'] = Const('INSTRUMENTMODE', value=None, dtype=str,
                                required=True)

# A suffix to identify this setup (e.g. a specific visit)
CDICT['SUFFIX'] = Const('SUFFIX', value='', dtype=str, required=False)

# Define the parameter file that seeded this run
CDICT['PARAM_FILE'] = Const('PARAM_FILE', value=None, dtype=str)

# A unique identifier for this data set
CDICT['SID'] = Const('SID', value=None, dtype=str)

# Log level (DEBUG, INFO, WARNING, ERROR, NONE)
CDICT['LOG_LEVEL'] = Const('LOG_LEVEL', value='INFO', dtype=str)

# Special splash
CDICT['SSPLASH'] = Const('SSPLASH', value=False, dtype=bool)

# list of files to be processed
# in form:
# files:
# - file1
# - file2
# - file3
CDICT['FILES'] = Const('FILES', value=None, dtype=list, dtypei=str,
                       required=True)

# The temporary in vs out file (should only be used by the code)
CDICT['TEMP_INFILE'] = Const('TEMP_INFILE', value=None, dtype=str)

# background file --> leave as None if the is no background available for
#                     the mode
CDICT['BKGFILE'] = Const('BKGFILE', value=None, dtype=str)

# flat field file --> leave as None if there is no flat field available
#                     for the mode
CDICT['FLATFILE'] = Const('FLATFILE', value=None, dtype=str)

# trace position file
# TODO: Required?
CDICT['POS_FILE'] = Const('POS_FILE', value=None, dtype=str)

# allow for temporary files to speed the process if you run the code more
#    than once
CDICT['ALLOW_TEMPORARY'] = Const('ALLOW_TEMPORARY', value=True, dtype=bool)

# Use temporary files (False we overwrite the files for a fresh start even
#   if they exist on disk)
CDICT['USE_TEMPORARY'] = Const('USE_TEMPORARY', value=True, dtype=bool)

# Save results at the end
CDICT['SAVE_RESULTS'] = Const('SAVE_RESULTS', value=True, dtype=bool)


# =============================================================================
# Definition of paths (normally created at run time but can be overridden)
# =============================================================================
# all data for this instrument mode will be stored under this directory
CDICT['MODEPATH'] = Const('MODEPATH', value=None, dtype=str)

# the calibration path is where we store all calibration files
CDICT['CALIBPATH'] = Const('CALIBPATH', value=None, dtype=str)

# the raw path is where we store all the raw data
CDICT['RAWPATH'] = Const('RAWPATH', value=None, dtype=str)

# the object path is where we store all the object data
CDICT['OBJECTPATH'] = Const('OBJECTPATH', value=None, dtype=str)

# the temp path is where we store temporary versions of the raw data
CDICT['TEMP_PATH'] = Const('TEMP_PATH', value=None, dtype=str)

# the plot path
CDICT['PLOT_PATH'] = Const('PLOT_PATH', value=None, dtype=str)

# the csv path
CDICT['CSV_PATH'] = Const('CSV_PATH', value=None, dtype=str)

# the fits path
CDICT['FITS_PATH'] = Const('FITS_PATH', value=None, dtype=str)


# =============================================================================
# Definition of inputs related to plots
# =============================================================================
# if you want to see the plots at the end of the code, add your login here
# this should be the string as returned by "os.getlogin()"
CDICT['USER_SHOW_PLOTS'] = Const('USER_SHOW_PLOTS', value=[], dtype=list,
                                 dtypei=str)

# Or you can use the following to always show plots regardless of the username
CDICT['SHOW_PLOTS'] = Const('SHOW_PLOTS', value=False, dtype=bool)

# output(s) type of figure
CDICT['FIGURE_TYPES'] = Const('FIGURE_TYPES', value=['png', 'pdf'],
                              dtype=list, dtypei=str,
                              options=['png', 'pdf'])

# Define the spectrum plot y limits in parts per million (ppm)
#  Formally spectrum_ylim_ppm
CDICT['PLOT_SPECTRUIM_YLIM'] = Const('PLOT_SPECTRUIM_YLIM',
                                     value=[-6000, 12000],
                                     dtype=list, dtypei=float)

# =============================================================================
# Definition of inputs related to the position within data cube timeseries
# =============================================================================
# TODO: Need comment
# TODO: Required?
CDICT['VALID_DQ'] = Const('VALID_DQ', value=[], dtype=list, dtypei=int,
                          minimum=0)

# Define the Nth frame for 1st contact [it1], 2nd contact [it2], ... through
#    to the 4th contact [itn]
# formally it
CDICT['NTH_FRAME'] = Const('NTH_FRAME', value=[], dtype=list, dtypei=int,
                           minimum=0, length=4)

# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values
#
# reject domain:
# - [0, 600]
CDICT['REJECT_DOMAIN'] = Const('REJECT_DOMAIN', value=[], dtype=list,
                               dtypei=list)

# If PRISM data or saturated, you can perform a CDS between these two readouts
#cds_id:
CDICT['CDS_IDS'] = Const('CDS_IDS', value=[], dtype=list, dtypei=int,
                         minimum=0)

# =============================================================================
# Definition of inputs to the linear model
# =============================================================================
# fit the dx -- along the dispersion
CDICT['FIT_DX'] = Const('FIT_DX', value=True, dtype=bool)

# fit the dy -- along the cross-dispersion
CDICT['FIT_DY'] = Const('FIT_DY', value=True, dtype=bool)

# fit the before - after morphological change
#   Formally before_after
CDICT['FIT_BEFORE_AFTER'] = Const('FIT_BEFORE_AFTER', value=False, dtype=bool)

# fit the rotation in the linear reconstruction of the trace
CDICT['FIT_ROTATION'] = Const('FIT_ROTATION', value=True, dtype=bool)

# fit the zero point offset in the linear model for the trace
#  Formally zero_point_offset
CDICT['FIT_ZERO_POINT_OFFSET'] = Const('FIT_ZERO_POINT_OFFSET', value=True,
                                       dtype=bool)

# fit a flux^2 dependency. This is mostly meant as a test of the quality
# of the non-linearity correction. Normally this term should be de-correlated
# with the amplitude term. Set to false for actual science analysis
# should not be used at the same time as the zero point offset
# formally quadratic_term
CDICT['FIT_QUAD_TERM'] = Const('FIT_QUAD_TERM', value=False, dtype=bool)

# fit the 2nd derivative in y, good to find glitches!
#  Formally ddy
CDICT['FIT_DDY'] = Const('FIT_DDY', value=False, dtype=bool)

# fit with a PCA
CDICT['FIT_PCA'] = Const('FIT_PCA', value=False, dtype=bool)

# Number of PCA components to use
#  Formally n_pca
CDICT['FIT_N_PCA'] = Const('N_PCA', value=0, dtype=int, minimum=0)

# =============================================================================
# Definition of inputs related to handling of the data within each frame
# =============================================================================
# Define whether to use pixel level detrending
CDICT['PIXEL_LEVEL_DETRENDING'] = Const('PIXEL_LEVEL_DETRENDING', value=False,
                                        dtype=bool)

# Define which orders to use
CDICT['TRACE_ORDERS'] = Const('TRACE_ORDERS', value=[1, 2], dtype=list,
                              dtypei=int, minimum=1)

# wavelength domain for the white light curve
CDICT['WLC_DOMAIN'] = Const('WLC_DOMAIN', value=[1.2, 1.6], dtype=list,
                            dtypei=float)

# median of out-of-transit values for reference trace construction.
# If set to false, then we have the median of the entire timeseries
# Formally ootmed
CDICT['MEDIAN_OOT'] = Const('MEDIAN_OOT', value=True, dtype=bool)

# The number of pixels in the x direction to offset the trace by
CDICT['X_TRACE_OFFSET'] = Const('X_TRACE_OFFSET', value=0, dtype=int)

# The number of pixels in the y direction to offset the trace by
CDICT['Y_TRACE_OFFSET'] = Const('Y_TRACE_OFFSET', value=0, dtype=int)

# Whether to mask order zero
# Formally mask_order_0
CDICT['MASK_ORDER_ZERO'] = Const('MASK_ORDER_ZERO', value=True, dtype=bool)

# Whether to recenter the trace position
CDICT['RECENTER_TRACE_POSITION'] = Const('RECENTER_TRACE_POSITION', value=True,
                                         dtype=bool)

# out of transit polynomial level correction
CDICT['TRANSIT_BASELINE_POLYORD'] = Const('TRANSIT_BASELINE_POLYORD', value=2,
                                          dtype=int, minimum=0)

# out-of-trace baseline polynomial order
CDICT['TRACE_BASELINE_POLYORD'] = Const('TRACE_BASELINE_POLYORD', value=2,
                                        dtype=int, minimum=0)

# degree of the polynomial for the 1/f correction
# degree_1f_corr = 0 -> just a constant through the 256 pix spatial
# degree_1f_corr = 1 -> slope ... and so on
CDICT['DEGREE_1F_CORR'] = Const('DEGREE_1F_CORR', value=0, dtype=int, minimum=0)

# set trace_width_masking to 0 to use the full image
# use in SOSSICE
CDICT['TRACE_WIDTH_EXTRACTION'] = Const('TRACE_WIDTH_EXTRACTION', value=40,
                                        dtype=int, minimum=0)

# used for masking and white light curve
CDICT['TRACE_WIDTH_MASKING'] = Const('TRACE_WIDTH_MASKING', value=40, dtype=int,
                                     minimum=0)

# do remove trend from out-of-transit
CDICT['REMOVE_TREND'] = Const('REMOVE_TREND', value=True, dtype=bool)

# define how the "white" transit depth is computed/assigned
#   "compute": Compute transit depth using median OOT relative flux from WLC,
#       and mean in-transit relative flux from WLC
#   OR
#   "known": provide the number in TDEPTH
CDICT['TDEPTH_MODE'] = Const('TDEPTH_MODE', value='compute', dtype=str,
                             options=['compute', 'known'])

# define the "white" transit depth if known
CDICT['TDEPTH'] = Const('TDEPTH', value=None, dtype=float)

# Define the resolution to bin to
CDICT['RESOLUTION_BIN'] = Const('RESOLUTION_BIN', value=20, dtype=int,
                                minimum=0)

# define the area around which we will optimize the background
#   this should be a length 4 list (x start, x end, y start, y end)
CDICT['SOSS_BACKGROUND_GLITCH_BOX'] = Const('SOSS_BACKGROUND_GLITCH_BOX',
                                            value=[650, 750, 200, 240],
                                            dtype=list, dtypei=int,
                                            length=4)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
