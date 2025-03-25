#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant definitions

Created on 2024-08-13

@author: cook
"""
from aperocore.constants import constant_functions

from sossisse.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.constants'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Constants definition
Const = constant_functions.Const
ConstDict = constant_functions.ConstantsDict
CDict = ConstDict(__NAME__)

# Define the title for the yaml
CDict.title = CDict.yaml_title('SOSSISSE', setup_program='sossisse_setup.py',
                               version=__version__, date=__date__)

# =============================================================================
# Notes from adding constant
# =============================================================================
# CDICT['KEY'] = Const('KEY', value=None, 
#                      dtype=DTYPE, dtypeio=DTYPEO,
#                      minimum=MINIMUM, maximum=MAXIMUM, length=LENGTH,
#                      options=OPTIONS, 
#                      not_none=NOT_NONE,
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
#   NOT_NONE = whether the key is required in the yaml file [Default = False]
#   COMMENT = a comment to add to the yaml file [Default = None] if None not 
#              added to yaml file
#   ACTIVE = whether the key is active (i.e. used in the code) [Default = False]
#   MODES = a list of modes that this key is active for [Default = None]

# =============================================================================
# Define switches
# =============================================================================
cgroup = 'RUN'
CDict.add_group(cgroup,
                description='Switches to turn large parts of the code on/off',
                active=True, user=True, source=__NAME__)
# -----------------------------------------------------------------------------
# Switch for turning on/off the white light curve step
CDict.add('LINEAR_RECON', value=True, dtype=bool,
          source=__NAME__, user=True, active=True,
          group=cgroup,
          description='Switch for turning on/off the linear '
                      'reconstruction step')
# -----------------------------------------------------------------------------
# Switch for turning on/off the spectral extraction step
CDict.add('SPECTRAL_EXTRACTION', value=True, dtype=bool,
          source=__NAME__, user=True, active=True,
          group=cgroup,
          description='Switch for turning on/off the spectral '
                      'extraction')

# =============================================================================
# Definition of inputs related to the data
# =============================================================================
cgroup = 'INPUTS'
CDict.add_group(cgroup, description='Definition of inputs related to the data',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# Define the data directory
CDict.add('SOSSIOPATH', value=None, dtype=str, not_none=True,
          source=__NAME__, user=True, active=True,
          cmd_arg='sossiopath', group=cgroup,
          description='The data directory')
# -----------------------------------------------------------------------------
# A unique identifier for this data set
CDict.add('SID', value=None, dtype=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Set the SOSSISSE ID (SID) for using the same '
                      'directory as before if left as None the code '
                      'will work out whether this yaml is found before '
                      'or whether we need to create a new SID')
# -----------------------------------------------------------------------------
# Log level (DEBUG, INFO, WARNING, ERROR, NONE)
CDict.add('LOG_LEVEL', value='INFO', dtype=str,
          source=__NAME__, user=True, active=True,
          options=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'NONE'],
          group=cgroup,
          description='Log level (DEBUG, INFO, WARNING, ERROR, NONE)')
# -----------------------------------------------------------------------------
# Define the name of the object (must match the object directory name)
CDict.add('OBJECTNAME', value=None, dtype=str, not_none=True,
          source=__NAME__, user=True, active=True,
          cmd_arg='objname',
          group=cgroup,
          description='Name of the object (must match the object '
                      'directory name)')
# -----------------------------------------------------------------------------
# Instrument mode i.e. JWST.NIRISS.SOSS or JWST.NIRISS.PRISM
CDict.add('INSTRUMENTMODE', value=None, dtype=str, not_none=True,
          source=__NAME__, user=True, active=True,
          options=base.INSTRUMENTS,
          cmd_arg='instmode', group=cgroup,
          description='Instrument mode e.g. one of the following: {0}'
                      ''.format('\n\t'.join(base.INSTRUMENTS)))
# -----------------------------------------------------------------------------
# A suffix to identify this setup (e.g. a specific visit)
CDict.add('SUFFIX', value='', dtype=str,
          source=__NAME__, user=True, active=True,
          group=cgroup,
          description='A suffix to identify this setup (e.g. a '
                      'specific visit)')
# -----------------------------------------------------------------------------
# Define the parameter file that seeded this run 
#     (not in yaml file: comment=None))
CDict.add('PARAM_FILE', value=None, dtype=str,
          source=__NAME__, user=False, active=False,
          cmd_arg='param_file',
          cmd_kwargs=dict(nargs='?', type=str, default=None,
                          action='store'),
          group=cgroup,
          description='The parameter yaml file to use.')
# -----------------------------------------------------------------------------
# Define the yaml name (for creating new param files only)
CDict.add('YAML_NAME', value=None, dtype=str,
          source=__NAME__, user=False, active=False,
          cmd_arg='yaml_name', group=cgroup,
          description='The name of the yaml file to create '
                      '(if not using a previous yaml file)')
# -----------------------------------------------------------------------------
# Define whether user wants all constants in yaml file created (doesn't go in
#     yaml file itself: comment = None)
CDict.add('ALL_CONSTANTS', value=False, dtype=bool,
          source=__NAME__, user=False, active=False,
          cmd_arg='all_const', group=cgroup,
          description='Add all constants to yaml file')
# -----------------------------------------------------------------------------
# Special splash
CDict.add('SSPLASH', value=False, dtype=bool,
          source=__NAME__, user=False, active=False, group=cgroup,
          description='Special splash screen for this run')

# =============================================================================
# Definition of general inputs
# =============================================================================
cgroup = 'GENERAL'
CDict.add_group(cgroup, description='Definition of general inputs',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# Raw files
CDict.add('FILES', value=None, dtype=list, dtypei=str, not_none=False,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='List of files to be processed in form: '
                      '\nFILES:\n - file1\n - file2\n - file3')
# -----------------------------------------------------------------------------
# background file --> leave as None if the is no background available for
#                     the mode - if empty DO_BACKGROUND is set to False
#                     regardless of the value
CDict.add('BKGFILE', value=None, dtype=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Background file, leave as None if there is no '
                      'background available for the mode, if empty '
                      'DO_BACKGROUND is set to False regardless of '
                      'its value')
# -----------------------------------------------------------------------------
# do background correction - must have BKGFILE defined to do this
CDict.add('DO_BACKGROUND', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Do background correction - must have BKGFILE '
                      'defined to do this')
# -----------------------------------------------------------------------------
# flat field file --> leave as None if there is no flat field available
#                     for the mode
CDict.add('FLATFILE', value=None, dtype=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Flat field file, leave as None if there is no '
                      'flat field available for the mode')
# -----------------------------------------------------------------------------
# trace position file
CDict.add('POS_FILE', value=None, dtype=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Trace position file')
# -----------------------------------------------------------------------------
# wavelength calibration
CDict.add('WAVE_FILE', value=None, dtype=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Wavelength calibration file')
# -----------------------------------------------------------------------------
# Wavelength calibration file type
CDict.add('WAVE_FILE_TYPE', value=None, dtype=str,
          options=['ext1d', 'fits', 'hdf5'],
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Wavelength calibration file type [ext1d, fits, hdf5]')

# -----------------------------------------------------------------------------
# allow for temporary files to speed the process if you run the code more
#    than once
CDict.add('ALLOW_TEMPORARY', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Allow for temporary files to speed the process if '
                      'you run the code more than once')
# -----------------------------------------------------------------------------
# Use temporary files (False we overwrite the files for a fresh start even
#   if they exist on disk)
CDict.add('USE_TEMPORARY', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Use temporary files (False we overwrite the files '
                      'for a fresh start even if they exist on disk)')
# -----------------------------------------------------------------------------
# Save results at the end
CDict.add('SAVE_RESULTS', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Save results at the end')
# -----------------------------------------------------------------------------
# Define whether to use pixel level de-trending
CDict.add('TRACE_ORDERS', value=None, dtype=list, dtypei=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS',
          description='Define which orders to use e.g. \n - 1 \n - 2')

# -----------------------------------------------------------------------------
# wavelength domain for the white light curve
#   For SOSS if this is defined we only get order 1
CDict.add('WLC_DOMAIN', value=None, dtype=list, dtypei=float, length=2,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='wavelength domain for the white light curve '
                      'e.g. \n - 1.2 \n - 1.6')

# =============================================================================
# Definition of paths (normally created at run time but can be overridden)
# =============================================================================
cgroup = 'PATHS'
CDict.add_group(cgroup, description='Definition of paths (normally created '
                                    'at run time but can be overridden)',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# all data for this instrument mode will be stored under this directory
CDict.add('MODEPATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='All data for this instrument mode will be stored '
                      'under this directory')
# -----------------------------------------------------------------------------
# the calibration path is where we store all calibration files
CDict.add('CALIBPATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The calibration path is where we store all '
                      'calibration files')
# -----------------------------------------------------------------------------
# the path to the yaml file backups
CDict.add('YAMLPATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The path to the yaml file backups')
# -----------------------------------------------------------------------------
# the raw path is where we store all the raw data
CDict.add('RAWPATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The raw path is where we store all the raw data')
# -----------------------------------------------------------------------------
# the object path is where we store all the object data
CDict.add('OBJECTPATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The object path is where we store all the object '
                      'data')
# -----------------------------------------------------------------------------
# the object path is where we store all the object data
CDict.add('SID_PATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The path within the OBJECTPATH where this SID run '
                      'is stored')
# -----------------------------------------------------------------------------
# the temp path is where we store temporary versions of the raw data
CDict.add('TEMP_PATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The temp path is where we store temporary '
                      'versions of the raw data')
# -----------------------------------------------------------------------------
# the plot path
CDict.add('PLOT_PATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The plot path')
# -----------------------------------------------------------------------------
# other data is stored in this path
CDict.add('OTHER_PATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='Other data is stored in this path')
# -----------------------------------------------------------------------------
# the fits path
CDict.add('FITS_PATH', value=None, dtype=str,
          source=__NAME__, active=False, user=False, group=cgroup,
          description='The fits path')

# =============================================================================
# Definition of inputs related to plots
# =============================================================================
cgroup = 'PLOTS'
CDict.add_group(cgroup, description='Definition of inputs related to plots',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# user show plots
CDict.add('USER_SHOW', value=[], dtype=list, dtypei=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='if you want to see the plots at the end of the '
                      'code, add your login here. \nThis should be the '
                      'string as returned by "os.getlogin()" or use '
                      'PLOTS.SHOW to always show plots')
# -----------------------------------------------------------------------------
# Or you can use the following to always show plots regardless of the username
CDict.add('SHOW', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Or you can use the following to always show '
                      'plots regardless of the username')
# -----------------------------------------------------------------------------
# output(s) type of figure
CDict.add('FIGURE_TYPES', value=['png', 'pdf'], dtype=list, dtypei=str,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='output(s) type of figure')

# =============================================================================
# Definition for white light curve
# =============================================================================
cgroup = 'WLC'
CDict.add_group(cgroup, description='Definition for white light curve',
                source=__NAME__, user=True, active=True)

# =============================================================================
# Definition of inputs related to the position within data cube timeseries
# =============================================================================
cgroup = 'WLC.INPUTS'
CDict.add_group(cgroup,
                description='Definition of inputs related to the position '
                            'within data cube timeseries',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# DQ flags that we should use (list)
CDict.add('VALID_DQ', value=[0, 2], dtype=list, dtypei=int,
          source=__NAME__, user=True, active=True,
          not_none=False, minimum=0, group=cgroup,
          description='DQ flags that we should use (list)'
                      ' e.g. \n -0 \n -2')
# -----------------------------------------------------------------------------
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
CDict.add('BASELINE_INTS', value=None, dtype=list, dtypei=list,
          source=__NAME__, user=True, active=True, not_none=False,
          group=cgroup,
          description='Integrations used to construct a model PSF. ' 
                       'This should be a list of [start, end] lists. '
                      '\nIdeally, these would be out-of-transit (if any) and '
                      'without flares. '
                      '\nIf you observed constantly a variable '
                      'object (brown dwarf, phase curve), you should enter the '
                      'first to last as frame IDs, except significant flares. '
                      'Note that on a first pass of data reduction, you may '
                      'not know the optimal parameters. '
                      '\nYou can enter a guess '
                      'and fine-tune in a second step. As a starting point, '
                      'BASELINE_INTS can be [0, NINTS-1] where NINTS can be '
                      'found in your fits header. '
                      '\nNote the 1st frame is zero '
                      '(pythonic numbering). '
                      '\nNote that leaving this blank sets '
                      'the baseline from 0 to NINTS-1.'
                      '\n    E.g. '
                      '\n         BASELINE_INTS:'
                      '\n            - [0, 89]'
                      '\n           - [111, 230]')

# -----------------------------------------------------------------------------
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
CDict.add('TRANSIT_INTS', value=None, dtype=list, dtypei=list,
          source=__NAME__, user=True, active=True, not_none=False,
          group=cgroup,
          description='If there are transit(s)/eclipse(s) in the data, enter '
                      'the frames corresponding to either the 1st, 2nd, 3rd '
                      'and 4th contact [length 4 list] or the 1st to 4th '
                      'contact [length 2 list]. These should not overlap with '
                      'the BASELINE_INTS, but there may be some domain that '
                      'is neither because of flares (i.e., not within a transit '
                      'but not within baseline). '
                      '\nYou can enter a guess and fine-tune in a second step.'
                      '\nNote the 1st frame is zero (pythonic numbering)'
                      '\nNote that leaving this blank assumes there is no '
                      'transit/eclipse'
                      '\ne.g. TRANSIT_INTS:'
                      '\n         - [90, 97, 103, 110]')

# -----------------------------------------------------------------------------
# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values
#
# reject domain:
# - [0, 600]
CDict.add('REJECT_DOMAIN', value=None, dtype=list, dtypei=list,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Used to reject bits of domain from the '
                      'analysis e.g. \n - [0, 600] '
                      '\n\n if you want to reject two bits of '
                      'domain, 0-600 and 3000-3200 just use '
                      '\n - [0, 600] \n - [3000, 3200]')

# -----------------------------------------------------------------------------
# If PRISM data or saturated, you can perform a CDS between these two readouts
# cds_id:
CDict.add('CDS_IDS', value=None, dtype=list, dtypei=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='If PRISM data or saturated, you can perform '
                      'a CDS between these two readouts e.g. '
                      '\n - 0 # first \n - 2 # last')
# -----------------------------------------------------------------------------
# If input is a CDS file you must define the read out noise
CDict.add('CDS_RON', value=None, dtype=float, minimum=0.0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='If input is a CDS file you must define the '
                      'read out noise')
# -----------------------------------------------------------------------------
# define the area around which we will optimize the background
#   this should be a length 4 list (x start, x end, y start, y end)
CDict.add('BACKGROUND_GLITCH_BOX', value=None,
          dtype=list, dtypei=int, length=4,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS',
          description='define the area around which we will '
                      'optimize the background '
                      '\nThis should be a length 4 list '
                      '(x start, x end, y start, y end)')
# -----------------------------------------------------------------------------
# define the area around which the background will be optimized
#    should be a list (start, end, step)
CDict.add('BACKGROUND_SHIFTS', value=None, dtype=list,
          dtypei=float, length=3,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS',
          description='define the area around which the background '
                      'will be optimized'
                      '\n   should be a list (start, end, step)')

# =============================================================================
# Definition of inputs to the linear model
# =============================================================================
cgroup = 'WLC.LMODEL'
CDict.add_group(cgroup, description='Definition of inputs to the linear model',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# fit the dx -- along the dispersion
CDict.add('FIT_DX', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the dx -- along the dispersion')

# -----------------------------------------------------------------------------
# fit the dy -- along the cross-dispersion
CDict.add('FIT_DY', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the dy -- along the cross-dispersion')
# -----------------------------------------------------------------------------
# fit the before - after morphological change
#   Formally before_after
CDict.add('FIT_BEFORE_AFTER', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the before - after morphological change')
# -----------------------------------------------------------------------------
# fit the rotation in the linear reconstruction of the trace
CDict.add('FIT_ROTATION', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the rotation in the linear '
                      'reconstruction of the trace')
# -----------------------------------------------------------------------------
# fit the zero point offset in the linear model for the trace
#  should not be used at the time as the quadratic term
#  Formally zero_point_offset
CDict.add('FIT_ZERO_POINT_OFFSET', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the zero point offset in the linear '
                      'model for the trace, should not be used at'
                      ' the time as the quadratic term')
# -----------------------------------------------------------------------------
# fit a flux^2 dependency. This is mostly meant as a test of the quality
# of the non-linearity correction. Normally this term should be de-correlated
# with the amplitude term. Set to false for actual science analysis
# should not be used at the same time as the zero point offset
# formally quadratic_term
CDict.add('FIT_QUAD_TERM', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit a flux^2 dependency. This is mostly '
                      'meant as a test of the quality of the '
                      'non-linearity correction. Normally this term '
                      'should be de-correlated with the amplitude '
                      'term. Set to false for actual science '
                      'analysis should not be used at the same time '
                      'as the zero point offset')
# -----------------------------------------------------------------------------
# fit the 2nd derivative in y, good to find glitches!
#  Formally ddy
CDict.add('FIT_DDY', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit the 2nd derivative in y, good to find '
                      'glitches!')
# -----------------------------------------------------------------------------
# fit with a PCA
CDict.add('FIT_PCA', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='fit with a PCA')
# -----------------------------------------------------------------------------
# Number of PCA components to use
#  Formally n_pca
CDict.add('FIT_N_PCA', value=0, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Number of PCA components to use')
# -----------------------------------------------------------------------------
# Bin the input data cube in time
# formally time_bin
CDict.add('DATA_BIN_TIME', value=False, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Bin the input data cube in time')
# -----------------------------------------------------------------------------
# Number of frames in each bin
CDict.add('DATA_BIN_NUMBER', value=1, dtype=int, minimum=1,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Number of frames in each bin')

# =============================================================================
# Definition of inputs related to handling of the data within each frame
# =============================================================================
cgroup = 'WLC.GENERAL'
CDict.add_group(cgroup, source=__NAME__, user=True, active=True,
                description='Definition of inputs related to handling of '
                            'the data within each frame')
# -----------------------------------------------------------------------------
# whether to patch isolated bad pixels
CDict.add('PATCH_ISOLATED_BADS', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='whether to patch isolated bad pixels')
# -----------------------------------------------------------------------------
# whether to remove cosmic rays
CDict.add('REMOVE_COSMIC_RAYS', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='whether to remove cosmic rays')
# -----------------------------------------------------------------------------
# sigma to flag cosmic rays at (away from the mean)
CDict.add('COSMIC_RAY_SIGMA', value=5, dtype=float, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='sigma to flag cosmic rays at (away from the '
                      'mean)')
# -----------------------------------------------------------------------------
# median of out-of-transit values for reference trace construction.
# If set to false, then we have the median of the entire timeseries
# Formally ootmed
CDict.add('MEDIAN_OOT', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='median of out-of-transit values for reference '
                      'trace construction. If set to false, then we '
                      'have the median of the entire timeseries')
# -----------------------------------------------------------------------------
# The number of pixels in the x direction to offset the trace by
CDict.add('X_TRACE_OFFSET', value=0, dtype=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='The number of pixels in the x direction to '
                      'offset the trace by')
# -----------------------------------------------------------------------------
# The number of pixels in the y direction to offset the trace by
CDict.add('Y_TRACE_OFFSET', value=0, dtype=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='The number of pixels in the y direction to '
                      'offset the trace by')
# -----------------------------------------------------------------------------
# used for masking and white light curve
CDict.add('TRACE_Y_SCALE', value=None, dtype=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS',
          description='Set the range of dys to scan over number of '
                      '-nbypix/trace_y_scale to +nbypix/tace_y_scale')
# -----------------------------------------------------------------------------
# used for masking and white light curve
CDict.add('TRACE_X_SCALE', value=None, dtype=int,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS',
          description='Set the range of dys to scan over number of '
                      '-nbxpix/trace_x_scale to +nbxpix/trace_x_scale')
# -----------------------------------------------------------------------------
# Whether to mask order zero
# Formally mask_order_0
CDict.add('MASK_ORDER_ZERO', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS, JWST.NIRISS.FGS',
          description='Whether to mask order zero')
# -----------------------------------------------------------------------------
# Whether to recenter the trace position
CDict.add('RECENTER_TRACE_POSITION', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          modes='JWST.NIRISS.SOSS, JWST.NIRISS.FGS',
          description='Whether to recenter the trace position')
# -----------------------------------------------------------------------------
# Use fancy centering of the trace (sets RECENTER_TRACE_POSITION to False)
CDict.add('USE_FANCY_CENTERING', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Use fancy centering of the trace (sets '
                      'RECENTER_TRACE_POSITION to False')
# -----------------------------------------------------------------------------
# Whether to fit a per pixel baseline correction
CDict.add('PER_PIXEL_BASELINE_CORRECTION', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Whether to fit a per pixel baseline correction')
# -----------------------------------------------------------------------------
# out of transit polynomial level correction
CDict.add('TRANSIT_BASELINE_POLYORD', value=2, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='out of transit polynomial level correction')
# -----------------------------------------------------------------------------
# out-of-trace baseline polynomial order
CDict.add('TRACE_BASELINE_POLYORD', value=2, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='out-of-trace baseline polynomial order')
# -----------------------------------------------------------------------------
# degree of the polynomial for the 1/f correction
# degree_1f_corr = 0 -> just a constant through the 256 pix spatial
# degree_1f_corr = 1 -> slope ... and so on
CDict.add('DEGREE_1F_CORR', value=0, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='degree of the polynomial for the 1/f correction'
                      '\n0 = just a constant through the pix spatial '
                      '\n1 = slope ... and so on')
# -----------------------------------------------------------------------------
# Trace extraction width. Set to 0 to use the full image
CDict.add('TRACE_WIDTH_EXTRACTION', value=40, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Trace extraction width. Set to 0 to use the '
                      'full image')
# -----------------------------------------------------------------------------
# define the width for masking the white light curve trace
CDict.add('TRACE_WIDTH_MASKING', value=40, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='define the width for masking the white light '
                      'curve trace')

# =============================================================================
# Definition for spectral extraction
# =============================================================================
cgroup = 'SPEC_EXT'
CDict.add_group(cgroup, description='Definition for white light curve',
                source=__NAME__, user=True, active=True)
# -----------------------------------------------------------------------------
# do remove trend from out-of-transit
CDict.add('REMOVE_TREND', value=True, dtype=bool,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='remove trend from out-of-transit')
# -----------------------------------------------------------------------------
# define how the "white" transit depth is computed/assigned
#   "compute": Compute transit depth using median OOT relative flux from WLC,
#       and mean in-transit relative flux from WLC
#   OR
#   "known": provide the number in TDEPTH
CDict.add('TDEPTH_MODE', value='compute', dtype=str,
          options=['compute', 'known'],
          source=__NAME__, user=True, active=True, group=cgroup,
          description='define how the "white" transit depth is '
                      'computed/assigned \n"compute": Compute transit '
                      'depth using median OOT relative flux from WLC, '
                      'and mean in-transit relative flux from WLC \nOR '
                      '\n "known": provide the number in TDEPTH')
# -----------------------------------------------------------------------------
# define the "white" transit depth if known
CDict.add('TDEPTH', value=None, dtype=float,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='define the "white" transit depth if known')
# -----------------------------------------------------------------------------
# Define the resolution to bin to
CDict.add('RESOLUTION_BIN', value=20, dtype=int, minimum=0,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='Define the resolution to bin to')

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
