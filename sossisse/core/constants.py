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
cgroup = 'SOSSISSE.SWITCHES'
CDict.add_group(cgroup,
                description='Switches to turn large parts of the code on/off')
CDict_switches = ConstDict(cgroup)
CDict.add('SWITCHES', value=CDict_switches, dtype=ConstDict,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='')
# -----------------------------------------------------------------------------
# Switch for turning on/off the white light curve step
CDict_switches.add('WHITE_LIGHT_CURVE', value=True, dtype=bool,
                   source=__NAME__, user=True, active=True,
                   description='Switch for turning on/off the white light '
                               'curve step')
# -----------------------------------------------------------------------------
# Switch for turning on/off the spectral extraction step
CDict_switches.add('SPECTRAL_EXTRACTION', value=True, dtype=bool,
                   source=__NAME__, user=True, active=True,
                   description='Switch for turning on/off the spectral '
                               'extraction')

# =============================================================================
# Definition of inputs related to the data
# =============================================================================
cgroup = 'SOSSISSE.INPUTS'
CDict.add_group(cgroup, description='Definition of inputs related to the data')
CDict_inputs = ConstDict(cgroup)
CDict.add('INPUTS', value=CDict_inputs, dtype=ConstDict,
          source=__NAME__, user=True, active=True, group=cgroup,
          description='')
# -----------------------------------------------------------------------------
# Define the data directory
CDict_inputs.add('SOSSIOPATH', value=None, dtype=str, not_none=True,
                 source=__NAME__, user=True, active=True,
                 cmd_arg='sossiopath',
                 description='The data directory')
# -----------------------------------------------------------------------------
# A unique identifier for this data set
CDict_inputs.add('SID', value=None, dtype=str,
                 source=__NAME__, user=True, active=True,
                 description='Set the SOSSISSE ID (SID) for using the same '
                             'directory as before if left as None the code '
                             'will work out whether this yaml is found before '
                             'or whether we need to create a new SID')
# -----------------------------------------------------------------------------
# Log level (DEBUG, INFO, WARNING, ERROR, NONE)
CDict_inputs.add('LOG_LEVEL', value='INFO', dtype=str,
                 source=__NAME__, user=True, active=True,
                 options=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'NONE'],
                 description='Log level (DEBUG, INFO, WARNING, ERROR, NONE)')
# -----------------------------------------------------------------------------
# Define the name of the object (must match the object directory name)
CDict_inputs.add('OBJECTNAME', value=None, dtype=str, not_none=True,
                 source=__NAME__, user=True, active=True,
                 cmd_arg='objname',
                 description='Name of the object (must match the object '
                             'directory name)')
# -----------------------------------------------------------------------------
# Instrument mode i.e. JWST.NIRISS.SOSS or JWST.NIRISS.PRISM
CDict_inputs.add('INSTRUMENTMODE', value=None, dtype=str, not_none=True,
                 source=__NAME__, user=True, active=True,
                 options=['JWST.NIRISS.SOSS', 'JWST.NIRISS.PRISM'],
                 cmd_arg='instmode',
                 description='Instrument mode i.e. JWST.NIRISS.SOSS or '
                             'JWST.NIRISS.PRISM')
# -----------------------------------------------------------------------------
# A suffix to identify this setup (e.g. a specific visit)
CDict_inputs.add('SUFFIX', value='', dtype=str,
                 source=__NAME__, user=True, active=True,
                 description='A suffix to identify this setup (e.g. a '
                             'specific visit)')
# -----------------------------------------------------------------------------
# Define the parameter file that seeded this run 
#     (not in yaml file: comment=None))
CDict_inputs.add('PARAM_FILE', value=None, dtype=str,
                 source=__NAME__, user=False, active=False,
                 cmd_arg='param_file',
                 cmd_kwargs=dict(nargs='?', type=str, default=None,
                                 action='store'),
                 description='The parameter yaml file to use.')
# -----------------------------------------------------------------------------
# Define the yaml name (for creating new param files only)
CDict_inputs.add('YAML_NAME', value=None, dtype=str,
                 source=__NAME__, user=False, active=False,
                 cmd_arg='yaml_name',
                 description='The name of the yaml file to create '
                             '(if not using a previous yaml file)')
# -----------------------------------------------------------------------------
# Define whether user wants all constants in yaml file created (doesn't go in
#     yaml file itself: comment = None)
CDict_inputs.add('ALL_CONSTANTS', value=False, dtype=bool,
                 source=__NAME__, user=False, active=False,
                 cmd_arg='all_const',
                 description='Add all constants to yaml file')
# -----------------------------------------------------------------------------
# Special splash
CDict_inputs.add('SSPLASH', value=False, dtype=bool,
                 source=__NAME__, user=False, active=False,
                 description='Special splash screen for this run')

# =============================================================================
# Definition of general inputs
# =============================================================================
cgroup = 'SOSSISSE.GENERAL'
CDict.add_group(cgroup, description='Definition of general inputs')
CDict_gen = ConstDict(cgroup)
CDict.add('GENERAL', value=CDict_gen, dtype=ConstDict, source=__NAME__,
          user=True, active=True, group=cgroup, description='')
# -----------------------------------------------------------------------------
# Raw files
CDict_gen.add('FILES', value=None, dtype=list, dtypei=str, not_none=False,
              source=__NAME__, user=True, active=True,
              description='List of files to be processed in form: '
                          '\nFILES:\n - file1\n - file2\n - file3')
# -----------------------------------------------------------------------------
# background file --> leave as None if the is no background available for
#                     the mode - if empty DO_BACKGROUND is set to False
#                     regardless of the value
CDict_gen.add('BKGFILE', value=None, dtype=str,
              source=__NAME__, user=True, active=True,
              description='Background file, leave as None if there is no '
                          'background available for the mode, if empty '
                          'DO_BACKGROUND is set to False regardless of '
                          'its value')
# -----------------------------------------------------------------------------
# do background correction - must have BKGFILE defined to do this
CDict_gen.add('DO_BACKGROUND', value=False, dtype=bool,
              source=__NAME__, user=True, active=True,
              description='Do background correction - must have BKGFILE '
                          'defined to do this')
# -----------------------------------------------------------------------------
# flat field file --> leave as None if there is no flat field available
#                     for the mode
CDict_gen.add('FLATFILE', value=None, dtype=str,
              source=__NAME__, user=True, active=True,
              description='Flat field file, leave as None if there is no '
                          'flat field available for the mode')
# -----------------------------------------------------------------------------
# trace position file
CDict_gen.add('POS_FILE', value=None, dtype=str,
              source=__NAME__, user=True, active=True,
              description='Trace position file')
# -----------------------------------------------------------------------------
# wavelength calibration file (just for SOSS/FGS)
CDict_gen.add('WAVE_FILE', value=None, dtype=str,
              source=__NAME__, user=True, active=True,
              description='Wavelength calibration file',
              modes='JWST.NIRISS.PRISM')
# -----------------------------------------------------------------------------
# allow for temporary files to speed the process if you run the code more
#    than once
CDict_gen.add('ALLOW_TEMPORARY', value=True, dtype=bool,
              source=__NAME__, user=True, active=True,
              description='Allow for temporary files to speed the process if '
                          'you run the code more than once')
# -----------------------------------------------------------------------------
# Use temporary files (False we overwrite the files for a fresh start even
#   if they exist on disk)
CDict_gen.add('USE_TEMPORARY', value=True, dtype=bool,
              source=__NAME__, user=True, active=True,
              description='Use temporary files (False we overwrite the files '
                          'for a fresh start even if they exist on disk)')
# -----------------------------------------------------------------------------
# Save results at the end
CDict_gen.add('SAVE_RESULTS', value=True, dtype=bool,
              source=__NAME__, user=True, active=True,
              description='Save results at the end')
# -----------------------------------------------------------------------------
# Define whether to use pixel level de-trending
CDict_gen.add('TRACE_ORDERS', value=[1,2], dtype=list, dtypei=int,
              source=__NAME__, user=True, active=True,
              description='Define which orders to use e.g. \n - 1 \n - 2')

# -----------------------------------------------------------------------------
# wavelength domain for the white light curve
#   For SOSS if this is defined we only get order 1
CDict_gen.add('WLC_DOMAIN', value=None, dtype=list, dtypei=float, length=2,
              source=__NAME__, user=True, active=True,
              description='wavelength domain for the white light curve '
                          'e.g. \n - 1.2 \n - 1.6')

# =============================================================================
# Definition of paths (normally created at run time but can be overridden)
# =============================================================================
cgroup = 'SOSSISSE.PATHS'
CDict.add_group(cgroup, description='Definition of paths (normally created '
                                    'at run time but can be overridden)')
CDict_paths = ConstDict(cgroup)
CDict.add('PATHS', value=CDict_paths, dtype=ConstDict,
          source=__NAME__, user=True, active=True,
          group=cgroup, description='')
# -----------------------------------------------------------------------------
# all data for this instrument mode will be stored under this directory
CDict_paths.add('MODEPATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='All data for this instrument mode will be stored '
                            'under this directory')
# -----------------------------------------------------------------------------
# the calibration path is where we store all calibration files
CDict_paths.add('CALIBPATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The calibration path is where we store all '
                            'calibration files')
# -----------------------------------------------------------------------------
# the path to the yaml file backups
CDict_paths.add('YAMLPATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The path to the yaml file backups')
# -----------------------------------------------------------------------------
# the raw path is where we store all the raw data
CDict_paths.add('RAWPATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The raw path is where we store all the raw data')
# -----------------------------------------------------------------------------
# the object path is where we store all the object data
CDict_paths.add('OBJECTPATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The object path is where we store all the object '
                            'data')
# -----------------------------------------------------------------------------
# the object path is where we store all the object data
CDict_paths.add('SID_PATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The path within the OBJECTPATH where this SID run '
                            'is stored')
# -----------------------------------------------------------------------------
# the temp path is where we store temporary versions of the raw data
CDict_paths.add('TEMP_PATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The temp path is where we store temporary '
                            'versions of the raw data')
# -----------------------------------------------------------------------------
# the plot path
CDict_paths.add('PLOT_PATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The plot path')
# -----------------------------------------------------------------------------
# other data is stored in this path
CDict_paths.add('OTHER_PATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='Other data is stored in this path')
# -----------------------------------------------------------------------------
# the fits path
CDict_paths.add('FITS_PATH', value=None, dtype=str,
                source=__NAME__, active=False, user=False,
                description='The fits path')

# =============================================================================
# Definition of inputs related to plots
# =============================================================================
cgroup = 'SOSSISSE.PLOTS'
CDict.add_group(cgroup, description='Definition of inputs related to plots')
CDict_plots = ConstDict(cgroup)
CDict.add('PLOTS', value=CDict_plots, dtype=ConstDict,
          source=__NAME__, user=True, active=True,
          group=cgroup, description='')
# -----------------------------------------------------------------------------
# user show plots
CDict_plots.add('USER_SHOW', value=[], dtype=list, dtypei=str,
                source=__NAME__, user=True, active=True,
                description='if you want to see the plots at the end of the '
                            'code, add your login here. \nThis should be the '
                            'string as returned by "os.getlogin()" or use '
                            'PLOTS.SHOW to always show plots')
# -----------------------------------------------------------------------------
# Or you can use the following to always show plots regardless of the username
CDict_plots.add('SHOW', value=False, dtype=bool,
                source=__NAME__, user=True, active=True,
                description='Or you can use the following to always show '
                            'plots regardless of the username')
# -----------------------------------------------------------------------------
# output(s) type of figure
CDict_plots.add('FIGURE_TYPES', value=['png', 'pdf'], dtype=list, dtypei=str,
                source=__NAME__, user=True, active=True,
                description='output(s) type of figure')

# =============================================================================
# Definition for white light curve
# =============================================================================
cgroup = 'SOSSISSE.WLC'
CDict.add_group(cgroup, description='Definition for white light curve')
CDict_wlc = ConstDict(cgroup)
CDict.add('WLC', value=CDict_wlc, dtype=ConstDict,
          source=__NAME__, user=True, active=True,
          group=cgroup, description='')

# =============================================================================
# Definition of inputs related to the position within data cube timeseries
# =============================================================================
cgroup = 'SOSSISSE.WLC.INPUTS'
CDict_wlc.add_group(cgroup,
                    description='Definition of inputs related to the position '
                                'within data cube timeseries')
CDict_wlc_inputs = ConstDict(cgroup)
CDict_wlc.add('INPUTS', value=CDict_wlc_inputs, dtype=ConstDict,
              source=__NAME__, user=True, active=True,
              group=cgroup, description='')
# -----------------------------------------------------------------------------
# DQ flags that we should use (list)
CDict_wlc_inputs.add('VALID_DQ', value=None, dtype=list, dtypei=int,
                     source=__NAME__, user=True, active=True,
                     not_none=False, minimum=0,
                     description='DQ flags that we should use (list)'
                                 ' e.g. \n -0 \n -2')
# -----------------------------------------------------------------------------
# Define the Nth frame for 1st contact [it1], 2nd contact [it2], ... through
#    to the 4th contact [itn]
# formally "it"
CDict_wlc_inputs.add('CONTACT_FRAMES', value=None, dtype=list, dtypei=int,
                     source=__NAME__, user=True, active=True,
                     not_none=False, length=4, minimum=0,
                     description='Define the Nth frame for 1st contact [it1], '
                                 '2nd contact [it2] ... through 4th contact '
                                 'e.g. \n -90 \n -97 \n -103 \n -110')

# -----------------------------------------------------------------------------
# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values
#
# reject domain:
# - [0, 600]
CDict_wlc_inputs.add('REJECT_DOMAIN', value=None, dtype=list, dtypei=list,
                     source=__NAME__, user=True, active=True,
                     description='Used to reject bits of domain from the '
                                 'analysis e.g. \n - [0, 600] '
                                 '\n\n if you want to reject two bits of '
                                 'domain, 0-600 and 3000-3200 just use '
                                 '\n - [0, 600] \n - [3000, 3200]')

# -----------------------------------------------------------------------------
# If PRISM data or saturated, you can perform a CDS between these two readouts
#cds_id:
CDict_wlc_inputs.add('CDS_IDS', value=None, dtype=list, dtypei=int,
                     source=__NAME__, user=True, active=True,
                     description='If PRISM data or saturated, you can perform '
                                 'a CDS between these two readouts e.g. '
                                 '\n - 0 # first \n - 2 # last')
# -----------------------------------------------------------------------------
# If input is a CDS file you must define the read out noise
CDict_wlc_inputs.add('CDS_RON', value=None, dtype=float, minimum=0.0,
                     source=__NAME__, user=True, active=True,
                     description='If input is a CDS file you must define the '
                                 'read out noise')
# -----------------------------------------------------------------------------
# define the area around which we will optimize the background
#   this should be a length 4 list (x start, x end, y start, y end)
CDict_wlc_inputs.add('BACKGROUND_GLITCH_BOX', value=[650, 750, 200, 240],
                     dtype=list,  dtypei=int, length=4,
                     source=__NAME__, user=True, active=True,
                     description='define the area around which we will '
                                 'optimize the background '
                                 '\nThis should be a length 4 list '
                                 '(x start, x end, y start, y end)')
# -----------------------------------------------------------------------------
# define the area around which the background will be optimized
#    should be a list (start, end, step)
CDict_wlc_inputs.add('BACKGROUND_SHIFTS', value=[-5, 5, 0.2], dtype=list,
                     dtypei=float, length=3,
                     source=__NAME__, user=True, active=True,
                     description='define the area around which the background '
                                 'will be optimized'
                                 '\n   should be a list (start, end, step)')

# =============================================================================
# Definition of inputs to the linear model
# =============================================================================
cgroup = 'SOSSISSE.WLC.LMODEL'
CDict_wlc.add_group(cgroup,
                    description='Definition of inputs to the linear model')
CDict_wlc_lmodel = ConstDict(cgroup)
CDict_wlc.add('LMODEL', value=CDict_wlc_lmodel, dtype=ConstDict,
              source=__NAME__, user=True, active=True, group=cgroup,
              description='')
# -----------------------------------------------------------------------------
# fit the dx -- along the dispersion
CDict_wlc_lmodel.add('FIT_DX', value=True, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the dx -- along the dispersion')

# -----------------------------------------------------------------------------
# fit the dy -- along the cross-dispersion
CDict_wlc_lmodel.add('FIT_DY', value=True, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the dy -- along the cross-dispersion')
# -----------------------------------------------------------------------------
# fit the before - after morphological change
#   Formally before_after
CDict_wlc_lmodel.add('FIT_BEFORE_AFTER', value=False, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the before - after morphological change')
# -----------------------------------------------------------------------------
# fit the rotation in the linear reconstruction of the trace
CDict_wlc_lmodel.add('FIT_ROTATION', value=True, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the rotation in the linear '
                                 'reconstruction of the trace')
# -----------------------------------------------------------------------------
# fit the zero point offset in the linear model for the trace
#  should not be used at the time as the quadratic term
#  Formally zero_point_offset
CDict_wlc_lmodel.add('FIT_ZERO_POINT_OFFSET', value=True, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the zero point offset in the linear '
                                 'model for the trace, should not be used at'
                                 ' the time as the quadratic term')
# -----------------------------------------------------------------------------
# fit a flux^2 dependency. This is mostly meant as a test of the quality
# of the non-linearity correction. Normally this term should be de-correlated
# with the amplitude term. Set to false for actual science analysis
# should not be used at the same time as the zero point offset
# formally quadratic_term
CDict_wlc_lmodel.add('FIT_QUAD_TERM', value=False, dtype=bool,
                     source=__NAME__, user=True, active=True,
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
CDict_wlc_lmodel.add('FIT_DDY', value=False, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit the 2nd derivative in y, good to find '
                                 'glitches!')
# -----------------------------------------------------------------------------
# fit with a PCA
CDict_wlc_lmodel.add('FIT_PCA', value=False, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='fit with a PCA')
# -----------------------------------------------------------------------------
# Number of PCA components to use
#  Formally n_pca
CDict_wlc_lmodel.add('FIT_N_PCA', value=0, dtype=int, minimum=0,
                     source=__NAME__, user=True, active=True,
                     description='Number of PCA components to use')
# -----------------------------------------------------------------------------
# Bin the input data cube in time
# formally time_bin
CDict_wlc_lmodel.add('DATA_BIN_TIME', value=False, dtype=bool,
                     source=__NAME__, user=True, active=True,
                     description='Bin the input data cube in time')
# -----------------------------------------------------------------------------
# Number of frames in each bin
CDict_wlc_lmodel.add('DATA_BIN_NUMBER', value=1, dtype=int, minimum=1,
                     source=__NAME__, user=True, active=True,
                     description='Number of frames in each bin')

# =============================================================================
# Definition of inputs related to handling of the data within each frame
# =============================================================================
cgroup = 'SOSSISSE.WLC.GENERAL'
CDict_wlc.add_group(cgroup,
                    description='Definition of inputs related to handling of '
                                'the data within each frame')
CDict_wlc_gen = ConstDict(cgroup)
CDict_wlc.add('GENERAL', value=CDict_wlc_gen, dtype=ConstDict,
              source=__NAME__, user=True, active=True,
              group=cgroup, description='')
# -----------------------------------------------------------------------------
# whether to patch isolated bad pixels
CDict_wlc_gen.add('PATCH_ISOLATED_BADS', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  description='whether to patch isolated bad pixels')
# -----------------------------------------------------------------------------
# whether to remove cosmic rays
CDict_wlc_gen.add('REMOVE_COSMIC_RAYS', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  description='whether to remove cosmic rays')
# -----------------------------------------------------------------------------
# sigma to flag cosmic rays at (away from the mean)
CDict_wlc_gen.add('COSMIC_RAY_SIGMA', value=5, dtype=float, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='sigma to flag cosmic rays at (away from the '
                              'mean)')
# -----------------------------------------------------------------------------
# median of out-of-transit values for reference trace construction.
# If set to false, then we have the median of the entire timeseries
# Formally ootmed
CDict_wlc_gen.add('MEDIAN_OOT', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  description='median of out-of-transit values for reference '
                              'trace construction. If set to false, then we '
                              'have the median of the entire timeseries')
# -----------------------------------------------------------------------------
# The number of pixels in the x direction to offset the trace by
CDict_wlc_gen.add('X_TRACE_OFFSET', value=0, dtype=int,
                  source=__NAME__, user=True, active=True,
                  description='The number of pixels in the x direction to '
                              'offset the trace by')
# -----------------------------------------------------------------------------
# The number of pixels in the y direction to offset the trace by
CDict_wlc_gen.add('Y_TRACE_OFFSET', value=0, dtype=int,
                  source=__NAME__, user=True, active=True,
                  description='The number of pixels in the y direction to '
                              'offset the trace by')
# -----------------------------------------------------------------------------
# used for masking and white light curve
CDict_wlc_gen.add('TRACE_Y_SCALE', value=10, dtype=int,
                  source=__NAME__, user=True, active=True,
                  description='Set the range of dys to scan over number of '
                              '-nbypix/trace_y_scale to +nbypix/tace_y_scale')
# -----------------------------------------------------------------------------
# used for masking and white light curve
CDict_wlc_gen.add('TRACE_X_SCALE', value=5, dtype=int,
                  source=__NAME__, user=True, active=True,
                  description='Set the range of dys to scan over number of '
                              '-nbxpix/trace_x_scale to +nbxpix/trace_x_scale')
# -----------------------------------------------------------------------------
# Whether to mask order zero
# Formally mask_order_0
CDict_wlc_gen.add('MASK_ORDER_ZERO', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  modes='JWST.NIRISS.SOSS, JWST.NIRISS.FGS',
                  description='Whether to mask order zero')
# -----------------------------------------------------------------------------
# Whether to recenter the trace position
CDict_wlc_gen.add('RECENTER_TRACE_POSITION', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  modes='JWST.NIRISS.SOSS, JWST.NIRISS.FGS',
                  description='Whether to recenter the trace position')
# -----------------------------------------------------------------------------
# Use fancy centering of the trace (sets RECENTER_TRACE_POSITION to False)
CDict_wlc_gen.add('USE_FANCY_CENTERING', value=True, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  description='Use fancy centering of the trace (sets '
                              'RECENTER_TRACE_POSITION to False')
# -----------------------------------------------------------------------------
# Whether to fit a per pixel baseline correction
CDict_wlc_gen.add('PER_PIXEL_BASELINE_CORRECTION', value=False, dtype=bool,
                  source=__NAME__, user=True, active=True,
                  description='Whether to fit a per pixel baseline correction')
# -----------------------------------------------------------------------------
# out of transit polynomial level correction
CDict_wlc_gen.add('TRANSIT_BASELINE_POLYORD', value=2, dtype=int, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='out of transit polynomial level correction')
# -----------------------------------------------------------------------------
# out-of-trace baseline polynomial order
CDict_wlc_gen.add('TRACE_BASELINE_POLYORD', value=2, dtype=int, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='out-of-trace baseline polynomial order')
# -----------------------------------------------------------------------------
# degree of the polynomial for the 1/f correction
# degree_1f_corr = 0 -> just a constant through the 256 pix spatial
# degree_1f_corr = 1 -> slope ... and so on
CDict_wlc_gen.add('DEGREE_1F_CORR', value=0, dtype=int, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='degree of the polynomial for the 1/f correction'
                              '\n0 = just a constant through the pix spatial '
                              '\n1 = slope ... and so on')
# -----------------------------------------------------------------------------
# Trace extraction width. Set to 0 to use the full image
CDict_wlc_gen.add('TRACE_WIDTH_EXTRACTION', value=40, dtype=int, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='Trace extraction width. Set to 0 to use the '
                              'full image')
# -----------------------------------------------------------------------------
# define the width for masking the white light curve trace
CDict_wlc_gen.add('TRACE_WIDTH_MASKING', value=40, dtype=int, minimum=0,
                  source=__NAME__, user=True, active=True,
                  description='define the width for masking the white light '
                              'curve trace')

# =============================================================================
# Definition for spectral extraction
# =============================================================================
cgroup = 'SOSSISSE.SPEC_EXT'
CDict.add_group(cgroup, description='Definition for white light curve')
CDict_spec = ConstDict(cgroup)
CDict.add('SPEC_EXT', value=CDict_spec, dtype=ConstDict,
          source=__NAME__, user=True, active=True,
          group=cgroup, description='')
# -----------------------------------------------------------------------------
# do remove trend from out-of-transit
CDict_spec.add('REMOVE_TREND', value=True, dtype=bool,
               source=__NAME__, user=True, active=True,
               description='remove trend from out-of-transit')
# -----------------------------------------------------------------------------
# define how the "white" transit depth is computed/assigned
#   "compute": Compute transit depth using median OOT relative flux from WLC,
#       and mean in-transit relative flux from WLC
#   OR
#   "known": provide the number in TDEPTH
CDict_spec.add('TDEPTH_MODE', value='compute', dtype=str,
               options=['compute', 'known'],
               source=__NAME__, user=True, active=True,
               description='define how the "white" transit depth is '
                           'computed/assigned \n"compute": Compute transit '
                           'depth using median OOT relative flux from WLC, '
                           'and mean in-transit relative flux from WLC \nOR '
                           '\n "known": provide the number in TDEPTH')
# -----------------------------------------------------------------------------
# define the "white" transit depth if known
CDict_spec.add('TDEPTH', value=None, dtype=float,
               source=__NAME__, user=True, active=True,
               description='define the "white" transit depth if known')
# -----------------------------------------------------------------------------
# Define the resolution to bin to
CDict_spec.add('RESOLUTION_BIN', value=20, dtype=int, minimum=0,
               source=__NAME__, user=True, active=True,
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
