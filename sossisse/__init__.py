#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Only top level function should be defined here

i.e. those expected to be used in the notebooks

2024-08-20 9:26:10

@author: cook
"""
from sossisse.core import base
from sossisse.core import constants
from sossisse.core import const_funcs
from sossisse.general import general

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse'
__STRNAME__ = 'SOSSISSE'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# top level function for dealing with getting parameters
get_parameters = const_funcs.get_parameters

# top level function for running the white light curve step
linear_recon = general.linear_recon

# top level function for running the spectral extraction step
spectral_extraction = general.spectral_extraction

# Get the constants dictionary
CDict = constants.CDict

# =============================================================================
# End of code
# =============================================================================
