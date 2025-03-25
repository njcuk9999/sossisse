#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.base.py

base variables here

Created on 2022-09-20

@author: cook

Rules: no sosssise imports
"""
from pathlib import Path
import yaml

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.base'

__PATH__ = Path(__file__).parent.parent
# load the yaml file
__YAML__ = yaml.load(open(__PATH__.joinpath('info.yaml')),
                     Loader=yaml.FullLoader)

# =============================================================================
# Get variables from info.yaml
# =============================================================================
__version__ = __YAML__['DRS.VERSION']
__authors__ = __YAML__['DRS.AUTHORS']
__date__ = __YAML__['DRS.DATE']
__release__ = __YAML__['DRS.RELEASE']

# Define basic types (non nested)
BASIC_TYPES = (int, float, bool, str)

# Define console width
CONSOLE_WIDTH = 120

# supported instruments modes
# JWST NIRISS SOSS
INSTRUMENTS = ['JWST.NIRISS.SOSS',
               'JWST.NIRISS.FGS',
               'JWST.NIRSPEC.PRISM',
               'JWST.NIRSPEC.G395',
               'JWST.NIRSPEC.G253',
               'JWST.NIRSPEC.G140']


# =============================================================================
# Define functions
# =============================================================================

# =============================================================================
# End of code
# =============================================================================