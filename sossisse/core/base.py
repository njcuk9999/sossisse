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

# ANSI colors used only for rendered exception messages. Traceback frames are
# generated separately by Python and remain in the terminal's normal color.
EXCEPTION_RED = '\033[1;91;1m'
COLOR_END = '\033[0;0m'


def format_exception(message: str) -> str:
    """Render an exception message in bright red and reset terminal color."""
    return EXCEPTION_RED + message + COLOR_END

# supported instruments modes
# JWST NIRISS SOSS
INSTRUMENTS = ['JWST.NIRISS.SOSS',
               'JWST.NIRISS.FGS',
               'JWST.NIRSPEC.BOTS.PRISM',
               'JWST.NIRSPEC.BOTS.G395M',
               'JWST.NIRSPEC.BOTS.G235M',
               'JWST.NIRSPEC.BOTS.G140M',
               'JWST.NIRSPEC.BOTS.G395H',
               'JWST.NIRSPEC.BOTS.G235H',
               'JWST.NIRSPEC.BOTS.G140H']

# Required primary-header values for each supported observing mode.
MODE_HEADERS = {
    'JWST.NIRISS.SOSS': {
        'INSTRUME': ('NIRISS',), 'DETECTOR': ('NIS',),
        'EXP_TYPE': ('NIS_SOSS',), 'PUPIL': ('GR700XD',),
    },
    'JWST.NIRISS.FGS': {
        'INSTRUME': ('FGS',), 'EXP_TYPE': ('FGS_IMAGE', 'FGS_FOCUS'),
    },
    'JWST.NIRSPEC.BOTS.PRISM': {
        'INSTRUME': ('NIRSPEC',), 'DETECTOR': ('NRS1', 'NRS2'),
        'EXP_TYPE': ('NRS_BRIGHTOBJ',), 'GRATING': ('PRISM',),
    },
}
for grating in ['G395M', 'G235M', 'G140M', 'G395H', 'G235H', 'G140H']:
    mode = f'JWST.NIRSPEC.BOTS.{grating}'
    MODE_HEADERS[mode] = {
        'INSTRUME': ('NIRSPEC',), 'DETECTOR': ('NRS1', 'NRS2'),
        'EXP_TYPE': ('NRS_BRIGHTOBJ',), 'GRATING': (grating,),
    }


# =============================================================================
# Define functions
# =============================================================================

# =============================================================================
# End of code
# =============================================================================