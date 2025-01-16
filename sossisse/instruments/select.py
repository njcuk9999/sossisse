#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:25

@author: cook
"""
from typing import Any, Dict

from aperocore.constants import param_functions

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.instruments import default
from sossisse.instruments import jwst_niriss
from sossisse.instruments import jwst_nirspec

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.select'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Get parameter dictionary
ParamDict = param_functions.ParamDict

# =============================================================================
# Define instruments
# =============================================================================
INSTRUMENTS = dict()
# -----------------------------------------------------------------------------
# add instrument classes here
# -----------------------------------------------------------------------------
# JWST NIRISS SOSS
INSTRUMENTS['JWST.NIRISS.SOSS'] = jwst_niriss.JWST_NIRISS_SOSS
# JWST NIRISS FGS
INSTRUMENTS['JWST.NIRISS.FGS'] = jwst_niriss.JWST_NIRISS_FGS
# JWST NIRSPEC PRISM
INSTRUMENTS['JWST.NIRSPEC.PRISM'] = jwst_nirspec.JWST_NIRSPEC_PRISM


# =============================================================================
# Define functions
# =============================================================================
def load_instrument(params: ParamDict) -> default.Instrument:
    """
    Load the instrument class

    :param params: dict, the parameters for the instrument
    :return: the instrument class
    """
    instrument_mode = params['INPUTS.INSTRUMENTMODE']
    # if we have the instrument return it
    if instrument_mode in INSTRUMENTS:
        return INSTRUMENTS[instrument_mode](params)
    else:
        emsg = 'Instrument mode "{0}" not recognised'
        eargs = [instrument_mode]
        raise exceptions.SossisseConstantException(emsg.format(*eargs))


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
