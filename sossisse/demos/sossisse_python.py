#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
from sossisse.core import const_funcs
from sossisse.general import general


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # deal with command line parameters - do not comment out this line
    # ----------------------------------------------------------------------
    inst = const_funcs.get_parameters()

    # ----------------------------------------------------------------------
    # white light curve
    # ----------------------------------------------------------------------
    if inst.params['WHITE_LIGHT_CURVE']:
        general.white_light_curve(inst)

    # ----------------------------------------------------------------------
    # spectral extraction
    # ----------------------------------------------------------------------
    if inst.params['SPECTRAL_EXTRACTION']:
        general.spectral_extraction(inst)


# =============================================================================
# End of code
# =============================================================================
