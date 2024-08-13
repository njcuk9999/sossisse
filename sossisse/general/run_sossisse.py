#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
from sossisse.core import base
from sossisse.core import const_funcs
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.general import general

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.run_sossisse'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
def main(param_file: str = None, **kwargs):
    # ----------------------------------------------------------------------
    # deal with command line parameters - do not comment out this line
    # ----------------------------------------------------------------------
    try:
        inst = const_funcs.get_parameters(param_file=param_file, **kwargs)
    except exceptions.SossisseException as e:
        misc.printc(e.message, msg_type='bad2')
        return

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
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    _ = main()


# =============================================================================
# End of code
# =============================================================================
