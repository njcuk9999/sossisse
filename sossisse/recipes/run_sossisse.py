#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
import sossisse
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.recipes.run_sossisse'
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
        inst = sossisse.get_parameters(__NAME__=__NAME__,
                                       param_file=param_file,
                                       **kwargs)
    except exceptions.SossisseException as e:
        misc.printc(e.message, msg_type='error')
        return

    # ----------------------------------------------------------------------
    # white light curve
    # ----------------------------------------------------------------------
    if inst.params['WHITE_LIGHT_CURVE']:
        inst = sossisse.white_light_curve(inst)

    # ----------------------------------------------------------------------
    # spectral extraction
    # ----------------------------------------------------------------------
    if inst.params['SPECTRAL_EXTRACTION']:
        inst = sossisse.spectral_extraction(inst)


    return locals()

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    _ = main()

# =============================================================================
# End of code
# =============================================================================
