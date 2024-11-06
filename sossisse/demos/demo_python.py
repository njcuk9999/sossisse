#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
import sossisse

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    # Here I just call the demo.yaml file but any parameter in the yaml file
    # can be put here, any other parameter can be
    param_file = 'demo.yaml'
    # any other keywords can be put in here
    pkwargs = dict()
    # pkwargs['SOSSIOPATH'] = '/path/to/data/demo/'

    # ----------------------------------------------------------------------
    # deal with parameters
    # ----------------------------------------------------------------------
    inst = sossisse.get_parameters(param_file=param_file, **pkwargs)

    # ----------------------------------------------------------------------
    # white light curve
    # ----------------------------------------------------------------------
    if inst.params['WHITE_LIGHT_CURVE']:
        sossisse.white_light_curve(inst)

    # ----------------------------------------------------------------------
    # spectral extraction
    # ----------------------------------------------------------------------
    if inst.params['SPECTRAL_EXTRACTION']:
        sossisse.spectral_extraction(inst)


# =============================================================================
# End of code
# =============================================================================
