#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
from typing import Union

import sossisse
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.instruments.default import Instrument

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
def main(**kwargs) -> Union[Instrument, None]:
    # ----------------------------------------------------------------------
    # deal with command line parameters - do not comment out this line
    # ----------------------------------------------------------------------
    inst = sossisse.get_parameters(__NAME__=__NAME__, **kwargs)

    # ----------------------------------------------------------------------
    # white light curve
    # ----------------------------------------------------------------------
    if inst.params['RUN.LINEAR_RECON']:
        inst = sossisse.linear_recon(inst)

    # ----------------------------------------------------------------------
    # spectral extraction
    # ----------------------------------------------------------------------
    if inst.params['RUN.SPECTRAL_EXTRACTION']:
        inst = sossisse.spectral_extraction(inst)

    # -------------------------------------------------------------------------
    # end script
    misc.end_recipe()
    # return the instrument object
    return inst


def run():
    """
    Wrapper script for command line - no return
    :return:
    """
    _ = main()


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    _ = main()

# =============================================================================
# End of code
# =============================================================================
