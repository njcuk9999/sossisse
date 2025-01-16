#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 12:50

@author: cook
"""
import sys
from typing import Union

import sossisse
from sossisse.core import const_funcs
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.instruments import select
from sossisse.instruments.default import Instrument


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.recipes.run_setup'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get the instrument modes
INSTRUMENTMODES = list(select.INSTRUMENTS.keys())


# =============================================================================
# Define functions
# =============================================================================
def main(**kwargs) -> Union[Instrument, None]:
    # ----------------------------------------------------------------------
    # deal with command line parameters - do not comment out this line
    # ----------------------------------------------------------------------
    try:
        inst = sossisse.get_parameters(__NAME__=__NAME__,
                                       no_yaml=True,
                                       only_create=True,
                                       log_level='setup',
                                       **kwargs)
    except exceptions.SossisseException as e:
        misc.printc(e.message, msg_type='error')
        return None
    # ----------------------------------------------------------------------
    # get log level
    misc.LOG_LEVEL = 'SETUP'
    # print message
    misc.printc('*' * 80, msg_type='setup')
    misc.printc('SOSSISSE SETUP', msg_type='setup')
    misc.printc('*' * 80 + '\n\n', msg_type='setup')
    # ----------------------------------------------------------------------
    # give user some instructions on what to do next
    # ----------------------------------------------------------------------
    msgs = ['Setup complete.\n\n']
    msgs += ['*' * 80 + '\n']
    msgs += ['What to do next:\n']
    msgs += ['*' * 80 + '\n']
    msgs += ['\n1. Open the yaml file: {0}'.format(inst.params['INPUTS.PARAM_FILE'])]
    msgs += ['\n2. Copy files to the correct directories:']
    msgs += ['\n\t - calibrations: {0}'.format(inst.params['PATHS.CALIBPATH'])]
    msgs += ['\n\t - raw data: {0}'.format(inst.params['PATHS.RAWPATH'])]
    msgs += ['\n3. Update the yaml parameters, for example:']
    msgs += ['\n\t - FILES (raw data dir)']
    msgs += ['\n\t - BKGFILE (calib dir, optional)']
    msgs += ['\n\t - FLATFILE (calib dir, optional)']
    msgs += ['\n3. Check every other value in the yaml file before running.']
    # print message
    for msg in msgs:
        misc.printc(msg, msg_type='setup')

    # end script
    misc.end_recipe()
    # return locals
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
