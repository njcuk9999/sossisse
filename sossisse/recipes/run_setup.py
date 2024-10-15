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
def main(param_file: str = None, **kwargs) -> Union[Instrument, None]:
    # get log level
    misc.LOG_LEVEL = 'SETUP'
    # print message
    misc.printc('*' * 80, msg_type='setup')
    misc.printc('SOSSISSE SETUP', msg_type='setup')
    misc.printc('*' * 80 + '\n\n', msg_type='setup')
    # ----------------------------------------------------------------------
    # Ask user for required parameters
    # ----------------------------------------------------------------------
    kwargs['SOSSIOPATH'] = misc.get_input('SOSSIOPATH', dtype='dir',
                                          comment='the path to store data in')
    kwargs['OBJECTNAME'] = misc.get_input('OBJECTNAME',
                                          comment='the name of the object '
                                                  'directory')
    kwargs['INSTRUMENTMODE'] = misc.get_input('INSTRUMENTMODE',
                                              comment='the instrument mode',
                                              options=INSTRUMENTMODES)
    # define the prompt for all constants
    allq = input('Do you want all constants for all modes/settings in the yaml '
                 'file?\n[Y]es or [N]o >> ')
    # deal with user response to prompt
    if 'Y' in str(allq).upper():
        kwargs['ALL_CONSTANTS'] = True
    else:
        kwargs['ALL_CONSTANTS'] = False
    # fake files to empty (this is a setup we don't ask for files here)
    kwargs['FILES'] = []
    # ----------------------------------------------------------------------
    # deal with command line parameters - do not comment out this line
    # ----------------------------------------------------------------------
    try:
        inst = sossisse.get_parameters(__NAME__=__NAME__,
                                       param_file=param_file,
                                       no_yaml=True,
                                       only_create=True,
                                       LOG_LEVEL='setup',
                                       **kwargs)
    except exceptions.SossisseException as e:
        misc.printc(e.message, msg_type='error')
        return None
    # ----------------------------------------------------------------------
    # give user some instructions on what to do next
    # ----------------------------------------------------------------------
    msgs = ['Setup complete.\n\n']
    msgs += ['*' * 80 + '\n']
    msgs += ['What to do next:\n']
    msgs += ['*' * 80 + '\n']
    msgs += ['\n1. Open the yaml file: {0}'.format(inst.params['PARAM_FILE'])]
    msgs += ['\n2. Copy files to the correct directories:']
    msgs += ['\n\t - calibrations: {0}'.format(inst.params['CALIBPATH'])]
    msgs += ['\n\t - raw data: {0}'.format(inst.params['RAWPATH'])]
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
