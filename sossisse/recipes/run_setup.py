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
from sossisse.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.recipes.run_setup'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get the instrument modes
INSTRUMENTMODES = select.INSTRUMENTS.keys()


# =============================================================================
# Define functions
# =============================================================================
def main(param_file: str = None, **kwargs):
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
                 'file? [Y]es or [N]o:\t')
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
        return
    # ----------------------------------------------------------------------
    # give user some instructions on what to do next
    # ----------------------------------------------------------------------
    msg = 'Setup complete.\n\n'
    msg += '*' * 80 + '\n'
    msg += 'What to do next:\n'
    msg += '*' * 80 + '\n'
    msg += '1. Edit the yaml file (at the very least add "FILES")'
    msg += '2. Check every other value in the yaml file before running.'
    # print message
    misc.printc(msg, msg_type='setup')
    # return locals
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
