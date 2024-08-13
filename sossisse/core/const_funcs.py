#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant functionality

Created on 2024-08-13

@author: cook
"""
import argparse
import os
from typing import Any, Dict

import yaml

from sossisse.core import base
from sossisse.core import base_classes
from sossisse.core import constants
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.instruments import load_instrument, Instrument

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.constants'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Get constants class
Const = base_classes.Const
# Get constants dictionary
CDICT = constants.CDICT
# Set the description of SOSSISSE
DESCRIPTION = 'SOSSISSE - SOSS Inspired SpectroScopic Extraction'

# =============================================================================
# Define functions that use CDICT
# =============================================================================
def command_line_args() -> Dict[str, Any]:
    """
    Get command line arguments

    :return:
    """
    # start parser
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    # add arguments
    parser.add_argument('param_file', nargs='?', type=str, default=None,
                        action='store',
                        help='The parameter (yaml) file to use')

    # parse arguments
    args = parser.parse_args()
    # return arguments
    return vars(args)


def run_time_params(params: Dict[str, Any]) -> Dict[str, Any]:

    # get the sossisse unique id (sid) for this run
    params['SID'] = misc.sossice_unique_id(params['PARAM_FILE'])

    # we show or don't show the plots based on the user
    if not params['SHOW_PLOTS']:
        params['SHOW_PLOTS'] = os.getlogin() in params['USER_SHOW_PLOTS']

    # return the updated parameters
    return params


def get_parameters(param_file: str = None, **kwargs) -> Instrument:
    """
    Get the parameters from the constants module

    :return:
    """
    # we start with an empty parameter dictionary
    params = dict()
    sources = dict()
    # -------------------------------------------------------------------------
    # get command line arguments
    args = command_line_args()
    # see if param_file is in command line args
    if param_file is None and args['param_file'] is not None:
        param_file = args['param_file']
    # -------------------------------------------------------------------------
    # deal with no param_file
    if param_file is None:
        emsg = ('No parameter file defined - must be defined in '
                'command line/function kwargs')
        raise exceptions.SossisseFileException(emsg)
    # check if yaml file exists
    if not os.path.exists(param_file):
        emsg = (f"Yaml file {param_file} does not exist")
        raise exceptions.SossisseFileException(emsg)
    # finally add the param file to the params
    params['PARAM_FILE'] = os.path.abspath(param_file)
    sources['PARAM_FILE'] = __NAME__
    # -------------------------------------------------------------------------
    # we load the yaml file
    with open(param_file, "r") as yamlfile:
        yaml_dict = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # -------------------------------------------------------------------------
    # loop around CDICT, verify and push into params
    for key in CDICT:
        # get the constant
        const = CDICT[key]
        # ---------------------------------------------------------------------
        # get the value from the yaml file
        value, source = yaml_dict.get(key, None), param_file
        # ---------------------------------------------------------------------
        # Deal with a none or null value
        if str(value).upper() in ['NONE', 'NULL', '']:
            value, source = CDICT[key], 'constants.py'
        # ---------------------------------------------------------------------
        # parameters in args overwrite the yaml file
        if key in args and args[key] is not None:
            value, source = args[key], 'command line arguments'
        # ---------------------------------------------------------------------
        # parameters in kwargs overwrite the yaml file
        if key in kwargs and kwargs[key] is not None:
            value, source = kwargs[key], 'kwargs'
        # ---------------------------------------------------------------------
        # verify the constant
        const.verify(value=value, source=source)
        # push into params
        params[key] = const.value
        sources[key] = source
    # -------------------------------------------------------------------------
    # now we load the instrument specific parameters
    instrument = load_instrument(params)
    instrument.sources = sources
    # return the parameters
    return instrument



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
