#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant functionality

Created on 2024-08-13

@author: cook
"""
import argparse
import os
from typing import Any, Dict, Tuple

import yaml

from sossisse.core import base
from sossisse.core import base_classes
from sossisse.core import constants
from sossisse.core import exceptions
from sossisse.core import io
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
DESCRIPTIONS = dict()
DESCRIPTIONS['sossisse.recipes.run_sossisse'] = 'SOSSISSE - SOSS Inspired SpectroScopic Extraction'
DESCRIPTIONS['sossisse.recipes.run_setup'] = 'Setup up SOSSISSE directories'


# =============================================================================
# Define functions that use CDICT
# =============================================================================
def command_line_args(name: str = None) -> Dict[str, Any]:
    """
    Get command line arguments

    :return:
    """
    # deal with no name
    if name is None:
        name = ''
    # if not a valid run recipe return empty dictionary
    if name not in DESCRIPTIONS:
        return dict()
    # get desription from descriptions
    description = DESCRIPTIONS[name]
    # start parser
    parser = argparse.ArgumentParser(description=description)
    # add arguments
    parser.add_argument('param_file', nargs='?', type=str, default=None,
                        action='store',
                        help='The parameter (yaml) file to use')
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return vars(args)


def run_time_params(params: Dict[str, Any],
                    sources: Dict[str, str],
                    only_create: bool = False
                    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Get run time parameters (set in the code)

    :param params: dict, the parameters dictionary
    :param sources: dict, the sources dictionary
    :param only_create: bool, if True only create directories and returns
                        does not do file operations
    :return:
    """
    # set the function name (for sources)
    func_name = f'{__NAME__}.run_time_params()'
    # get the sossisse unique id (sid) for this run
    if params['SID'] is None:
        params['SID'] = misc.sossice_unique_id(params['PARAM_FILE'])
        sources['SID'] = func_name

    # we show or don't show the plots based on the user
    if not params['SHOW_PLOTS']:
        params['SHOW_PLOTS'] = os.getlogin() in params['USER_SHOW_PLOTS']
        sources['SHOW_PLOTS'] = func_name

    # -------------------------------------------------------------------------
    # set up paths
    # -------------------------------------------------------------------------
    # lets create the sossiopath directory if it doesn't exist
    io.create_directory(params['SOSSIOPATH'])
    # -------------------------------------------------------------------------
    # all data for this instrument mode will be stored under this directory
    if params['MODEPATH'] is None:
        params['MODEPATH'] = os.path.join(params['SOSSIOPATH'],
                                          params['INSTRUMENTMODE'])
        sources['MODEPATH'] = func_name
        io.create_directory(params['MODEPATH'])
    # -------------------------------------------------------------------------
    # the calibration path is where we store all calibration files
    if params['CALIBPATH'] is None:
        params['CALIBPATH'] = os.path.join(params['MODEPATH'], 'calibration')
        sources['CALIBPATH'] = func_name
    io.create_directory(params['CALIBPATH'])
    # -------------------------------------------------------------------------
    # the raw path is where we store all the raw data
    if params['RAWPATH'] is None:
        params['RAWPATH'] = os.path.join(params['MODEPATH'], 'rawdata')
        sources['RAWPATH'] = func_name
    io.create_directory(params['RAWPATH'])
    # -------------------------------------------------------------------------
    # the object path is where we store all the object data
    #   note we add the sid to the path for multiple reductions
    if params['OBJECTPATH'] is None:
        params['OBJECTPATH'] = os.path.join(params['MODEPATH'],
                                            params['OBJECTNAME'],
                                            params['SID'])
        sources['OBJECTPATH'] = func_name
    io.create_directory(params['OBJECTPATH'])
    # -------------------------------------------------------------------------
    # the temp path is where we store temporary versions of the raw data
    #   that have been opened and modified
    if params['TEMP_PATH'] is None:
        params['TEMP_PATH'] = os.path.join(params['OBJECTPATH'], 'temporary')
        sources['TEMP_PATH'] = func_name
    io.create_directory(params['TEMP_PATH'])
    # -------------------------------------------------------------------------
    # the plot path
    if params['PLOT_PATH'] is None:
        params['PLOT_PATH'] = os.path.join(params['OBJECTPATH'], 'plots')
        sources['PLOT_PATH'] = func_name
    io.create_directory(params['PLOT_PATH'])
    # -------------------------------------------------------------------------
    # the csv path
    if params['CSV_PATH'] is None:
        params['CSV_PATH'] = os.path.join(params['OBJECTPATH'], 'csv')
        sources['CSV_PATH'] = func_name
    io.create_directory(params['CSV_PATH'])
    # -------------------------------------------------------------------------
    # the fits paths
    if params['FITS_PATH'] is None:
        params['FITS_PATH'] = os.path.join(params['OBJECTPATH'], 'fits')
        sources['FITS_PATH'] = func_name
    io.create_directory(params['FITS_PATH'])
    # -------------------------------------------------------------------------
    # load the raw files
    # -------------------------------------------------------------------------
    # deal with only creating directory - do not do this step
    if not only_create:
        # get the list of input files
        basenames = list(params['FILES'])
        # loop around basenames, check they are on disk and convert to abs paths
        for b_it, basename in enumerate(basenames):
            # get the absolute path
            abspath = os.path.join(params['RAWPATH'], basename)
            # print progress
            misc.printc(f'Checking file {abspath}', msg_type='debug')
            # check if the file exists
            if not os.path.exists(abspath):
                emsg = f'File {abspath} does not exist'
                raise exceptions.SossisseFileException(emsg)
            # print progress
            misc.printc(f'File {abspath} exists', msg_type='debug')
            # push into params
            params['FILES'][b_it] = abspath
    # -------------------------------------------------------------------------
    # set some file paths
    # -------------------------------------------------------------------------
    # deal with only creating directory - do not do this step
    if not only_create:
        # find the background file
        absbkgfile = str(os.path.join(params['CALIBPATH'], params['BKGFILE']))
        params['BKGFILE'] = io.get_file(absbkgfile, 'background')
        sources['BKGFILE'] = func_name
        # find the flat file
        absflatfile = str(os.path.join(params['CALIBPATH'], params['FLATFILE']))
        params['FLATFILE'] = io.get_file(absflatfile, 'flat')
        sources['FLATFILE'] = func_name
        # find the trace position file
        absposfile = str(os.path.join(params['CALIBPATH'], params['POS_FILE']))
        params['POS_FILE'] = io.get_file(absposfile, 'trace')
        sources['POS_FILE'] = func_name

    # return the updated parameters
    return params, sources


def get_parameters(param_file: str = None, no_yaml: bool = False,
                   only_create: bool = False,
                   **kwargs) -> Instrument:
    """
    Get the parameters from the constants module

    :param param_file: str, the parameter file to use (yaml file) if None
                       must set no_yaml to True and provide all required
                       arguments via kwargs
    :param no_yaml: bool, if True we do not use a yaml file and the user must
                    provide all required arguments via kwargs
    :param only_create: bool, if True only create directories (not file
                        operations)

    :param kwargs: any additional keyword arguments

    :return: Instrument, the correct instrument class with all parameters
    """
    # print splash
    misc.sossart()
    # print progress
    misc.printc('Getting parameters', msg_type='info')
    # we start with an empty parameter dictionary
    params = dict()
    sources = dict()
    # -------------------------------------------------------------------------
    # get command line arguments
    args = command_line_args(kwargs.get('__NAME__', None))

    # see if param_file is in command line args
    if param_file is None and args['param_file'] is not None:
        param_file = args['param_file']
    # -------------------------------------------------------------------------
    # deal with no param_file
    # -------------------------------------------------------------------------
    # if no_yaml is True we get all arguments from kwargs
    if no_yaml:
        # loop around CDICT, verify and push into params
        for key in CDICT:
            # get the constant
            const = CDICT[key]
            # -----------------------------------------------------------------
            # parameters in kwargs overwrite the yaml file
            if key in kwargs and kwargs[key] is not None:
                value, source = kwargs[key], 'kwargs'

            else:
                value, source = None, None
            # verify the constant - this will raise an exception if the value
            # is not in kwargs and is required
            const.verify(value=value, source=source)
            # push into params
            params[key] = const.value
            sources[key] = source
    # otherwise we should display an error that we require a param file
    elif param_file is None:
        emsg = ('No parameter file defined - must be defined in '
                'command line/function kwargs')
        raise exceptions.SossisseFileException(emsg)
    # check if yaml file exists
    if not os.path.exists(param_file):
        emsg = f"Yaml file {param_file} does not exist"
        raise exceptions.SossisseFileException(emsg)
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
            value, source = CDICT[key].value, 'constants.py'
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
    # deal with special parameters that need checking
    # -------------------------------------------------------------------------
    # FIT_ZERO_POINT_OFFSET and FIT_QUAD_TERM cannot both be True
    if params['FIT_ZERO_POINT_OFFSET'] and params['FIT_QUAD_TERM']:
        emsg = 'Cannot have "FIT_ZERO_POINT_OFFSET" and "FIT_QUAD_TERM" true.'
        raise exceptions.SossisseConstantException(emsg)
    # -------------------------------------------------------------------------
    # force global log level to match
    misc.LOG_LEVEL = str(params['LOG_LEVEL']).upper()
    # -------------------------------------------------------------------------
    # finally add the param file to the params
    params['PARAM_FILE'] = os.path.abspath(param_file)
    sources['PARAM_FILE'] = __NAME__
    # get run time parameters (set in the code)
    params, sources = run_time_params(params, sources, only_create=only_create)
    # -------------------------------------------------------------------------
    # copy parameter file to csv path
    # -------------------------------------------------------------------------
    if not only_create:
        param_file_basename = os.path.basename(param_file)
        param_file_csv = str(os.path.join(params['CSV_PATH'],
                                          param_file_basename))
        io.copy_file(param_file, param_file_csv)
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