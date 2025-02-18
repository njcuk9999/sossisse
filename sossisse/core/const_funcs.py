#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant functionality

Created on 2024-08-13

@author: cook
"""
import sys
import json
import os
from typing import List, Union

import yaml

from aperocore.constants import load_functions
from aperocore.core import drs_log
from aperocore.constants import param_functions

from sossisse.core import base
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
# Get the parameter dictionary
ParamDict = param_functions.ParamDict
# get the logger
WLOG = drs_log.wlog
# Set the description of SOSSISSE
DESCRIPTIONS = dict()
DESCRIPTIONS['sossisse.recipes.run_sossisse'] = 'SOSSISSE - SOSS Inspired SpectroScopic Extraction'
DESCRIPTIONS['sossisse.recipes.run_setup'] = 'Setup up SOSSISSE directories/yaml file'

INPUTARGS = dict()
INPUTARGS['sossisse.recipes.run_setup'] = ['INPUTS.PARAM_FILE',
                                           'INPUTS.SOSSIOPATH',
                                           'INPUTS.OBJECTNAME',
                                           'INPUTS.INSTRUMENTMODE',
                                           'INPUTS.YAML_NAME',
                                           'INPUTS.ALL_CONSTANTS']
INPUTARGS['sossisse.recipes.run_sossisse'] = ['INPUTS.PARAM_FILE']


# list of constants to exlucde from hash
EXCLUDED_HASH_KEYS = ['SID']

# =============================================================================
# Define functions that use CDICT
# =============================================================================
def get_parameters(no_yaml: bool = False,
                   only_create: bool = False, log_level: str = None,
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
    # -------------------------------------------------------------------------
    # get the descriptions and inputs
    description = DESCRIPTIONS.get(kwargs['__NAME__'], 'UNKNOWN')
    inputs = INPUTARGS.get(kwargs['__NAME__'], None)
    # deal with yaml dict passed
    if '__YAML_DICT__' in kwargs:
        params = kwargs['__YAML_DICT__']
    else:
        # get parameters
        params = load_functions.get_all_params(name=__NAME__,
                                               description=description,
                                               inputargs=inputs,
                                               param_file_path='INPUTS.PARAM_FILE',
                                               config_list=[constants.CDict],
                                               from_file=not no_yaml,
                                               kwargs=kwargs)
    # ask user for any missing arguments
    params = load_functions.ask_for_missing_args(params)
    # -------------------------------------------------------------------------
    # deal with no param_file
    # -------------------------------------------------------------------------
    # get param file
    param_file = params['INPUTS.PARAM_FILE']
    # if no_yaml is True we get all arguments from kwargs
    if no_yaml:
        # create tmp dir
        tmp_path = os.path.expanduser('~/.sossisse/')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        # get some parameters for the param file
        _, _, rval = misc.unix_char_code()
        # add the filename to the tmp_path
        if params['INPUTS.YAML_NAME'] is None:
            tmp_path = os.path.join(tmp_path, f'params_{rval.lower()}.yaml')
        else:
            # make sure we have a yaml file
            if not params['INPUTS.YAML_NAME'].endswith('.yaml'):
                params['INPUTS.YAML_NAME'] += '.yaml'
            # create the tmp path
            tmp_path = os.path.join(tmp_path, params['INPUTS.YAML_NAME'])
        # re-create the yaml
        param_file = create_yaml(params, log=False, outpath=tmp_path)
    # otherwise we should display an error that we require a param file
    elif param_file is None:
        emsg = ('No parameter file defined - must be defined in '
                'command line/function kwargs')
        raise exceptions.SossisseFileException(emsg)
    else:
        tmp_path = os.path.realpath(param_file)
    # -------------------------------------------------------------------------
    # check if yaml file exists
    if not os.path.exists(param_file):
        emsg = f"Yaml file {param_file} does not exist"
        raise exceptions.SossisseFileException(emsg)
    # -------------------------------------------------------------------------
    # print that we are using yaml file
    if not no_yaml:
        misc.printc(f'\tUsing parameter file: {param_file}', msg_type='info')
    # -------------------------------------------------------------------------
    # deal with special parameters that need checking
    # -------------------------------------------------------------------------
    lm_params = params.get('WLC.LMODEL')
    # FIT_ZERO_POINT_OFFSET and FIT_QUAD_TERM cannot both be True
    if lm_params['FIT_ZERO_POINT_OFFSET'] and lm_params['FIT_QUAD_TERM']:
        emsg = 'Cannot have "FIT_ZERO_POINT_OFFSET" and "FIT_QUAD_TERM" true.'
        raise exceptions.SossisseConstantException(emsg)
    # -------------------------------------------------------------------------
    # force global log level to match
    if log_level is not None:
        misc.LOC_LEVEL = str(log_level).upper()
    else:
        misc.LOG_LEVEL = str(params['INPUTS.LOG_LEVEL']).upper()
    # -------------------------------------------------------------------------
    # finally add the param file to the params
    params['INPUTS.PARAM_FILE'] = os.path.abspath(param_file)
    params.set_source('INPUTS.PARAM_FILE', __NAME__)
    # get run time parameters (set in the code)
    params = run_time_params(params, only_create=only_create)
    # -------------------------------------------------------------------------
    # copy parameter file to other path
    # -------------------------------------------------------------------------
    if not only_create:
        param_file_basename = os.path.basename(param_file)
        param_file_csv = str(os.path.join(params['PATHS.OTHER_PATH'],
                                          param_file_basename))
        io.copy_file(param_file, param_file_csv)
    # -------------------------------------------------------------------------
    # re-create the yaml with updated parameters but at the new path
    if no_yaml:
        _ = create_yaml(params, log=False, outpath=tmp_path)
    # create the yaml file in the directory
    if only_create:
        outpath = str(os.path.join(params['PATHS.YAMLPATH'],
                                   os.path.basename(tmp_path)))
        _ = create_yaml(params, log=False, outpath=outpath)
        # update param file path
        params['INPUTS.PARAM_FILE'] = os.path.abspath(outpath)
    # -------------------------------------------------------------------------
    # create a copy of the yaml file in the object path
    _ = create_yaml(params, log=False)
    # -------------------------------------------------------------------------
    # create hash file (for quick check on SID
    create_hash(params)
    # -------------------------------------------------------------------------
    # now we load the instrument specific parameters
    instrument = load_instrument(params)
    # return the parameters
    return instrument



def run_time_params(params: ParamDict, only_create: bool = False
                    ) -> ParamDict:
    """
    Get run time parameters (set in the code)

    :param params: ParamDict, the parameters dictionary
    :param only_create: bool, if True only create directories and returns
                        does not do file operations
    :return:
    """
    # set the function name (for sources)
    func_name = f'{__NAME__}.run_time_params()'

    # get input parameters
    inputs = params.get('INPUTS')
    general = params.get('GENERAL')
    paths = params.get('PATHS')
    # -------------------------------------------------------------------------
    # we show or don't show the plots based on the user
    if not params['PLOTS.SHOW']:
        params['PLOTS.SHOW'] = os.getlogin() in params['PLOTS.USER_SHOW']
        params.set_source('PLOTS.SHOW', func_name)
    # -------------------------------------------------------------------------
    # set up core paths
    # -------------------------------------------------------------------------
    # lets create the sossiopath directory if it doesn't exist
    io.create_directory(inputs['SOSSIOPATH'])
    # -------------------------------------------------------------------------
    # all data for this instrument mode will be stored under this directory
    if paths['MODEPATH'] is None:
        paths['MODEPATH'] = os.path.join(inputs['SOSSIOPATH'],
                                         inputs['INSTRUMENTMODE'])
        paths.set_source('MODEPATH', func_name)
    # -------------------------------------------------------------------------
    # the calibration path is where we store all calibration files
    if paths['CALIBPATH'] is None:
        paths['CALIBPATH'] = os.path.join(paths['MODEPATH'], 'calibration')
        paths.set_source('CALIBPATH', func_name)
    io.create_directory(paths['CALIBPATH'])
    # -------------------------------------------------------------------------
    # the calibration path is where we store all calibration files
    if paths['YAMLPATH'] is None:
        paths['YAMLPATH'] = os.path.join(paths['MODEPATH'], 'yamls')
        paths.set_source('YAMLPATH', func_name)
    io.create_directory(paths['YAMLPATH'])
    # -------------------------------------------------------------------------
    # the raw path is where we store all the raw data
    if paths['RAWPATH'] is None:
        paths['RAWPATH'] = os.path.join(paths['MODEPATH'], inputs['OBJECTNAME'],
                                        'rawdata')
        paths.set_source('RAWPATH', func_name)
    io.create_directory(paths['RAWPATH'])
    # -------------------------------------------------------------------------
    # the object path is where we store all the object data
    #   note we add the sid to the path for multiple reductions
    if paths['OBJECTPATH'] is None:
        paths['OBJECTPATH'] = os.path.join(paths['MODEPATH'],
                                           inputs['OBJECTNAME'])
        paths.set_source('OBJECTPATH', func_name)
    io.create_directory(paths['OBJECTPATH'])
    # -------------------------------------------------------------------------
    # deal with the SID
    # -------------------------------------------------------------------------
    # get the sossisse unique id (sid) for this run
    if inputs['SID'] is None:
        # check whether we have a hash that matches the current yaml file
        # if so this gives us our SID
        sid = hash_match(params)
        # deal with having a yaml that matches a previous run
        if sid is None:
            inputs['SID'] = misc.sossice_unique_id(inputs['PARAM_FILE'])
            inputs.set_source('SID', func_name)
        else:
            inputs['SID'] = sid
            inputs.set_source('SID', func_name)
    # -------------------------------------------------------------------------
    # set up other paths
    # -------------------------------------------------------------------------
    # the object path is where we store all the object data
    #   note we add the sid to the path for multiple reductions
    if paths['SID_PATH'] is None:
        paths['SID_PATH'] = os.path.join(paths['OBJECTPATH'], inputs['SID'])
        paths.set_source('SID_PATH', func_name)
    io.create_directory(paths['SID_PATH'])
    # -------------------------------------------------------------------------
    # the temp path is where we store temporary versions of the raw data
    #   that have been opened and modified
    if paths['TEMP_PATH'] is None:
        paths['TEMP_PATH'] = os.path.join(paths['SID_PATH'], 'temporary')
        paths.set_source('TEMP_PATH', func_name)
    io.create_directory(paths['TEMP_PATH'])
    # -------------------------------------------------------------------------
    # the plot path
    if paths['PLOT_PATH'] is None:
        paths['PLOT_PATH'] = os.path.join(paths['SID_PATH'], 'plots')
        paths.set_source('PLOT_PATH', func_name)
    io.create_directory(paths['PLOT_PATH'])
    # -------------------------------------------------------------------------
    # the csv path
    if paths['OTHER_PATH'] is None:
        paths['OTHER_PATH'] = os.path.join(paths['SID_PATH'], 'other')
        paths.set_source('OTHER_PATH', func_name)
    io.create_directory(paths['OTHER_PATH'])
    # -------------------------------------------------------------------------
    # the fits paths
    if paths['FITS_PATH'] is None:
        paths['FITS_PATH'] = os.path.join(paths['SID_PATH'], 'fits')
        paths.set_source('FITS_PATH', func_name)
    io.create_directory(paths['FITS_PATH'])

    # -------------------------------------------------------------------------
    # load the raw files
    # -------------------------------------------------------------------------
    # deal with only creating directory - do not do this step
    if not only_create:
        # deal with no files
        if general['FILES'] is None:
            emsg = 'Must set FILES parameter in yaml file: {0}'
            eargs = [inputs['PARAM_FILE']]
            raise exceptions.SossisseFileException(emsg.format(*eargs))
        # get the list of input files
        basenames = list(general['FILES'])
        # loop around basenames, check they are on disk and convert to abs paths
        for b_it, basename in enumerate(basenames):
            # get the absolute path
            abspath = os.path.join(paths['RAWPATH'], basename)
            # print progress
            misc.printc(f'Checking file {abspath}', msg_type='debug')
            # check if the file exists
            if not os.path.exists(abspath):
                emsg = f'File {abspath} does not exist'
                raise exceptions.SossisseFileException(emsg)
            # print progress
            misc.printc(f'File {abspath} exists', msg_type='debug')
            # push into params
            general['FILES'][b_it] = abspath
    # -------------------------------------------------------------------------
    # set some file paths
    # -------------------------------------------------------------------------
    # deal with only creating directory - do not do this step
    if not only_create:
        # find the background file
        if general['BKGFILE'] is not None:
            absbkgfile = str(os.path.join(paths['CALIBPATH'],
                                          general['BKGFILE']))
            general['BKGFILE'] = io.get_file(absbkgfile, 'background')
            general.set_source('BKGFILE', func_name)
        # find the flat file
        if general['FLATFILE'] is not None:
            absflatfile = str(os.path.join(paths['CALIBPATH'],
                                           general['FLATFILE']))
            general['FLATFILE'] = io.get_file(absflatfile, 'flat')
            general.set_source('FLATFILE', func_name)
        # find the trace position file
        if general['POS_FILE'] is not None:
            absposfile = str(os.path.join(paths['CALIBPATH'],
                                          general['POS_FILE']))
            general['POS_FILE'] = io.get_file(absposfile, 'trace',
                                              required=False)
            general.set_source('POS_FILE', func_name)
        # deal with no background file given - other we use that the user set
        if general['BKGFILE'] is None:
            general['DO_BACKGROUND'] = False
            general.set_source('DO_BACKGROUND', func_name)
    # -------------------------------------------------------------------------
    # make sure sub-dicts are pushed back to params
    params['INPUTS'] = inputs
    params['GENERAL'] = general
    params['PATHS'] = paths
    # return the updated parameters
    return params


def create_yaml(params: ParamDict, log: bool = True,
                outpath: str = None) -> str:
    """
    Create a yaml file from input parameters

    :param params: Dict[str, Any], the input parameters
    :param log: bool, if True print log messages

    :return: None writes yaml file
    """
    # get the output path
    if outpath is None:
        outpath = os.path.join(params['PATHS.OTHER_PATH'],
                               'params_backup.yaml')
    # -------------------------------------------------------------------------
    # print progress
    if log:
        # print progress
        msg = 'Saving constants to yaml file: {0}'
        WLOG(params, '', msg.format(os.path.realpath(outpath)))
    # -------------------------------------------------------------------------
    # Get the constants dictionary
    cdict = constants.CDict
    # save the constants dictionary to yaml file
    cdict.save_yaml(params, outpath=outpath, log=log)
    # -------------------------------------------------------------------------
    # return the yaml file path
    return outpath


# =============================================================================
# Hash functions
# =============================================================================
def prearg_check(args: List[str]) -> bool:
    """
    Pre-argument check - check if any of the arguments are in the system
    arguments return True

    :param args: List[str], the arguments to check for

    :return:
    """
    # loop around args to check
    for arg in args:
        # loop around system arguments
        for sysarg in sys.argv[1:]:
            # if we find our argument return True
            if arg in sysarg:
                return True
    # if we get here return False
    return False


def create_hash(params: ParamDict):
    """
    Create a hash file for the current run

    :param params: dict, the parameters dictionary

    :return: None writes hashlist file
    """
    # get the hash file path
    hashpath = os.path.join(params['PATHS.OBJECTPATH'], 'hashlist.txt')
    # get the current SID
    sid = params['INPUTS.SID']
    # get the current yaml file path
    yaml_file = params['INPUTS.PARAM_FILE']
    # we load the yaml file
    with open(yaml_file, "r") as yamlfile:
        yaml_dict = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # remove SID from yaml_dict (we can't compare this)
    for key in EXCLUDED_HASH_KEYS:
        if key in yaml_dict:
            del yaml_dict[key]
    # create a jason string
    yaml_string = json.dumps(yaml_dict, sort_keys=True)
    # get the hash
    hashvalue = io.get_hash(yaml_string)
    # read the current hash list
    if os.path.exists(hashpath):
        with open(hashpath, "r") as hashfile:
            hashlist = hashfile.readlines()
    else:
        hashlist = []
    # add the new line to the hash
    hashlist.append(f'{sid} {hashvalue}\n')
    # wait for lock to release
    io.lock_wait(hashpath)
    # try to remove hash file
    try:
        # remove old hash file
        if os.path.exists(hashpath):
            os.remove(hashpath)
        # write the new hash file
        with open(hashpath, "w") as hashfile:
            hashfile.writelines(hashlist)
    finally:
        # unlock hash list file
        io.lock_wait(hashpath, unlock=True)


def hash_match(params: ParamDict) -> Union[str, None]:
    """
    Look for a match between the current yaml file and the hash list of
    previous runs SIDS and hashes

    :param params: Dict[str, Any], the input parameters

    :return: None if SID or hashlist.txt not found, otherwise returns the SID
    """
    # get the hash file path
    hashpath = os.path.join(params['PATHS.OBJECTPATH'], 'hashlist.txt')
    # get the current yaml file path
    yaml_file = params['INPUTS.PARAM_FILE']
    # if we don't have a current yaml file return
    if not os.path.exists(yaml_file):
        return None
    # we load the yaml file
    with open(yaml_file, "r") as yamlfile:
        yaml_dict = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # remove SID from yaml_dict (we can't compare this)
    if 'SID' in yaml_dict['INPUTS']:
        del yaml_dict['INPUTS']['SID']
    # create a jason string
    yaml_string = json.dumps(yaml_dict, sort_keys=True)
    # get the hash for this file
    hashvalue = io.get_hash(yaml_string)
    # if we don't have a hash file return None (this is a new run)
    if not os.path.exists(hashpath):
        return None
    else:
        # wait for lock to release
        io.lock_wait(hashpath)
        # try to read hash list file
        try:
            with open(hashpath, "r") as hashfile:
                hashlist = hashfile.readlines()
        finally:
            # unlock hash list file
            io.lock_wait(hashpath, unlock=True)
    # look for sid in the hast list
    for line in hashlist:
        sid, hashline = line.split(' ')
        if hashline == hashvalue:
            return sid
    # if we get to here we don't have a match --> return None
    return None


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
