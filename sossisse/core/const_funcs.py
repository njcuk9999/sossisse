#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constant functionality

Created on 2024-08-13

@author: cook
"""
import os
import sys
from typing import List

from aperocore.constants import load_functions
from aperocore.constants.param_functions import ParamDict
from aperocore.core import drs_log
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
EXCLUDED_HASH_KEYS = ['SUBDIRECTORY']

# =============================================================================
# Define functions that use CDICT
# =============================================================================
def get_parameters(no_yaml: bool = False,
                   only_create: bool = False, log_level: str = None,
                   setup_mode: bool = False, **kwargs) -> Instrument:
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
    # set function name
    func_name = __NAME__ + '.get_parameters()'
    # print splash
    if log_level != 'setup':
        misc.sossart()
    # in setup mode we force some parameters
    if setup_mode:
        no_yaml = True
        only_create = True
        log_level = 'setup'
    # -------------------------------------------------------------------------
    # get the descriptions and inputs
    description = DESCRIPTIONS.get(kwargs['__NAME__'], 'UNKNOWN')
    inputs = INPUTARGS.get(kwargs['__NAME__'], None)
    # deal with yaml dict passed
    if '__YAML_DICT__' in kwargs:
        _params = kwargs['__YAML_DICT__']
        _params.set('__SOURCE__', 'POGOS', source=func_name, instance=None)
        params = _params.as_param_dict()
    else:
        # get parameters
        params = load_functions.get_all_params(name=__NAME__,
                                               description=description,
                                               inputargs=inputs,
                                               param_file_path='INPUTS.PARAM_FILE',
                                               config_list=[constants.CDict],
                                               from_file=not no_yaml,
                                               kwargs=kwargs)
        # set source from SOSSISSE
        params.set('__SOURCE__', 'SOSSISSE', source=func_name)
    # -------------------------------------------------------------------------
    # deal with start point (setup only)
    if setup_mode:
        # only load this if we are in setup mode
        from sossisse.resources import demos as demo_mod
        # ask user to start from demo or blank
        params = load_functions.starting_point(params, 'INPUTS.INSTRUMENTMODE',
                                               demo_mod)
    # -------------------------------------------------------------------------
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
        param_file = create_yaml(params, log=False, outpath=tmp_path,
                                 force=True)
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
    # deal with downloading the data (setup mode only)
    if setup_mode:
        # get the demo local value
        demolocal = params.get('GLOBAL.LOCAL_DOWNLOAD_DATA', None)
        demosymlink = params.get('GLOBAL.DEMO_SYMLINK_DATA', False)
        # ask for data download
        load_functions.download_data(params, demolocal=demolocal,
                                     demosymlink=demosymlink)
    # -------------------------------------------------------------------------
    # copy pos file to FITS path
    if params['GENERAL.POS_FILE'] is not None:
        posfile = params['GENERAL.POS_FILE']
        posfile_basename = os.path.basename(posfile)
        posfile_out = str(os.path.join(params['PATHS.FITS_PATH'],
                                       posfile_basename))
        # only copy pos file if it, exists (otherwise we just update the
        # filename - this is fine as we can create this file sometimes)
        if os.path.exists(posfile):
            io.copy_file(posfile, posfile_out)
        # update pos file path
        params['GENERAL.POS_FILE'] = os.path.abspath(posfile_out)
        params['GENERAL'].set_source('POS_FILE', func_name)

    # -------------------------------------------------------------------------
    # If we didn't have a yaml to start with create it now
    # re-create the yaml with updated parameters but at the new path
    if no_yaml:
        _ = create_yaml(params, log=False, outpath=tmp_path)
    # -------------------------------------------------------------------------
    # copy parameter file to other path (as a record of what was actually used)
    #  we only do this if we are running data (i.e. only_create is False)
    # -------------------------------------------------------------------------
    if not only_create:
        # we use the tmp file path to create the backup name
        param_bname = os.path.basename(tmp_path)
        # remove the yaml ending if it exists
        if param_bname.endswith('.yaml'):
            param_bname = param_bname[:-len('.yaml')]
        # Backup file should be:
        #   SOSSISSE_<input yaml basename>_asrun.yaml
        #   POGOS_<input yaml basename>_asrun.yaml
        param_oname = '{0}_{1}_asrun.yaml'
        pbargs = [params['__SOURCE__'], param_bname]
        param_oname = param_oname.format(*pbargs)
        param_file_csv = os.path.join(params['PATHS.OTHER_PATH'], param_oname)
        # copy file
        io.copy_file(param_file, str(param_file_csv))
    # -------------------------------------------------------------------------
    # If we are creating and running SOSSISSE we need to create the yaml at
    #  its proper path
    # -------------------------------------------------------------------------
    # create the yaml file in the directory for SOSSISSE
    else:
        if params['__SOURCE__'] == 'SOSSISSE':
            outpath = str(os.path.join(params['PATHS.YAMLPATH'],
                                       os.path.basename(tmp_path)))
            _ = create_yaml(params, log=False, outpath=outpath)
            # update param file path
            params['INPUTS.PARAM_FILE'] = os.path.abspath(outpath)
            # remove the previous tmp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
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
        username = misc.safe_getuser()
        params['PLOTS.SHOW'] = username in params['PLOTS.USER_SHOW']
        params.set_source('PLOTS.SHOW', func_name)
    # -------------------------------------------------------------------------
    # set up core paths
    # -------------------------------------------------------------------------
    # lets create the sossiopath directory if it doesn't exist
    io.create_directory(inputs['SOSSIOPATH'])
    # -------------------------------------------------------------------------
    # the calibration path is where we store all calibration files
    if params['__SOURCE__'] == 'SOSSISSE':
        if paths['YAMLPATH'] is None:
            paths['YAMLPATH'] = os.path.join(paths['MODEPATH'], 'yamls')
            paths.set_source('YAMLPATH', func_name)
        io.create_directory(paths['YAMLPATH'])
    # -------------------------------------------------------------------------
    # deal with the SUBDIRECTORY
    # -------------------------------------------------------------------------
    # get the sossisse unique id (sid) for this run
    if inputs['SUBDIRECTORY'] is None:
        sid = misc.sossice_unique_id(inputs['PARAM_FILE'])
        imode = inputs['INSTRUMENTMODE']
        oname = inputs['OBJECTNAME']
        inputs['SUBDIRECTORY'] = f'{imode}_{oname}_{sid}'
        inputs.set_source('SUBDIRECTORY', func_name)
    # -------------------------------------------------------------------------
    # set up other paths
    # -------------------------------------------------------------------------
    # the object path is where we store all the object data
    #   note we add the sid to the path for multiple reductions
    if paths['SUBDIRECTORY_PATH'] is None:
        paths['SUBDIRECTORY_PATH'] = os.path.join(inputs['SOSSIOPATH'],
                                                  inputs['SUBDIRECTORY'])
        paths.set_source('SUBDIRECTORY_PATH', func_name)
    io.create_directory(paths['SUBDIRECTORY_PATH'])
    # -------------------------------------------------------------------------
    # the raw path is where we store all the raw data
    if paths['RAWPATH'] is None:
        paths['RAWPATH'] = os.path.join(paths['SUBDIRECTORY_PATH'], 'inputs')
        paths.set_source('RAWPATH', func_name)
    io.create_directory(paths['RAWPATH'])
    # -------------------------------------------------------------------------
    # the calibration path is where we store all calibration files
    if paths['CALIBPATH'] is None:
        paths['CALIBPATH'] = os.path.join(paths['SUBDIRECTORY_PATH'],
                                          'calibration')
        paths.set_source('CALIBPATH', func_name)
    io.create_directory(paths['CALIBPATH'])
    # -------------------------------------------------------------------------
    # the temp path is where we store temporary versions of the raw data
    #   that have been opened and modified
    if paths['TEMP_PATH'] is None:
        paths['TEMP_PATH'] = os.path.join(paths['SUBDIRECTORY_PATH'],
                                          'temporary')
        paths.set_source('TEMP_PATH', func_name)
    io.create_directory(paths['TEMP_PATH'])
    # -------------------------------------------------------------------------
    # the plot path
    if paths['PLOT_PATH'] is None:
        paths['PLOT_PATH'] = os.path.join(paths['SUBDIRECTORY_PATH'], 'plots')
        paths.set_source('PLOT_PATH', func_name)
    io.create_directory(paths['PLOT_PATH'])
    # -------------------------------------------------------------------------
    # the csv path
    if paths['OTHER_PATH'] is None:
        paths['OTHER_PATH'] = os.path.join(paths['SUBDIRECTORY_PATH'], 'other')
        paths.set_source('OTHER_PATH', func_name)
    io.create_directory(paths['OTHER_PATH'])
    # -------------------------------------------------------------------------
    # the fits paths
    if paths['FITS_PATH'] is None:
        paths['FITS_PATH'] = os.path.join(paths['SUBDIRECTORY_PATH'], 'fits')
        paths.set_source('FITS_PATH', func_name)
    io.create_directory(paths['FITS_PATH'])
    # -------------------------------------------------------------------------
    # the out paths
    if paths['OUT_PATH'] is None:
        paths['OUT_PATH'] = os.path.join(paths['SUBDIRECTORY_PATH'], 'outputs')
        paths.set_source('OUT_PATH', func_name)
    io.create_directory(paths['OUT_PATH'])
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
        # deal with creating a common file prefix
        general['PREFIX'] = load_functions.common_prefix(general['FILES'])
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

        # ---------------------------------------------------------------------
        # find the trace position file
        if general['POS_FILE'] is not None:
            absposfile = str(os.path.join(paths['CALIBPATH'],
                                          general['POS_FILE']))
        # if no pos file was given we create it
        else:
            wmsg = f'No POS_FILE set, creating pos_file.fits'
            misc.printc(wmsg, msg_type='warning')
            absposfile = str(os.path.join(paths['CALIBPATH'], 'pos_file.fits'))
        # update POS_FILE in parameter dictionary
        general['POS_FILE'] = io.get_file(absposfile, 'trace',
                                          required=False)
        general.set_source('POS_FILE', func_name)
        # ---------------------------------------------------------------------
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
                outpath: str = None, force: bool = False) -> str:
    """
    Create a yaml file from input parameters

    :param params: Dict[str, Any], the input parameters
    :param log: bool, if True print log messages

    :return: None writes yaml file
    """
    # get the output path
    if outpath is None:
        if params['__SOURCE__'] == 'POGOS':
            if force:
                outpath = os.path.join(params['PATHS.OTHER_PATH'],
                                       'params_backup_sossisse.yaml')
            else:
                return ''
        else:
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
    cdict.save_yaml(params, outpath=outpath, log=log,
                    mode=params['INSTRUMENTMODE'])
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
