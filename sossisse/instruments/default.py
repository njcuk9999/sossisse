#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import copy
import os
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
from tqdm import tqdm
from wpca import EMPCA

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.core import math as mp
from sossisse.core import misc
from sossisse.general import plots

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.default'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define instrument class
# =============================================================================
class Instrument:
    """
    Base instrument class - must have all methods given in
    any child class
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        self.name = 'Default'
        # set the parameters from input
        self.params = copy.deepcopy(params)
        # set all sources to Unknown
        self.sources = {key: 'Unknown' for key in self.params}
        # set up the instrument
        self.param_override()
        # variables to keep in memory and pass around with class
        #  to keep this in order they must be defined here
        #  with a source function (given to the user if run before setting)
        self._variables = dict()
        # array sizes
        self._variables['DATA_X_SIZE'] = None
        self._variables['DATA_Y_SIZE'] = None
        self._variables['DATA_N_FRAMES'] = None
        # temp files
        self._variables['TEMP_INI_CUBE'] = None
        self._variables['TEMP_INI_ERR'] = None
        self._variables['TEMP_CLEAN_NAN'] = None
        self._variables['MEIDAN_IMAGE_FILE'] = None
        self._variables['CLEAN_CUBE_FILE'] = None
        self._variables['TEMP_BEFORE_AFTER_CLEAN1F'] = None
        self._variables['TEMP_PCA_FILE'] = None
        self._variables['TEMP_TRANSIT_IN_VS_OUT'] = None
        self._variables['WLC_ERR_FILE'] = None
        self._variables['WLC_RES_FILE'] = None
        self._variables['WLC_RECON_FILE'] = None
        self._variables['WLC_LTBL_FILE'] = None
        self._variables['SPE_SED_TBL'] = None
        # true/false flags
        self._variables['DO_BACKGROUND'] = None
        self._variables['FLAG_CDS'] = None
        self._variables['HAS_OOT'] = None
        # meta data
        self._variables['TAG1'] = None
        self._variables['TAG2'] = None
        self._variables['META'] = None
        self._variables['OUTPUT_NAMES'] = None
        self._variables['OUTPUT_UNITS'] = None
        self._variables['OUTPUT_FACTOR'] = None
        # simple vectors
        self._variables['OOT_DOMAIN'] = None
        self._variables['OOT_DOMAIN_BEFORE'] = None
        self._variables['OOT_DOMAIN_AFTER'] = None
        self._variables['INT_DOMAIN'] = None
        # ---------------------------------------------------------------------
        # define source for variables
        self.vsources = dict()
        # array sizes
        self.vsources['DATA_X_SIZE'] = f'{self.name}.load_data_with_dq()'
        self.vsources['DATA_Y_SIZE'] = f'{self.name}.load_data_with_dq()'
        self.vsources['DATA_N_FRAMES'] = f'{self.name}.load_data_with_dq()'
        # construct temporary file name function
        tmp_func = f'{self.name}.temporary_filenames()'
        # temp files
        self.vsources['TEMP_INI_CUBE'] = tmp_func
        self.vsources['TEMP_INI_ERR'] = tmp_func
        self.vsources['TEMP_CLEAN_NAN'] = tmp_func
        self.vsources['MEIDAN_IMAGE_FILE'] = tmp_func
        self.vsources['CLEAN_CUBE_FILE'] = tmp_func
        self.vsources['TEMP_BEFORE_AFTER_CLEAN1F'] =tmp_func
        self.vsources['TEMP_PCA_FILE'] = tmp_func
        self.vsources['TEMP_TRANSIT_IN_VS_OUT'] = tmp_func
        self.vsources['WLC_ERR_FILE'] = tmp_func
        self.vsources['WLC_RES_FILE'] = tmp_func
        self.vsources['WLC_RECON_FILE'] = tmp_func
        self.vsources['WLC_LTBL_FILE'] = tmp_func
        self.vsources['SPE_SED_TBL'] = tmp_func
        # true/false flags
        self.vsources['DO_BACKGROUND'] = f'{self.name}.remove_background()'
        self.vsources['FLAG_CDS'] = f'{self.name}.load_data_with_dq()'
        self.vsources['HAS_OOT'] = f'{self.name}.get_valid_oot()'
        # meta data
        self.vsources['TAG1'] = f'{self.name}.update_meta_data()'
        self.vsources['TAG2'] = f'{self.name}.update_meta_data()'
        self.vsources['META'] = f'{self.name}.update_meta_data()'
        self.vsources['OUTPUT_NAMES'] = f'{self.name}.setup_linear_reconstruction()'
        self.vsources['OUTPUT_UNITS'] = f'{self.name}.setup_linear_reconstruction()'
        self.vsources['OUTPUT_FACTOR'] = f'{self.name}.setup_linear_reconstruction()'
        # simple vectors
        self.vsources['OOT_DOMAIN'] = f'{self.name}.get_valid_oot()'
        self.vsources['OOT_DOMAIN_BEFORE'] = f'{self.name}.get_valid_oot()'
        self.vsources['OOT_DOMAIN_AFTER'] = f'{self.name}.get_valid_oot()'
        self.vsources['INT_DOMAIN'] = f'{self.name}.get_valid_int()'

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        pass

    def set_variable(self, key: str, value: Any, func_name: str = None):
        """
        Set a variable in the variables dictionary

        :param key: str, the key to set
        :param value: any, the value to set
        :param func_name: set the function name (should be set but can be
                          updated)
        :return:
        """
        if key in self._variables and key in self.vsources:
            self._variables[key] = value
            # push the function name if given
            if func_name is not None:
                self.vsources[key] = func_name
        else:
            emsg = ('Key {0} not found in variables dictionary.'
                    'Please set it in Instrument class')
            raise exceptions.SossisseInstException(emsg.format(key), self.name)

    def get_variable(self, key: str, func_name: str) -> Any:
        """
        Get a variable from the variables dictionary

        :param key: str, the key to get
        :param func_name: str, the name of the function calling this function
        :return: any, the variable
        """
        # deal with key not in variables
        if key not in self._variables:
            emsg = ('Variable {0} not found in variables dictionary.'
                    'Please set it in Instrument class')
            raise exceptions.SossisseInstException(emsg.format(key), self.name)
        # deal with key not set (must run another function first)
        if self._variables[key] is None:
            emsg = ('Variable {0} not set. '
                    'Please run {0} before running {1}.')
            eargs = [key, self.vsources[key], func_name]
            raise exceptions.SossisseInstException(emsg.format(*eargs),
                                                   self.name)
        # otherwise return the variable
        return self._variables[key]

    def update_meta_data(self):
        """
        Update the meta data dictionary and the tag1 and tag2 for files

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.update_meta_data()'
        # set up storage
        meta_data = dict()
        tag1, tag2 = '', ''

        # ---------------------------------------------------------------------
        # deal with CDS data
        # ---------------------------------------------------------------------
        if self.get_variable('FLAG_CDS', func_name) is not None:
            # add to tag 1
            tag1 += '_cds-{0}-{1}'.format(*self.params['CDS_IDS'])
            # add to meta data
            meta_data['CDS'] = (True, 'Whether data is calculated from a CDS')
            meta_data['CDS_FIRST'] = (self.params['CDS_IDS'][0],
                                      'First frame of CDS')
            meta_data['CDS_LAST'] = (self.params['CDS_IDS'][1],
                                     'Last frame of CDS')
        # ---------------------------------------------------------------------
        # deal with wlc domain
        # ---------------------------------------------------------------------
        if self.params['WLC_DOMAIN'] is not None:
            # add to tag 1
            tag1 += '_wlcdomain-{0}-{1}um'.format(*self.params['WLC_DOMAIN'])
            # add to meta data
            meta_data['WLC_DOM0'] = (self.params['WLC_DOMAIN'][0],
                                     'Wavelength domain start [um]')
            meta_data['WLC_DOM1'] = (self.params['WLC_DOMAIN'][1],
                                     'Wavelength domain end [um]')
        # ---------------------------------------------------------------------
        # deal with median out of transit
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_ootmed{0}'.format(int(self.params['MEDIAN_OOT']))
        # add to meta data
        meta_data['OOTMED'] = (self.params['MEDIAN_OOT'],
                               'Median out of transit')
        # ---------------------------------------------------------------------
        # deal with background correction
        # ---------------------------------------------------------------------
        # deal with DO_BACKGROUND set
        if self._variables['DO_BACKGROUND'] is not None:
            # get variable
            do_background = self.get_variable('DO_BACKGROUND', func_name)
            # add to tag 1
            tag1 += '_bkg{0}'.format(int(do_background))
            # add to meta data
            meta_data['DO_BKG'] = (int(do_background),
                                   'Background was removed')

        # ---------------------------------------------------------------------
        # deal with degree of the polynomial for the 1/f correction
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_1fpolyord{0}'.format(self.params['DEGREE_1F_CORR'])
        # add to meta data
        meta_data['DEG1FCORR'] = (self.params['DEGREE_1F_CORR'],
                                  'Degree of polynomial for 1/f correction')
        # ---------------------------------------------------------------------
        # deal with whether we fit the rotation in the linear model
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_fitrot{0}'.format(int(self.params['FIT_ROTATION']))
        # add to meta data
        meta_data['FITROT'] = (self.params['FIT_ROTATION'],
                               'Rotation was fitted in linear model')
        # ---------------------------------------------------------------------
        # deal with whether we fit the zero point offset in the linear model
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_fitzp{0}'.format(int(self.params['FIT_ZERO_POINT_OFFSET']))
        # add to meta data
        meta_data['FITZP'] = (self.params['FIT_ZERO_POINT_OFFSET'],
                              'Zero point offset was fitted in linear model')
        # ---------------------------------------------------------------------
        # deal with whether we fit the 2nd derivative in y
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_fitddy{0}'.format(int(self.params['FIT_DDY']))
        # add to meta data
        meta_data['FITDDY'] = (self.params['FIT_DDY'],
                               '2nd derivative was fitted in y')
        # ---------------------------------------------------------------------
        # add the transit points
        # ---------------------------------------------------------------------
        # add to tag 1
        tag1 += '_it1{0}-it4{1}'.format(*self.params['CONTACT_FRAMES'])
        tag2 += tag1 + '_it2{0}-it3{1}'.format(*self.params['CONTACT_FRAMES'])
        # add to meta data
        meta_data['IT1'] = (self.params['CONTACT_FRAMES'][0],
                            '1st contact frame')
        meta_data['IT2'] = (self.params['CONTACT_FRAMES'][1],
                            '2nd contact frame')
        meta_data['IT3'] = (self.params['CONTACT_FRAMES'][2],
                            '3rd contact frame')
        meta_data['IT4'] = (self.params['CONTACT_FRAMES'][3],
                            '4th contact frame')
        # ---------------------------------------------------------------------
        # deal with removing trend from out-of-transit data
        # ---------------------------------------------------------------------
        # add to tag 2
        tag2 += '_remoottrend{0}'.format(int(self.params['REMOVE_TREND']))
        # add to meta data
        meta_data['RMVTREND'] = (self.params['REMOVE_TREND'],
                                 'Trend was removed from out-of-transit')
        # ---------------------------------------------------------------------
        # deal with out of transit polynomial level correction
        # ---------------------------------------------------------------------
        if self.params['REMOVE_TREND']:
            transit_base_polyord = self.params['TRANSIT_BASELINE_POLYORD']
        else:
            transit_base_polyord = 'None'
        # add to tag 2
        tag2 += '_transit-base-polyord-{0}'.format(transit_base_polyord)
        # ---------------------------------------------------------------------
        # push tag1, tag2 and meta data to variables
        self.set_variable('TAG1', tag1)
        self.set_variable('TAG2', tag2)
        self.set_variable('META', meta_data)

    def temporary_filenames(self):
        # set function name
        func_name = f'{__NAME__}.temporary_filenames()'
        # update meta data
        self.update_meta_data()
        # get tag1
        tag1 = self.get_variable('TAG1', func_name)
        # ---------------------------------------------------------------------
        # construct temporary file names
        # ---------------------------------------------------------------------
        median_image_file = 'median{0}.fits'.format(tag1)
        median_image_file = os.path.join(self.params['TEMP_PATH'],
                                         median_image_file)
        # ---------------------------------------------------------------------
        clean_cube_file = 'cleaned_cube{0}.fits'.format(tag1)
        clean_cube_file = os.path.join(self.params['TEMP_PATH'],
                                        clean_cube_file)
        # ---------------------------------------------------------------------
        tmp_before_after_clean1f = 'temporary_before_after_clean1f.fits'
        tmp_before_after_clean1f = os.path.join(self.params['TEMP_PATH'],
                                                tmp_before_after_clean1f)
        # ---------------------------------------------------------------------
        tmp_pcas = 'temporary_pcas.fits'.format(tag1)
        tmp_pcas = os.path.join(self.params['TEMP_PATH'], tmp_pcas)
        # ---------------------------------------------------------------------
        temp_transit_invsout = 'temporary_transit_in_vs_out.fits'
        tmp_transit_invsout = os.path.join(self.params['TEMP_PATH'],
                                           temp_transit_invsout)
        # ---------------------------------------------------------------------
        temp_clean_nan = 'temporary_cleaned_isolated.fits'
        temp_clean_nan = os.path.join(self.params['TEMP_PATH'], temp_clean_nan)
        # ---------------------------------------------------------------------
        temp_ini_cube = 'temporary_initial_cube.fits'
        temp_ini_cube = os.path.join(self.params['TEMP_PATH'], temp_ini_cube)
        temp_ini_err = 'temporary_initial_err.fits'
        temp_ini_err = os.path.join(self.params['TEMP_PATH'], temp_ini_err)
        # ---------------------------------------------------------------------
        errfile = os.path.join(self.params['TEMP_PATH'],
                               'errormap{}.fits'.format(self.params['tag']))
        # ---------------------------------------------------------------------
        resfile = os.path.join(self.params['TEMP_PATH'],
                               'residual{}.fits'.format(self.params['tag']))
        # ---------------------------------------------------------------------
        reconfile = os.path.join(self.params['TEMP_PATH'],
                                 'recon{}.fits'.format(self.params['tag']))
        # ---------------------------------------------------------------------
        ltbl_file = os.path.join(self.params['CSV_PATH'],
                                 'stability{}.csv'.format(self.params['tag']))
        # ---------------------------------------------------------------------
        sed_table = os.path.join(self.params['CSV_PATH'], 'sed_{0}_ord{1}.csv')
        # ---------------------------------------------------------------------
        # save these for later
        self.set_variable('MEDIAN_IMAGE_FILE', median_image_file)
        self.set_variable('CLEAN_CUBE_FILE', clean_cube_file)
        self.set_variable('TEMP_BEFORE_AFTER_CLEAN1F', tmp_before_after_clean1f)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        self.set_variable('TEMP_TRANSIT_IN_VS_OUT', tmp_transit_invsout)
        self.set_variable('TEMP_CLEAN_NAN', temp_clean_nan)
        self.set_variable('TEMP_INI_CUBE', temp_ini_cube)
        self.set_variable('TEMP_INI_ERR', temp_ini_err)
        self.set_variable('WLC_ERR_FILE', errfile)
        self.set_variable('WLC_RES_FILE', resfile)
        self.set_variable('WLC_RECON_FILE', reconfile)
        self.set_variable('WLC_LTBL_FILE', ltbl_file)
        self.set_variable('SPE_SED_TBL', sed_table)

    # ==========================================================================
    # White light curve functionality
    # ==========================================================================
    def load_data(self, filename: str, ext: int = None, extname: str = None):
        """
        Load the data from a file

        :param filename: str, the filename to load
        :param ext: int, the extension number to load
        :param extname: str, the extension name to load

        :return: data, the loaded data
        """
        _ = self
        # default is to just load the fits file
        data = io.load_fits(filename, ext, extname)
        # return the data
        return data

    def load_table(self, filename: str, ext: int = None):
        """
        Load the table from a file

        :param filename: str, the filename to load
        :param ext: int, the extension number to load

        :return: data, the loaded data
        """
        _ = self
        # default is to just load the table file
        data = io.load_table(filename, ext)
        # return the data
        return data

    def load_cube(self, n_slices: int, raw_shape: Tuple[int, int, int],
                  flag_cds: bool
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # create the containers for the cube of science data,
        # the error cube, and the DQ cube
        cube = np.zeros([n_slices, raw_shape[1], raw_shape[2]])
        err = np.zeros([n_slices, raw_shape[1], raw_shape[2]])
        dq = np.zeros([n_slices, raw_shape[1], raw_shape[2]])
        # counter for the slice we are on
        n_slice = 0
        # ---------------------------------------------------------------------
        # loop around files and push them into the cube/err/dq
        # ---------------------------------------------------------------------
        for ifile, filename in enumerate(self.params['FILES']):
            # load the data
            with fits.open(filename) as hdul:
                # get data from CDS format
                if flag_cds:
                    # bin the data
                    tmp_data = self.bin_cube(hdul[1].data, bin_type='Flux')
                    # convert from cds format
                    tmp_data, tmp_err, tmp_dq = self.from_cds(tmp_data)
                # otherwise get data
                else:
                    tmp_data = self.bin_cube(hdul[1].data, bin_type='Flux')
                    tmp_err = self.bin_cube(hdul[2].data, bin_type='Error')
                    tmp_dq = self.bin_cube(hdul[3].data, bin_type='DQ')
            # get start and end points of cube
            start = n_slice
            end = n_slice + tmp_data.shape[0]
            # push into the cube
            cube[start:end, :, :] = tmp_data
            err[start:end, :, :] = tmp_err
            dq[start:end, :, :] = tmp_dq
            # propagate nslice
            n_slice += tmp_data.shape[0]
        # return the cube, error and DQ
        return cube, err, dq

    def from_cds(self, data: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a conversion from CDS format to standard format

        :param data: np.ndarray, the data to convert [frames, reads, y, x[
        :return:
        """
        # get definitions of first and last from params['CDS_IDS']
        first = self.params['CDS_IDS'][0]
        last = self.params['CDS_IDS'][1]
        # deal with wrong order
        if first > last:
            first, last = last, first
        # deal with first and last being the same
        if first == last:
            emsg = 'CDS_IDS: First and last frame cannot be the same'
            raise exceptions.SossisseConstantException(emsg)
        # get the CDS readout noise
        cds_ron = self.params['CDS_RON']
        # deal with no cds_ron given
        if cds_ron is None:
            emsg = ('FILE(s) found to be CDS: CDS_RON must be set in the '
                    'parameters')
            raise exceptions.SossisseConstantException(emsg)
        # get the difference between the first and last frames
        tmp_data = data[:, last, :, :] - data[:, first, :, :]
        # work out the err and dq values from the cds
        tmp_err = np.sqrt(np.abs(tmp_data) + cds_ron)
        tmp_dq = ~np.isfinite(tmp_data)
        # return the data, err, dq
        return tmp_data, tmp_err, tmp_dq

    def load_data_with_dq(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and correct the data
        :return:
        """
        # set function name
        func_name = f'{__NAME__}.load_data_with_dq()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = self.get_variable('TEMP_INI_CUBE', func_name)
        temp_ini_err = self.get_variable('TEMP_INI_ERR', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            if os.path.exists(temp_ini_cube) and os.path.exists(temp_ini_err):
                # load the data
                cube = self.load_data(temp_ini_cube)
                err = self.load_data(temp_ini_err)
                # for future reference in the code, we keep track of data size
                self.set_variable('DATA_X_SIZE', cube.shape[2])
                self.set_variable('DATA_Y_SIZE', cube.shape[1])
                self.set_variable('DATA_N_FRAMES', cube.shape[0])
                # return
                return cube, err
        # ---------------------------------------------------------------------
        # number of slices in the final cube
        n_slices = 0
        # storage of raw data and shape
        raw_shapes = []
        # handling the files with their varying sizes. We read and count slices
        # TODO: Should be replaced with header keys
        for ifile, filename in enumerate(self.params['FILES']):
            # get the raw files
            tmp_data = self.load_data(filename)
            # get the shape of the bins
            bin_shape = tuple(self.bin_cube(tmp_data, get_shape=True))
            # add to the number of slices
            n_slices += bin_shape[0]
            # store the shape of the raw data
            raw_shapes.append(bin_shape)
            # make sure tmp data is deleted
            del tmp_data
        # ---------------------------------------------------------------------
        # deal with inconsistent data shapes - must all be the same
        if len(set(raw_shapes)) != 1:
            emsg = 'Inconsistent data shapes:'
            for f_it, filename in enumerate(self.params['FILES']):
                emsg += f'\n\t{filename}: {raw_shapes[f_it]}'
            raise exceptions.SossisseFileException(emsg.format(raw_shapes))
        # ---------------------------------------------------------------------
        # store the shape (now it should be consistent)
        raw_shape = raw_shapes[0]
        # raw data might be a cds - flag this and adjust raw shape accordingly
        if len(raw_shape) == 4:
            flag_cds = True
            raw_shape = raw_shape[:-1]
        else:
            flag_cds = False
        #
        self.set_variable('FLAG_CDS', flag_cds)
        # ---------------------------------------------------------------------
        # recalculate tags
        self.update_meta_data()
        # ---------------------------------------------------------------------
        # get flat
        flat, no_flat = self.get_flat(cube_shape=raw_shape)
        # ---------------------------------------------------------------------
        # load and bin the cube
        cube, err, dq = self.load_cube(n_slices, raw_shape, flag_cds)
        # ---------------------------------------------------------------------
        # apply the flat (may be ones)
        if not no_flat:
            # print progress
            misc.printc('Applying flat field to data', 'info')
            # apply the flat field
            for iframe in tqdm(range(cube.shape[0]), leave=False):
                cube[iframe] /= flat
                err[iframe] /= flat
        # ---------------------------------------------------------------------
        # patch to avoid annoying zeros in error map
        # Question: All instruments?
        with warnings.catch_warnings(record=True) as _:
            err[err == 0] = np.nanmin(err[err != 0])
        # ---------------------------------------------------------------------
        # trick to get a mask where True is valid
        cube_mask = np.zeros_like(cube, dtype=bool)
        for valid_dq in self.params['VALID_DQ']:
            # print DQ values
            misc.printc(f'Accepting DQ = {valid_dq}', 'number')
            # get the mask
            cube_mask[dq == valid_dq] = True
        # ---------------------------------------------------------------------
        # remove the background
        cube = self.remove_background(cube)
        # ---------------------------------------------------------------------
        # mask the values in cube_mask
        cube[~cube_mask] = np.nan
        err[~cube_mask] = np.inf
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then save them
        if allow_temp:
            # print progress
            msg = ('We write intermediate files, they will be read to speed '
                   'things next time\n\ttemp cube: {0}\n\ttemp err: {1}')
            margs = [temp_ini_cube, temp_ini_err]
            misc.printc(msg.format(*margs), 'info')
            # force cubes to be float
            cube = cube.astype(float)
            err = err.astype(float)
            # save the data
            fits.writeto(temp_ini_cube, cube, overwrite=True)
            fits.writeto(temp_ini_err, err, overwrite=True)
        # ---------------------------------------------------------------------
        # for future reference in the code, we keep track of data size
        self.set_variable('DATA_X_SIZE', cube.shape[2])
        self.set_variable('DATA_Y_SIZE', cube.shape[1])
        self.set_variable('DATA_N_FRAMES', cube.shape[0])
        # ---------------------------------------------------------------------
        # return the cube and error
        return cube, err

    def bin_cube(self, cube: np.ndarray, bin_type: str = 'Flux',
                 get_shape: bool = False
                 ) -> Union[np.ndarray, Tuple[int, int, int]]:
        """
        Bin the cube

        :param cube:
        :param bin_type:
        :param get_shape: bool, if True return the shape of the binned cube
                          instead of the binned cube
        :return:
        """
        # get function name
        func_name = f'{__NAME__}.bin_cube()'
        # don't bin if user doesn't want to bin
        if not self.params['DATA_BIN_TIME']:
            return cube
        # don't bin if the number of frames in each bin is 1
        if self.params['DATA_BIN_SIZE'] == 1:
            return cube
        # get the bin size
        bin_size = self.params['DATA_BIN_SIZE']
        # we can only bin certain types of files
        if bin_type not in ['Flux', 'Error', 'DQ']:
            emsg = 'Cannot bin cube of type: {0}. \n\tFunction={1}'
            eargs = [bin_type, func_name]
            raise exceptions.SossisseInstException(emsg.format(*eargs),
                                                   self.name)
        # get the dimension size
        n_bins = cube.shape[0] // bin_size
        # shape of the binned image
        bin_shape = (n_bins, cube.shape[1], cube.shape[2])
        # deal with get shape
        if get_shape:
            return bin_shape
        # make an empty cube to store the binned data
        new_cube = np.zeros(bin_shape)
        # print progress
        msg = 'We bin data by a factor {0}'
        margs = [bin_size]
        misc.printc(msg.format(*margs), 'number')
        # loop around bins
        for it in tqdm(range(n_bins), leave=False):
            # get the start frame and end frame
            start = it * bin_size
            end = (it + 1) * bin_size
            # deal with flux binning: sum of cube slice
            if bin_type == 'Flux':
                new_cube[it] = np.nansum(cube[start:end], axis=0)
            # deal with error binning: square root of sum of squares
            elif bin_type == 'Error':
                new_cube[it] = np.sqrt(np.nansum(cube[start:end] ** 2, axis=0))
            # deal with DQ binning: maximum value
            elif bin_type == 'DQ':
                new_cube[it] = np.nanmax(cube[start:end], axis=0)
        # return the new cube
        return new_cube.astype(float)

    def get_flat(self, cube_shape: Tuple[int, int, int]
                 ) -> Tuple[np.ndarray, bool]:
        """
        Get the flat field

        :return: tuple, 1. the flat field, 2. a boolean indicating if the flat
                 field is all ones
        """
        if self.params['FLATFILE'] is None:
            # flat field is a single frame
            return np.ones((cube_shape[1], cube_shape[2])), True
        else:
            # load the flat field
            flat = self.load_data(self.params['FLATFILE'])
            # check the shape of the flat field
            if flat.shape[1:] != cube_shape[1:]:
                emsg = 'Flat field shape does not match data frame shape'
                raise exceptions.SossisseInstException(emsg, self.name)
            # some sanity checks in flat
            flat[flat == 0] = np.nan
            flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
            flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
            # return the flat field
            return flat, False

    def remove_background(self, cube: np.ndarray) -> np.ndarray:
        """
        Removes the background with a 3 DOF model. It's the background image
        times a slope + a DC offset (1 amp, 1 slope, 1 DC)

        Background file is loaded from params['BKGFILE']

        :param cube: np.ndarray, the cube to remove the background from

        :return: np.ndarray, the background corrected cube
        """
        # force the cube to be floats (it should be)
        cube = cube.astype(float)
        # deal with no background file given
        if self.params['BKGFILE'] is None:
            # print progress
            msg = 'We do not clean background. BKGFILE is not set.'
            misc.printc(msg, 'warning')
            # update DO_BACKGROUND variable
            self.set_variable('DO_BACKGROUND', False)
            # return the cube (unchanged)
            return cube
        else:
            # update DO_BACKGROUND variable
            self.set_variable('DO_BACKGROUND', True)
        # update the meta data
        self.update_meta_data()
        # ---------------------------------------------------------------------
        # optimal background correction
        # ---------------------------------------------------------------------
        # print progress
        msg = 'Apply the background model correction'
        misc.printc(msg, 'info')
        # get the background file
        background = self.load_data(self.params['BKGFILE'])
        # force the background to floats
        background = background.astype(float)
        # calculate the mean of the cube (along time axis)
        mcube = np.nanmean(cube, axis=0)
        # get the box size from params
        box = self.params['BACKGROUND_GLITCH_BOX']
        # get the background shifts
        bgnd_shifts_values = self.params['BACKGROUND_SHIFTS']
        bgnd_shifts = np.arange(*bgnd_shifts_values)
        # print progress
        msg = '\tTweaking the position of the background'
        misc.printc(msg, 'info')
        # storage for the rms
        rms = np.zeros_like(bgnd_shifts)
        # loop around shifts
        for ishift in tqdm(range(len(bgnd_shifts)), leave=False):
            # get the shift
            bgnd_shift = bgnd_shifts[ishift]
            # get the shifted background
            background2 = shift(background, (0, bgnd_shift))
            # get the values for robust polyfit
            xvalues = background2[box[2]:box[3], box[0]:box[1]].ravel()
            yvalues = mcube[box[2]:box[3], box[0]:box[1]].ravel()
            # get the background fit
            bfit, _ = mp.robust_polyfit(xvalues, yvalues, 1, 5)
            # get the residuals between background and background fit
            residuals = yvalues - np.polyval(bfit, background2)
            # get the median of the residuals
            med_res = np.nanmedian(residuals[box[2]:box[3], box[0]:box[1]],
                                   axis=0)
            # get the rms
            rms[ishift] = np.std(med_res)

        # find the optimal shift
        rms_min = np.argmin(rms)
        # fit this rms
        bstart = rms_min - 1
        bend = rms_min + 1
        rms_fit = np.polyfit(bgnd_shifts[bstart:bend], rms[bstart:bend], 2)
        # the optimal offset is the half way point
        optimal_offset = -0.5 * rms_fit[1] / rms_fit[0]
        # print the optimal offset
        msg = '\tOptimal background offset {:.3f} pix'
        margs = [optimal_offset]
        misc.printc(msg.format(*margs), 'number')
        # apply the optimal offset
        background2 = shift(background, (0, optimal_offset))
        # fit this background
        background2_box = background2[box[2]:box[3], box[0]:box[1]]
        mean_cube_box = mcube[box[2]:box[3], box[0]:box[1]]
        bfit, _ = mp.robust_polyfit(background2_box.ravel(),
                                    mean_cube_box.ravel(), 1, 5)
        # apply fit to full image
        background = np.polyval(bfit, background2)
        # apply the background correction to the cube
        for frame in tqdm(range(cube.shape[0]), leave=False):
            cube[frame] -= background
        # work out the mean of the cube
        with warnings.catch_warnings(record=True) as _:
            mcube = np.nanmean(cube, axis=0)
        # ---------------------------------------------------------------------
        # low pass filtering
        # ---------------------------------------------------------------------
        # print progress
        msg = '\tLow pass filtering the data'
        misc.printc(msg, 'info')
        # create a mask for values less than the median
        mean_mask = np.zeros_like(mcube, dtype=bool)
        # loop around pixels in the x directory
        for ix_pix in tqdm(range(mcube.shape[1])):
            # get the median value
            mean_cut = np.nanpercentile(mcube[:, ix_pix], 50)
            # get the mask
            mean_mask[:, ix_pix] = mcube[:, ix_pix] < mean_cut
        # convert to float
        mean_mask = mean_mask.astype(float)
        # set all not equal to one to nan
        mean_mask[mean_mask != 1] = np.nan
        # loop around frames
        for iframe in tqdm(range(cube.shape[0]), leave=False):
            with warnings.catch_warnings(record=True) as _:
                med = np.nanmedian(cube[iframe] * mean_mask, axis=0)
                lowp = mp.lowpassfilter(med, 25)
            # subtract the low pass filter from this frame of the cube
            cubetile = np.tile(lowp, mcube.shape[0]).reshape(mcube.shape)
            cube[iframe] -= cubetile
        # return the background corrected cube
        return cube

    def patch_isolated_bads(self, cube: np.ndarray) -> np.ndarray:
        """
        Patch isolated bad pixels in the cube

        :param cube: np.ndarray, the cube to patch

        :return: np.ndarray, the patched cube
        """
        # set function name
        func_name = f'{__NAME__}.patch_isolated_bads()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_clean_nan = self.get_variable('TEMP_CLEAN_NAN', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            if os.path.exists(temp_clean_nan):
                # print progress
                msg = ('patch_isolated_bads: \t we read temporary files to '
                       'speed things up. \\nReading: {0}')
                margs = [temp_clean_nan]
                misc.printc(msg.format(*margs), 'info')
                # load the data
                cube = self.load_data(temp_clean_nan)
                return cube
        # ---------------------------------------------------------------------
        # print progress
        msg = ' Removing isolated NaNs'
        misc.printc(msg, 'info')
        # loop around frames in cube
        for iframe in tqdm(range(cube.shape[0]), leave=False):
            # get the cube frame
            cframe = np.array(cube[iframe, :, :])
            # get a mask of the nan values
            mframe = np.isfinite(cframe)
            # make a kernel surrounding the pixel (3x3)
            kernel = np.zeros([3, 3], dtype=float)
            # set any pixels and the pixel itself to 1
            kernel[1, :] = 1
            kernel[:, 1] = 1
            # find bad pixels that are isolated
            n_bad = convolve2d(np.array(mframe, dtype=float), kernel,
                               mode='same')
            isolated_bad = (n_bad == 4) & ~mframe
            # get the pixel positions with isolated bad pixels
            ypix, xpix = np.where(isolated_bad)
            # for each pixel replace the value with the mean of the 4 pixels
            #   around it
            mean_vals = np.mean([cframe[ypix - 1, xpix],
                                 cframe[ypix + 1, xpix],
                                 cframe[ypix, xpix - 1],
                                 cframe[ypix, xpix + 1]], axis=0)
            # update cframe and set make
            cframe[ypix, xpix] = mean_vals
            # mframe[ypix, xpix] = True
            # push back into the cube
            cube[iframe, :, :] = cframe
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then save them
        if allow_temp:
            # print progress
            msg = ('We write intermediate files, they will be read to speed '
                   'things next time\n\ttemp cleaned: {0}')
            margs = [temp_clean_nan]
            misc.printc(msg.format(*margs), 'info')
            # save the data
            fits.writeto(temp_clean_nan, cube, overwrite=True)
        # ---------------------------------------------------------------------
        # return the cube
        return cube

    def get_trace_positions(self, log: bool = True) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map
        """
        _ = self, log
        raise NotImplementedError('get_trace_pos() must be implemented in '
                                  'child Instrument class')

    def get_trace_map(self, log: bool = True) -> np.ndarray:
        # set function name
        func_name = f'{__NAME__}.get_trace_map()'
        # get x and y size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        # deal with no trace map required
        if self.params['TRACE_WIDTH_EXTRACTION'] < 1:
            # set the tracemap to ones
            tracemap = np.ones((ysize, xsize), dtype=bool)
            # return the tracemap
            return tracemap
        # ---------------------------------------------------------------------
        # get the trace map (instrument dependent)
        tracemap = self.get_trace_positions()
        # ---------------------------------------------------------------------
        # deal with wavelength domain cut down
        if self.params['WLC_DOMAIN'] is not None:
            # get the low and high values
            wavelow, wavehigh = self.params['WLC_DOMAIN']
            # print that we are cutting
            if log:
                msg = 'We cut the domain of the WLC to {0:.2f} - {1:.2f} um'
                margs = [wavelow, wavehigh]
                misc.printc(msg.format(*margs), 'number')
            # get the wavegrid
            wavegrid = self.get_wavegrid()
            # mask the trace map
            tracemap[:, wavegrid < wavelow] = False
            tracemap[:, wavegrid > wavehigh] = False
            # mask any nan values in the wavegrid
            tracemap[:, ~np.isfinite(wavegrid)] = False
        # ---------------------------------------------------------------------
        # return the tracemap
        return tracemap

    def get_trace_pos(self, map2d: bool = False,
                      order_num: int = 1, round_pos: bool = True,
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the trace position

        :param map2d: bool, if True return a 2D map of the trace
        :param order_num: int, the order number to use
        :param round_pos: bool, if True round the positions to integers
        :param log: bool, if True print log messages

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.get_wavegrid()'
        # get x and y size from cube
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        xoffset = self.params['TRACE_X_OFFSET']
        yoffset = self.params['TRACE_Y_OFFSET']
        trace_wid_mask = self.params['TRACE_WIDTH_MASKING']
        # load the trace position
        tbl_ref = self.load_table(self.params['TRACEPOS'], ext=order_num)
        # ---------------------------------------------------------------------
        # get columns
        xpos = tbl_ref['X']
        ypos = tbl_ref['Y']
        # ---------------------------------------------------------------------
        # if we don't have throughput we assume it is all ones
        if 'THROUGHPUT' not in tbl_ref.colnames:
            # print debug message
            msg = ('The trace table does not have a "THROUGHPUT" column, '
                   'we set throughput to 1')
            misc.printc(msg, 'debug')
            # add the throughput column of ones
            throughput = np.ones(len(xpos))
        else:
            throughput = tbl_ref['THROUGHPUT']
        # ---------------------------------------------------------------------
        # get the valid trace positions
        valid = (xpos > 0) & (xpos < xsize)
        # mask the table by these valid positions
        xpos, ypos, throughput = xpos[valid], ypos[valid], throughput[valid]
        # ---------------------------------------------------------------------
        # sort by x positions
        sort = np.argsort(xpos)
        xpos, ypos, throughput = xpos[sort], ypos[sort], throughput[sort]
        # ---------------------------------------------------------------------
        # interpolate the y positions and throughput using splines
        spline_y = ius(xpos + xoffset, ypos + yoffset, ext=0, k=1)
        spline_throughput = ius(xpos + xoffset, throughput, ext=0, k=1)
        # ---------------------------------------------------------------------
        # deal with rounding to integer
        if round_pos:
            dtype = int
        else:
            dtype = float
        # define the required x positions
        rxpos = np.arange(xsize, dtype=dtype)
        # get the positions and throughput
        posmax = np.array(spline_y(rxpos - 0.5), dtype=dtype)
        throughput = np.array(spline_throughput(rxpos), dtype=float)
        # ---------------------------------------------------------------------
        # deal with map2d
        if map2d:
            # make a mask of the image
            posmap = np.zeros([ysize, xsize], dtype=bool)
            # loop around pixels in the x direction
            for ix_pix in range(xsize):
                # get the top and bottom of the trace
                bottom = posmax[ix_pix] - trace_wid_mask // 2
                top = posmax[ix_pix] + trace_wid_mask // 2
                # deal with edge cases
                bottom = max(0, bottom)
                top = min(ysize - 1, top)
                # set the posmap to True
                posmap[bottom:top, ix_pix] = True
            # return the posmap and the throughput
            return posmap, throughput
        else:
            # return the trace position
            return posmax, throughput

    def get_wavegrid(self, source: str = 'pos',
                     order_num: Union[int, None] = None,
                     return_xpix: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the wave grid for the instrument

        :param source: str, the source of the wave grid
        :param order_num: int, the order number to use (if source is pos)
        :param return_xpix: bool, if True return xpix as well as wave

        :return: np.ndarray, the wave grid
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_wavegrid()'
        # get x size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        # deal with case where we need POS_FILE and it is not given
        if self.params['POS_FILE'] is None:
            emsg = (f'POS_FILE must be defined for {self.name}'
                    f'\n\tfunction = {func_name}')
            raise exceptions.SossisseInstException(emsg, self.name)
        # deal with order num not being set
        elif order_num is None:
            emsg = ('order_num must be defined when using source="pos"'
                    '\n\tfunction = {0}')
            raise exceptions.SossisseInstException(emsg.format(func_name),
                                                   self.name)
        # otherwise we use POS_FILE
        else:
            # get the trace position file
            tbl_ref = self.load_table(self.params['POS_FILE'], order_num)
            # get the valid pixels
            valid = tbl_ref['X'] > 0
            valid &= tbl_ref['X'] < xsize - 1
            valid &= np.isfinite(np.array(tbl_ref['WAVELENGTH']))
            # mask the table by these valid positions
            tbl_ref = tbl_ref[valid]
            # sort by the x positions
            tbl_ref = tbl_ref[np.argsort(tbl_ref['X'])]
            # spline the wave grid
            spl_wave = ius(tbl_ref['X'], tbl_ref['WAVELENGTH'], ext=1, k=1)
            # push onto our wave grid
            wavevector = spl_wave(np.arange(xsize))
            # deal with zeros
            wavevector[wavevector == 0] = np.nan
            # get xpix
            xpix = np.arange(xsize)
        # return the wave grid
        if return_xpix:
            return xpix, wavevector
        else:
            return wavevector

    def clean_1f(self, cube: np.ndarray,
                 err: np.ndarray,
                 tracemap: np.ndarray) -> List[Union[np.ndarray, None]]:
        """
        Clean the 1/f noise from the cube

        :param cube: np.ndarray, the cube to clean
        :param err: np.ndarray, the error cube
        :param tracemap: np.ndarray, the trace map

        :return: tuple, 1. the clean cube, 2. the median image,
                 3. the tmp before after clean1f, 4. the transit in vs out
        """
        # define the function name
        func_name = f'{__NAME__}.{self.name}.clean_1f()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        # ---------------------------------------------------------------------
        # construct temporary file names
        self.temporary_filenames()
        # save these for later
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        clean_cube_file = self.set_variable('CLEAN_CUBE_FILE', func_name)
        tmp_before_after_c1f = self.set_variable('TEMP_BEFORE_AFTER_CLEAN1F',
                                                 func_name)
        tmp_pcas = self.set_variable('TEMP_PCA_FILE', func_name)
        tmp_transit_invsout = self.set_variable('TEMP_TRANSIT_IN_VS_OUT',
                                                func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            # make sure all required files exist
            cond = os.path.exists(median_image_file)
            cond &= os.path.exists(clean_cube_file)
            cond &= os.path.exists(tmp_before_after_c1f)
            cond &= os.path.exists(tmp_transit_invsout)
            # only look for fit pca file if we are fitting pca
            if self.params['FIT_PCA']:
                cond &= os.path.exists(tmp_pcas)
            # if all conditions are satisfied we load the files
            if cond:
                # print that we are loading temporary files to speed things up
                msg = 'We read temporary files to speed things up.'
                misc.printc(msg, 'info')
                # load the median image
                misc.printc('\tReading: {0}'.format(median_image_file), 'info')
                median_image = self.load_data(median_image_file)
                # load the clean cube
                misc.printc('\tReading: {0}'.format(clean_cube_file), 'info')
                clean_cube = self.load_data(clean_cube_file)
                # load the before after clean 1f
                misc.printc('\tReading: {0}'.format(tmp_before_after_c1f),
                            'info')
                before_after_clean1f = self.load_data(tmp_before_after_c1f)
                # load the transit in vs out
                misc.printc('\tReading: {0}'.format(tmp_transit_invsout),
                            'info')
                transit_invsout = self.load_data(tmp_transit_invsout)
                # only look for fit pca file if we are fitting pca
                if self.params['FIT_PCA']:
                    misc.printc('\tReading: {0}'.format(tmp_pcas), 'info')
                    pcas = self.load_data(tmp_pcas)
                else:
                    pcas = None
                # return these files
                return_list = [clean_cube, median_image, before_after_clean1f,
                               transit_invsout, pcas]
                return return_list
        # ---------------------------------------------------------------------
        # do the 1/f filtering
        # ---------------------------------------------------------------------
        # create a copy of the cube, we will normalize the amplitude of each
        # trace
        cube2 = np.array(cube, dtype=float)
        # first estimate of the trace amplitude
        misc.printc('First median of cube to create trace esimate', 'info')
        # validate out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        oot_domain_before = self.get_variable('OOT_DOMAIN_BEFORE', func_name)
        oot_domain_after = self.get_variable('OOT_DOMAIN_AFTER', func_name)
        int_domain = self.get_variable('INT_DOMAIN', func_name)
        # get flag for median out of transit
        med_oot = self.params['MEDIAN_OOT']
        # ---------------------------------------------------------------------
        # deal with creating median
        with warnings.catch_warnings:
            if med_oot and has_oot:
                med = np.nanmedian(cube2[oot_domain], axis=0)
            else:
                med = np.nanmedian(cube2, axis=0)
        # ---------------------------------------------------------------------
        # do a dot product of each trace to the median and adjust amplitude
        #   so that they all match
        amps = np.zeros(nframes)
        # loop around frames
        for iframe in tqdm(range(nframes), leave=False):
            # only keep finite values in all frames
            valid = np.isfinite(cube2[iframe]) & np.isfinite(med)
            # work out the amps
            part1a = np.nansum(cube2[iframe][valid] * med[valid])
            part2a = np.nansum(med[valid] ** 2)
            amps[iframe] = part1a / part2a
            # normalize the data in the cube by these amplitues
            cube2[iframe] /= amps[iframe]
        # ---------------------------------------------------------------------
        # median of the normalized cube
        misc.printc('Second median of cube with proper normalization', 'info')
        # normalize cube
        with warnings.catch_warnings(record=True) as _:
            if med_oot:
                med = np.nanmedian(cube2[oot_domain], axis=0)
                before = np.nanmedian(cube2[oot_domain_before], axis=0)
                after = np.nanmedian(cube2[oot_domain_after], axis=0)
                # get the median difference
                med_diff = before - after
                # low pass the median difference
                for frame in range(med_diff.shape[0]):
                    med_diff[frame] = mp.lowpassfilter(med_diff[frame], 15)
                # get the square ratio between median and med diff
                ratio = np.sqrt(np.nansum(med**2) / np.nansum(med_diff**2))
                # scale the median difference
                med_diff *= ratio
            else:
                med = np.nanmedian(cube2, axis=0)
        # ---------------------------------------------------------------------
        # also keep track of the in vs out-of-transit 2D image.
        med_out = np.nanmedian(cube2[oot_domain], axis=0)
        med_in = np.nanmedian(cube2[int_domain], axis=0)
        # get the diff in vs out
        transit_invsout = med_in - med_out
        # ---------------------------------------------------------------------
        # work out the residuals
        residuals = np.zeros_like(cube)
        # loop around frames
        for iframe in tqdm(range(nframes), leave=False):
            # mask the nans and apply trace map
            valid = np.isfinite(cube2[iframe]) & np.isfinite(med)
            valid &= tracemap
            # work out the amplitudes in the valid regions
            part1b = np.nansum(cube2[iframe][valid] * med[valid])
            part2b = np.nansum(med[valid] ** 2)
            amps[iframe] = part1b / part2b
            # we get the appropriate slice of the error cube
            residuals[iframe] = cube[iframe] - med * amps[iframe]
        # ---------------------------------------------------------------------
        # Subtract of the 1/f noise
        # ---------------------------------------------------------------------
        cube = self.subtract_1f(residuals, cube, err, tracemap)
        # ---------------------------------------------------------------------
        # fit the pca
        # ---------------------------------------------------------------------
        # fit the pca
        pcas = self.fit_pca(cube, err, med, tracemap)
        # ---------------------------------------------------------------------
        # write files to disk
        # ---------------------------------------------------------------------
        if allow_temp:
            # write the median image
            misc.printc('\tWriting: {0}'.format(median_image_file), 'info')
            fits.writeto(median_image_file, med, overwrite=True)
            # write the clean cube
            misc.printc('\tWriting: {0}'.format(clean_cube_file), 'info')
            fits.writeto(clean_cube_file, cube, overwrite=True)
            # write the before after clean 1f
            misc.printc('\tWriting: {0}'.format(tmp_transit_invsout),
                        'info')
            fits.writeto(tmp_transit_invsout, transit_invsout,
                         overwrite=True)
        # ---------------------------------------------------------------------
        # return the cleaned cube, the median image, the median difference
        return_list =  [cube, med, med_diff, transit_invsout, pcas]
        return return_list

    def get_valid_oot(self):
        """
        Get the out-of-transit domain (before, after and full removing any
        rejected domain)

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_valid_oot()'
        # get the oot domain
        oot_domain = self._variables['OOT_DOMAIN']
        # deal with value already found
        if oot_domain is not None:
            return
        # get the number of frames
        data_n_frames = self.get_variable('DATA_N_FRAMES', func_name)
        # get the contact points
        cframes = self.params['CONTACT_FRAMES']
        # deal with no cframes set (can happen)
        if cframes is None:
            # set flag
            self.set_variable('HAS_OOT', False)
            # update variables
            self.set_variable('OOT_DOMAIN', 
                              np.zeros(data_n_frames, dtype=bool))
            self.set_variable('OOT_DOMAIN_BEFORE',  
                              np.zeros(data_n_frames, dtype=bool))
            self.set_variable('OOT_DOMAIN_AFTER',  
                              np.zeros(data_n_frames, dtype=bool))
            self.set_variable('INT_DOMAIN', np.ones(data_n_frames, dtype=bool))
            return
        # get the rejection domain
        rej_domain = self.params['REJECT_DOMAIN']
        # if we don't have out-of-transit domain work it out
        valid_oot = np.ones(data_n_frames, dtype=bool)
        # set the frames in the transit to False
        valid_oot[cframes[0]:cframes[3]] = False
        # deal with the rejection of domain
        if rej_domain is not None:
            # get the rejection domain
            for ireject in range(len(rej_domain) // 2):
                # get the start and end of the domain to reject
                start = rej_domain[ireject * 2]
                end = rej_domain[ireject * 2 + 1]
                # set to False in valid_oot
                valid_oot[start:end] = False
        # get the frames before
        valid_oot_before = np.array(valid_oot)
        valid_oot_before[cframes[0]:] = False
        # get the frames after
        valid_oot_after = np.array(valid_oot)
        valid_oot_after[:cframes[3]] = False
        # ---------------------------------------------------------------------
        # get the valid in transit domain
        valid_int = np.ones(data_n_frames, dtype=bool)
        # set the frames out of transit to False
        valid_int[:cframes[0]] = False
        valid_int[cframes[3]:] = False
        # deal with rejection of domain
        if rej_domain is not None:
            # get the rejection domain
            for ireject in range(len(rej_domain) // 2):
                # get the start and end of the domain to reject
                start = rej_domain[ireject * 2]
                end = rej_domain[ireject * 2 + 1]
                # set to False in valid_int
                valid_int[start:end] = False
        # ---------------------------------------------------------------------
        # set flag
        self.set_variable('HAS_OOT', True)
        # update variables
        self.set_variable('OOT_DOMAIN', valid_oot)
        self.set_variable('OOT_DOMAIN_BEFORE', valid_oot_before)
        self.set_variable('OOT_DOMAIN_AFTER', valid_oot_after)
        self.set_variable('INT_DOMAIN', valid_int)

    def subtract_1f(self, residuals: np.ndarray,
                    cube: np.ndarray, err: np.ndarray,
                    tracemap: np.ndarray):
        """
        Do the actual subtraction of the 1/f noise, once we have the residuals

        :param residuals: np.ndarray, the residuals
        :param cube: np.ndarray, the cube
        :param err: np.ndarray, the error cube
        :param tracemap: np.ndarray, the trace map

        :return:
        """
        # define the function name
        func_name = f'{__NAME__}.{self.name}.clean_1f()'
        # get the degree for the 1/f polynomial fit
        degree_1f_corr = self.params['DEGREE_1F_CORR']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # deal with no poly fit of the 1/f noise
        if degree_1f_corr == 0:
            # get the median noise contribution
            noise_1f = np.nanmedian(residuals, axis=1)
            # subtract this off the cube frame-by-frame
            for iframe in tqdm(range(nframes), leave=False):
                # we subtract the 1/f noise off each column
                for col in range(nbxpix):
                    cube[iframe, :, col] -= noise_1f[iframe, col]
        # otherwise we fit the 1/f noise
        else:
            # loop around every frame
            for iframe in tqdm(range(nframes), leave=False):
                # get the residuals
                res = residuals[iframe]
                err2 = np.array(err[iframe])
                # loop around columns
                for col in range(nbxpix):
                    # subtract only the 0th term of the odd_ratio_mean, the
                    # next one is the uncertainty in the mean
                    # deal with not enough data points to fit
                    if np.sum(np.isfinite(res[:, col])) < degree_1f_corr + 3:
                        continue

                    # otherwise try to fit the 1/f noise with a polynomial
                    v1 = residuals[:, col]
                    err1 = err2[:, col]
                    index = np.arange(res.shape[0], dtype=float)
                    # find valid pixels
                    valid = np.isfinite(v1 + err1)
                    valid &= ~tracemap[:, col]
                    valid &= np.abs(v1 / err1) < 5
                    # try fit the polynomial
                    try:
                        pfit = np.polyfit(index[valid], v1[valid],
                                          degree_1f_corr, w=1 / err1[valid])
                        # subtract the fit from the cube
                        cube[iframe, :, col] -= np.polyval(pfit, index)
                    except Exception as _:
                        # if the fit fails we just set the column to NaN
                        cube[iframe, :, col] = np.nan
        return cube

    def fit_pca(self, cube2: np.ndarray, err: np.ndarray,
                med: np.ndarray, tracemap: np.ndarray):
        """
        Fit the PCA to the tracemap

        :param tracemap:
        :return:
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.fit_pca()'
        # ---------------------------------------------------------------------
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        # update meta data
        self.update_meta_data()
        # get tag1
        tag1 = self.get_variable('TAG1', func_name)
        # make sure we have a pca file
        tmp_pcas = 'temporary_pcas.fits'.format(tag1)
        tmp_pcas = os.path.join(self.params['TEMP_PATH'], tmp_pcas)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        # ---------------------------------------------------------------------
        # get whether to fit pca and the number of fit components
        flag_fit_pca = self.params['FIT_PCA']
        n_comp = self.params['FIT_N_PCA']
        # if we aren't fitting or we fit no components return None
        if flag_fit_pca or n_comp == 0:
            return None
        # get the shape of the data
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # validate out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # if we don't have oot domain we cannot do the pca analysis
        if not has_oot:
            wmsg = ('Cannot do PCA analysis without out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to use PCA.')
            misc.printc(wmsg, 'warning')
            return None
        # ---------------------------------------------------------------------
        # only fit the pca to the flux in the trace (nan everything else)
        nanmask = np.ones_like(tracemap, dtype=float)
        nanmask[~tracemap] = np.nan
        # copy the normalized cube
        cube3ini = cube2[oot_domain]
        # subtract off the median
        for iframe in tqdm(range(cube3ini.shape[0]), leave=False):
            cube3ini[iframe] -= med
        # ---------------------------------------------------------------------
        # copy the normalized cube
        cube3 = np.array(cube3ini)
        # get the valid error domain
        err3 = err[oot_domain]
        # apply the nanmask to the cube3
        for iframe in range(cube3.shape[0]):
            cube3[iframe] *= nanmask
        # reshape the cubes to flatten each frame into a 1D array
        cube3 = cube3.reshape([cube3.shape[0], nbypix * nbxpix])
        err3 = err3.reshape([err3.shape[0], nbypix * nbxpix])
        # find the bad pixels
        badpix = np.isfinite(cube3) & np.isfinite(err3)
        # work out the weights
        weights = 1 / err3
        # set the bad weights to zero
        weights[badpix] = 0
        cube3[badpix] = 0
        # ---------------------------------------------------------------------
        # find out the regions that are valid at least 95% of the time
        with warnings.catch_warnings(record=True) as _:
            valid = np.where(np.nanmean(weights != 0, axis=0) > 0.95)[0]
        # ---------------------------------------------------------------------
        # compute the principle components
        with warnings.catch_warnings(record=True) as _:
            # set up the pca class
            pca_class = EMPCA(n_components=n_comp)
            # fit out data
            pca_fit = pca_class.fit(cube3[:, valid], weights=weights[:, valid])
            # get the raito of all components
            variance_ratio = np.array(pca_class.explained_variance_ratio_)
            # normalize by the ratio of the first component
            variance_ratio /= variance_ratio[0]
        # ---------------------------------------------------------------------
        # get the amplitudes
        amps = np.zeros((n_comp, cube3.shape[0]))
        # loop around frames
        for iframe in range(cube3.shape[0]):
            # get the error squared
            err3_2 = err3[iframe, valid] ** 2
            # loop around components
            for icomp in range(n_comp):
                # get the component value
                comp_val = pca_fit.components_[icomp]
                # get the amplitude of this component
                part1 = np.nansum(comp_val * cube3[iframe, valid] / err3_2)
                part2 = np.nansum(1 / err3_2)
                amps[icomp, iframe] = part1 / part2
        # ---------------------------------------------------------------------
        # calculate the pca values
        pcas = np.zeros((n_comp, nbypix, nbxpix), dtype=float)
        # reset cube3 to its initial value
        cube3 = np.array(cube3ini)
        # set all nan pixels to zero
        cube3[~np.isfinite(cube3)] = 0
        # loop around components
        for icomp in range(n_comp):
            # add the contribution from each frame
            for iframe in range(cube3.shape[0]):
                pcas[icomp] += amps[icomp, iframe] * cube3[iframe]
            # TODO: Etienne explain this
            # trick to avoid getting slices of a 3D cube
            tmp = np.array(pcas[icomp, :, :])
            for icol in range(nbxpix):
                tmp[:, icol] -= np.nanmedian(tmp[:, icol])
            # normalize the tmp component
            # TODO: Etienne explain this
            part1 = np.nansum(tmp ** 2 * nanmask)
            part2 = np.nansum(med ** 2 * nanmask)
            tmp /= np.sqrt(part1 / part2)
            # update the pcas ith the tmp array
            pcas[icomp, :, :] = tmp
        # ---------------------------------------------------------------------
        # deal with plotting
        plots.pca_plot(self.params, n_comp, pcas, variance_ratio)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then save them
        if allow_temp:
            tmp_pcas = self.get_variable('TEMP_PCA_FILE', func_name)
            # write the pca
            misc.printc('\tWriting: {0}'.format(tmp_pcas), 'info')
            fits.writeto(tmp_pcas, pcas, overwrite=True)
        # ---------------------------------------------------------------------
        # return the pcas
        return pcas

    def recenter_trace_position(self, tracemap: np.ndarray,
                                med: np.ndarray) -> np.ndarray:
        # set function name
        func_name = f'{__NAME__}.{self.name}.recenter_trace_position()'
        # deal with not wanting to recenter trace position
        if not self.params['RECENTER_TRACE_POSITION']:
            return tracemap
        # ---------------------------------------------------------------------
        # get some parameters from instrument
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # ---------------------------------------------------------------------
        # print what we are doing
        msg = 'Scan to optimize position of trace'
        misc.printc(msg, 'info')
        # save the current width (we will reset it later
        width_current = float(self.params['TRACE_WIDTH_MASKING'])
        width_source = self.sources['TRACE_WIDTH_MASKING']
        # force a trace width masking
        self.params['TRACE_WIDTH_MASKING'] = 20
        self.sources['TRACE_WIDTH_MASKING'] = func_name

        # get a range of dys and dxs to scan over for best trace position
        dys = np.arange(-nbypix // 10, nbypix // 10 + 1)
        dxs = np.arange(-nbypix // 10, nbypix // 10 + 1)
        sums = np.zeros([len(dxs), len(dys)], dtype=float)
        # storage for best dx and dy
        best_dx = 0
        best_dy = 0
        best_sum = 0
        # loop around dxs and dys
        for ix in tqdm(range(len(dxs)), leave=False):
            for iy in tqdm(range(len(dys)), leave=False):
                # update the x and y positions
                self.params['X_TRACE_OFFSET'] = dxs[ix]
                self.params['Y_TRACE_OFFSET'] = dys[iy]
                # re-gen the trace map (without logging) using new x/y trace
                #  offset
                params = self.get_trace_map(log=False)
                sums[ix, iy] = np.nansum(params['TRACEMAP'] * med)
                if sums[ix, iy] > best_sum:
                    best_sum = sums[ix, iy]
                    best_dx = dxs[ix]
                    best_dy = dys[iy]
        # print the best dx and dy
        misc.printc('Best dx : {} pix'.format(best_dx), 'number')
        misc.printc('Best dy : {} pix'.format(best_dy), 'number')
        # update the trace offsets with the best values found
        self.params['X_TRACE_OFFSET'] = best_dx
        self.sources['X_TRACE_OFFSET'] = func_name
        self.params['Y_TRACE_OFFSET'] = best_dy
        self.sources['Y_TRACE_OFFSET'] = func_name
        # reset the trace width masking to the user defined value
        self.params['TRACE_WIDTH_MASKING'] = width_current
        self.sources['TRACE_WIDTH_MASKING'] = width_source
        # return the updated trace map
        return self.get_trace_map()

    def get_gradients(self, med: np.ndarray) -> List[np.ndarray]:
        """
        Get the gradients of the median image

        :param med: np.ndarray, the median image

        :return: tuple, 1. the x gradient, 2. the y gradient, 3. the rotation
                    pattern, 4. the second derivative, 5. the median image
        """
        # print progress
        msg = 'We find the gradients'
        misc.printc(msg, 'info')
        # ---------------------------------------------------------------------
        # copy the med array into a new array
        med2 = np.array(med)
        # iterate four times and replace bad pixels with the median filter
        # of the data of the 5 pixels around them (in the x direction)
        for _ in range(4):
            # get median filter
            med_filter = medfilt2d(med2, kernel_size=[1, 5])
            # mask bad pixels
            bad = ~np.isfinite(med2)
            # update the med2 array
            med2[bad] = med_filter[bad]
        # ---------------------------------------------------------------------
        # find gradients along the x and y direction
        dx, dy = np.gradient(med2)
        # find the second derivative
        ddy = np.gradient(dy, axis=0)
        # ---------------------------------------------------------------------
        # find the rotation pattern as per the coupling between the two axes
        # ---------------------------------------------------------------------
        # get the indices
        yy, xx = np.indices(med2.shape, dtype=float)
        # we assume a pivot relative to the center of the array
        xx -= med2.shape[1] / 2.0
        yy -= med2.shape[0] / 2.0
        # infinitesimal motion in rotation scaled to 1 radian
        rotxy = xx * dy - yy * dx
        # make sure dx and dy are floats
        dx, dy = np.array(dx, dtype=float), np.array(dy, dtype=float)
        # make sure rotation and ddy are floats
        rotxy, ddy = np.array(rotxy, dtype=float), np.array(ddy, dtype=float)
        # ---------------------------------------------------------------------
        plots.gradient_plot(self.params, dx, dy, rotxy)
        # ---------------------------------------------------------------------
        # return these values
        return [dx, dy, rotxy, ddy, med2]

    def get_mask_trace_pos(self, med: np.ndarray, tracemap: np.ndarray
                           ) -> List[np.ndarray]:
        """
        Get the mask trace positions

        :param med: np.ndarray, the median image
        :param tracemap: np.ndarray, the trace map

        :return: list, 1. the mask trace positions, 2. the x order 0 positions,
                       3. the y order 0 positions, 4. the x trace positions,
                       5. the y trace positions
        """
        # set up the mask trace (all true to start)
        mask_trace_pos = np.ones_like(med, dtype=int)
        # ---------------------------------------------------------------------
        if self.params['TRACE_WIDTH_MASKING'] != 0:
            mask_trace_pos[~tracemap] = 0
            # define a box for binary dilation
            box = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            # binary dilate the mask trace positions (expand them)
            bdilate = binary_dilation(mask_trace_pos, structure=box)
            # get the x and y trace positions from the binary dilation and mask
            y_trace_pos, x_trace_pos = np.where(bdilate != mask_trace_pos)
        else:
            y_trace_pos, x_trace_pos = np.array([np.nan]), np.array([np.nan])
        # ---------------------------------------------------------------------
        # deal with masking order zero
        if self.params['MASK_ORDER_0']:
            # adding the masking of order 0
            mask_order0, x_order0, y_order0 = self.get_mask_order0(mask_trace_pos)
            mask_trace_pos[mask_order0] = 0
        else:
            # if we don't have a mask, we set dummy values for the plot later in the code
            x_order0 = [np.nan]
            y_order0 = [np.nan]
        # return the mask trace positions
        return [mask_trace_pos, x_order0, y_order0, x_trace_pos, y_trace_pos]

    def get_mask_order0(self, mask_trace_pos: np.ndarray, tracemap: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the mask for order 0 - this is a dummy function that returns
        the default values and overriden by JWST.NIRISS.SOSS

        :param mask_trace_pos: np.ndarray, the mask trace positions
        :param tracemap: np.ndarray, the trace map

        :return: tuple, 1. the updated mask trace positions, 2. the x order 0
                        positions, 3. the y order 0 positions
        """
        # default option does not use tracemap
        _ = tracemap
        # default option is not to mask order 0 (overridden by SOSS)
        empty_x = np.array([np.nan])
        empty_y = np.array([np.nan])
        # return the default values
        return mask_trace_pos, empty_x, empty_y

    def setup_linear_reconstruction(self, med: np.ndarray, dx: np.ndarray,
                                    dy: np.ndarray, rotxy: np.ndarray,
                                    ddy: np.ndarray, pca: np.ndarray,
                                    med_diff: np.ndarray) -> np.ndarray:
        """
        Setup the linear reconstruction vector and output column names

        Size of the vector is (M X N) where N is the dx.ravel() and M is the
        options which are switched on

        M = 1  (amplitude) by default in all cases

        then we add the following depending on the user input
        - FIT_DX [M += 1]
        - FIT_DY [M += 1]
        - FIT_ROTATION [M += 1]
        - FIT_ZERO_POINT_OFFSET [M += 1]
        - FIT_DDY [M += 1]
        - FIT_BEFORE_AFTER [M += 1]
        - FIT_PCA [M += n_comp]
        - FIT_QUAD_TERM [M += 1]

        :param med: np.ndarray, the median image
        :param dx: np.ndarray, the x gradient
        :param dy: np.ndarray, the y gradient
        :param rotxy: np.ndarray, the rotation pattern
        :param ddy: np.ndarray, the second derivative
        :param pca: np.ndarray, the pca components
        :param med_diff: np.ndarray, the median difference

        :return: np.ndarray, the starting linear reconstruction vector
                 depending on which parameters are switched on
                 (M x N) where N is the dx.ravel() and M is the options which
                 are switched on
        """
        # vector is the median ravelled
        vector = [med.ravel()]
        # output parameters
        output_names: List[str] = []
        output_units: List[str] = []
        output_factor: List[float] = []
        # add the amplitude
        output_names.append('amplitude')
        output_units.append('flux')
        output_factor.append(1.0)
        # ---------------------------------------------------------------------
        # deal with fit dx
        if self.params['FIT_DX']:
            vector.append(dx.ravel())
            output_names.append('dx')
            output_units.append('mpix')
            output_factor.append(1e3)
        # ---------------------------------------------------------------------
        # deal with fix dy
        if self.params['FIT_DY']:
            vector.append(dy.ravel())
            output_names.append('dy')
            output_units.append('mpix')
            output_factor.append(1e3)
        # ---------------------------------------------------------------------
        # deal with fit rotation
        if self.params['FIT_ROTATION']:
            vector.append(rotxy.ravel())
            output_names.append('theta')
            output_units.append('mpix')
            output_factor.append(129600 / (2 * np.pi))
        # ---------------------------------------------------------------------
        # deal with zero point offset fit
        if self.params['FIT_ZERO_POINT_OFFSET']:
            vector.append(np.ones_like(dx.ravel()))
            output_names.append('zeropoint')
            output_units.append('flux')
            output_factor.append(1.0)
        # ---------------------------------------------------------------------
        # deal with fit second derivative
        if self.params['FIT_DDY']:
            vector.append(ddy.ravel())
            output_names.append('ddy')
            output_units.append('mpix$^2$')
            output_factor.append(1e6)
        # ---------------------------------------------------------------------
        # deal with fit before / after
        if self.params['FIT_BEFORE_AFTER']:
            vector.append(np.zeros_like(med).ravel())
            output_names.append('before_after')
            output_units.append('ppm')
            output_factor.append(1e6)
        # ---------------------------------------------------------------------
        # deal with fit pca
        if self.params['FIT_PCA'] and pca is not None:
            n_comp = self.params['FIT_N_PCA']
            for icomp in range(n_comp):
                vector.append(pca[icomp].ravel())
                output_names.append(f'PCA{icomp+1}')
                output_units.append('ppm')
                output_factor.append(1.0)
        # ---------------------------------------------------------------------
        # deal with quadratic term
        if self.params['FIT_QUAD_TERM']:
            vector.append(med_diff.ravel() ** 2)
            output_names.append('flux^2')
            output_units.append('flux$^2$')
            output_factor.append(1.0)
        # ---------------------------------------------------------------------
        # convert vector to numpy array
        vector = np.array(vector)
        # ---------------------------------------------------------------------
        # push into variables
        self.set_variable('OUTPUT_NAMES', output_names)
        self.set_variable('OUTPUT_UNITS', output_units)
        self.set_variable('OUTPUT_FACTOR', output_factor)
        # return the vector
        return vector

    def apply_amp_recon(self, cube: np.ndarray, err: np.ndarray,
                        med: np.ndarray, mask_trace_pos: np.ndarray,
                        lvector: np.ndarray,
                        x_trace_pos: np.ndarray, y_trace_pos: np.ndarray,
                        x_order0: np.ndarray, y_order0: np.ndarray
                        ) -> Tuple[Table, np.ndarray, np.ndarray]:
        """
        Apply the amplitude reconstruction to the cube

        outputs:

        1. 'amplitude' -> the amplitude of the trace
        2. 'dx' -> amplitude of the dx of the trace (if FIT_DX)
        3. 'dy' -> amplitude of the dy of the trace (if FIT_DY)
        4. 'theta' -> amplitude of the rotation of the trace (if FIT_ROTATION)
        5. 'zeropoint' -> amplitude of the zero point offset (if FIT_ZERO_POINT_OFFSET)
        6. 'ddy' -> amplitude of the second derivative of the trace (if FIT_DDY)
        7. 'before_after' -> amplitude of the before/after trace (if FIT_BEFORE_AFTER)
        8+Q. 'PCAQ' -> amplitude of the first PCA component (if FIT_PCA)
                       note there are Q components
        9. 'flux^2' -> amplitude of the flux squared (if FIT_QUAD_TERM)

        :param cube: np.ndarray, the cube
        :param err: np.ndarray, the error cube
        :param med: np.ndarray, the median image
        :param mask_trace_pos: np.ndarray, the mask trace positions
        :param lvector: np.ndarray, the linear reconstruction vector
        :param x_trace_pos: np.ndarray, the x trace positions
        :param y_trace_pos: np.ndarray, the y trace positions
        :param x_order0: np.ndarray, the x order 0 positions
        :param y_order0: np.ndarray, the y order 0 positions

        :return: tuple, 1. dict, the outputs (see above), 2. the recon cube,
                 3. the valid mask
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.apply_amp_recon()'
        # get parameters
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        output_names = self.get_variable('OUTPUT_NAMES', func_name)
        zpoint = self.params['FIT_ZERO_POINT_OFFSET']
        # vectors to keep track of the rotation/amplitudes/dx/dy
        all_recon = np.zeros_like(cube)
        # ---------------------------------------------------------------------
        # storage to append to (one value for each frame of the cube)
        outputs = dict()
        for output_name in output_names:
            outputs[output_name] = np.zeros(cube.shape[0])
            outputs[output_name + '_error'] = np.zeros(cube.shape[0])
        # add rms cube of the reconstruction
        outputs['rms_cube_recon'] = np.zeros(cube.shape[0])
        # add the sum of the trace
        outputs['sum_trace'] = np.zeros(cube.shape[0])
        # add the sum of the trace error
        outputs['sum_trace_error'] = np.zeros(cube.shape[0])
        # add the amplitude without the model
        outputs['amplitude_no_model'] = np.zeros(cube.shape[0])
        # add the amplitude without the model error
        outputs['amplitude_no_model error'] = np.zeros(cube.shape[0])
        # storage of a mask cube of bad pixels
        valid_arr = np.ones_like(cube).astype(bool)
        # ---------------------------------------------------------------------
        # storage a trace correction array
        trace_corr = np.zeros(nframes, dtype=float)
        # loop around the frames
        for iframe in tqdm(range(cube.shape[0]), leave=False):
            # find the best combination of scale/dx/dy/rotation
            # amps is a vector with the amplitude of all fitted terms
            # -----------------------------------------------------------------
            # set up a mask of valid pixels
            valid = np.isfinite(cube[iframe], dtype=float)
            valid[valid != 1] = np.nan
            valid[mask_trace_pos == 0] = np.nan
            # -----------------------------------------------------------------
            # calculate the sum of the trace
            with warnings.catch_warnings(record=True) as _:
                # work out the sum and error on the sum of the trace
                sum_trace = np.nansum(cube[iframe] * valid)
                err_sum_trace = np.sqrt(np.nansum(err[iframe]**2 * valid))
                # push into outputs
                outputs['sum_trace'][iframe] = sum_trace
                outputs['sum_trace_error'][iframe] = err_sum_trace
            # -----------------------------------------------------------------
            # calculate the amplitude when there is no model applied
            with warnings.catch_warnings(record=True) as _:
                # normalize cube by median
                tmp_slice = valid * (cube[iframe] / med.ravel())
                tmp_slice_err = valid * (err[iframe] / med.ravel())
                # get the odd ratio mean
                amp0 = mp.odd_ratio_mean(tmp_slice, tmp_slice_err)
                # push into outputs
                outputs['amplitude_no_model'][iframe] = amp0[0]
                outputs['amplitude_no_model error'][iframe] = amp0[1]
            # -----------------------------------------------------------------
            # >5 sigma clipping of linear system
            bad = np.abs((cube[iframe] - med * amp0[0]) / err[iframe]) > 5
            valid[bad] = np.nan
            # setup inputs to lin mini errors
            largs = [cube[iframe] * valid, err[iframe], lvector]
            amp_model, err_model, recon = mp.lin_mini_errors(*largs)
            # -----------------------------------------------------------------
            # deal with zero point offset (subtract it off the recon)
            if zpoint:
                recon -= amp_model[output_names == 'zero point']
            # -----------------------------------------------------------------
            # calculate the trace correction
            part1 = np.nansum(recon * valid / err[iframe]**2)
            part2 = np.nansum(med * amp_model[0] * valid / err[iframe]**2)
            trace_corr[iframe] = part1 / part2
            # -----------------------------------------------------------------
            # plot the trace correction sample plot
            if iframe == 0:
                plots.trace_correction_sample(self.params, iframe,
                                              cube, recon,
                                              x_trace_pos, y_trace_pos,
                                              x_order0, y_order0)
            # -----------------------------------------------------------------
            # push amplitudes into outputs
            for amp_it in range(len(amp_model)):
                # get the amp name
                amp_name = output_names[amp_it]
                amp_err_name = output_names[amp_it] + '_error'
                # put into storage
                outputs[amp_name][iframe] = amp_model[amp_it]
                outputs[amp_err_name][iframe] = err_model[amp_it]
            # -----------------------------------------------------------------
            # update the cube
            cube[iframe] -= recon
            # update the recons (relative to the no model amplitude)
            all_recon[iframe] = recon / amp_model[0]
            # -----------------------------------------------------------------
            # force cube to floats
            tmp_slice = np.array(cube[iframe], dtype=float)
            # calculate the rms of the recon cube
            outputs['rms_cube_recon'][iframe] = mp.estimate_sigma(tmp_slice)
            # keep the valid mask for later
            valid_arr[iframe] = valid
        # ---------------------------------------------------------------------
        # normalize the trace correction by the median
        trace_corr /= np.nanmedian(trace_corr)
        # push into outputs
        outputs['amplitude_uncorrected'] = np.array(outputs['amplitude'])
        outputs['aperture_correction'] = trace_corr
        # update the amplitudes by the trace correction
        outputs['amplitude'] = outputs['amplitude'] * trace_corr
        # ---------------------------------------------------------------------
        # plot the aperture correction plot
        plots.aperture_correction_plot(self.params, outputs, trace_corr)
        # ---------------------------------------------------------------------
        # convert outputs to an astropy table
        output_table = Table(outputs)
        # return the outputs
        return output_table, all_recon, valid_arr

    def normalize_sum_trace(self, loutputs: Table) -> Table:
        """
        Normalize the sum trace by the median of the sum trace

        :param loutputs: dict, the outputs

        :return: dict, the normalized outputs
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.normalize_sum_trace()'
        # validate out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # if we don't have oot domain we cannot do the normalization
        if not has_oot:
            wmsg = ('Cannot normalize sum trace without out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to normalize by the sum of '
                    'the trace.')
            misc.printc(wmsg, 'warning')
            # return loutputs without normalization
            return loutputs
        # ---------------------------------------------------------------------
        # get the normalization factor
        with warnings.catch_warnings(record=True) as _:
            norm_factor = np.nanmedian(loutputs[oot_domain]['sum_trace'])
        # apply the normalization factor
        loutputs['sum_trace'] /= norm_factor
        loutputs['sum_trace_error'] /= norm_factor
        # return the outputs
        return loutputs

    def per_pixel_baseline(self, cube: np.ndarray,
                           valid_arr: np.ndarray) -> np.ndarray:
        """
        Correct the per pixel baseline

        :param cube: np.ndarray, the cube
        :param valid: np.ndarray, the valid pixels

        :return: np.ndarray, the corrected cube
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.per_pixel_baseline()'
        # get params
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # ---------------------------------------------------------------------
        # copy the cube
        cube = np.array(cube)
        # get the frame numbers
        frames = np.arange(nframes, dtype=float)
        # get the out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # if we don't have oot domain we cannot do the normalization
        if not has_oot:
            wmsg = ('Cannot do per pixel baseline correction without '
                    'out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to do per pixel baseline'
                    ' correction.')
            misc.printc(wmsg, 'warning')
            # return loutputs without normalization
            return cube
        # ---------------------------------------------------------------------
        # get the polynomial degree for the transit baseline
        poly_order = self.params['TRANSIT_BASELINE_POLYORD']
        # get the mid transit frame
        mid_transit_frame = int(np.nanmean(self.params['CONTACT_FRAMES']))
        # get the image for the mid transit frame
        mid_transit_slice = (cube[mid_transit_frame] *
                             valid_arr[mid_transit_frame])
        # get the rms of the cube (before correction) for mid transit
        rms1_cube = mp.estimate_sigma(mid_transit_slice)
        # ---------------------------------------------------------------------
        # print progress
        msg = 'Correcting the per pixel transit baseline'
        misc.printc(msg, 'info')
        # ---------------------------------------------------------------------
        # loop around x pix
        for ix in tqdm(range(nbxpix), leave=False):
            # get the slice of the cube
            cube_slice = cube[:, :, ix]
            # if we don't have any valid pixels skip
            if np.sum(valid_arr[:, :, ix]) == 0:
                continue
            # loop around y pix
            for iy in range(nbypix):
                # if pixel is not valid we skip
                if  np.sum(valid_arr[:, iy, ix]) == 0:
                    continue
                # get the sample column
                sample = cube_slice[:, iy]
                # get the out of transit domain in the sample
                sample_oot = sample[oot_domain, :]
                # get the indices of the out of transit domain
                frames_oot = frames[oot_domain]
                # find any nans in the oot sample
                finite_mask = np.isfinite(sample_oot)
                # -------------------------------------------------------------
                # only fit the polynomial if we have enough points
                if np.sum(finite_mask) <= poly_order:
                    continue
                # -------------------------------------------------------------
                # deal with having NaNs in the sample
                if np.sum(~finite_mask) > 0:
                    # remove non-finite values
                    sample_oot = sample_oot[finite_mask]
                    frames_oot = frames_oot[finite_mask]
                # -------------------------------------------------------------
                # robustly fit the out of transit part
                oot_fit, _ = mp.robust_polyfit(frames_oot, sample_oot,
                                               poly_order, 5)
                # -------------------------------------------------------------
                # subtract this off the cube_slice
                cube_slice[:, iy] -= np.polyval(oot_fit, frames)
            # -----------------------------------------------------------------
            # push the updated cube slice back into the cube
            cube[:, :, ix] = np.array(cube_slice)

        # ---------------------------------------------------------------------
        # re-get the image for the mid transit frame
        mid_transit_slice = (cube[mid_transit_frame] *
                             valid_arr[mid_transit_frame])
        # recalculate the rms of the cube
        rms2_cube = mp.estimate_sigma(mid_transit_slice)
        # print the mid transit frame used
        msg = f'\tMid transit frame used: {mid_transit_frame}'
        misc.printc(msg, 'info')
        # print the rms of the cube before and after
        msg_before = f'\tRMS[before]: {rms1_cube:.3f}'
        misc.printc(msg_before, 'number')
        msg_after = f'\tRMS[after]: {rms2_cube:.3f}'
        misc.printc(msg_after, 'number')
        # ---------------------------------------------------------------------
        # return the updated cube
        return cube

    def get_rms_baseline(self, vector: Union[np.ndarray, None] = None,
                         method: str = 'linear_sigma'
                         ) -> Union[float, List[str]]:
        """
        Get the RMS of the baseline (for a given method)

        Available methods are:
        - naive_sigma: naive sigma method
        - linear_sigma: linear sigma method
        - lowpass_sigma: lowpass sigma method
        - quadratic_sigma: quadratic sigma method

        vector set to None returns the methods

        :param vector: np.ndarray or None, the vector to calculate the RMS
                       or None to return the methods
        :param method: str, the method to use

        :return: float or list, the RMS of the vector or the methods
                 (if vector=None)
        """
        # deal with just getting the baseline methods
        if vector is None:
            return ['naive_sigma', 'linear_sigma', 'lowpass_sigma',
                    'quadratic_sigma']
        # ---------------------------------------------------------------------
        # native method: we don't know that there's a transit
        if method == 'naive_sigma':
            # get the percentiles
            p16_84 = np.nanpercentile(vector, [16, 84])
            # return the result of half the difference between the two
            return (p16_84[1] - p16_84[0]) / 2.0
        # ---------------------------------------------------------------------
        # linear sigma method: difference to immediate neighbours
        if method == 'linear_sigma':
            # calculate the difference between immediate neighbours
            diff = vector[1: -1] - (vector[2:] + vector[:-2]) / 2
            # calculate the sigma of the diff
            nsig = mp.estimate_sigma(diff)
            # normalize by 1 + 0.5 and return
            return nsig / np.sqrt(1 + 0.5)
        # ---------------------------------------------------------------------
        # lowpass sigma method: difference to lowpass filtered
        if method == 'lowpass_sigma':
            # calculate the low pass filter
            lowpass = mp.lowpassfilter(vector, width=15)
            # subtract off the low pass filter
            diff = vector - lowpass
            # return the sigma of the diff
            return mp.estimate_sigma(diff)
        # ---------------------------------------------------------------------
        # quadratic sigma method: difference to quadratic fit
        if method == 'quadratic_sigma':
            # calculate the vector rolled
            roll1 = np.roll(vector, -2)
            roll2 = np.roll(vector, -1)
            roll3 = np.roll(vector, 1)
            # get the contributions from each roll
            vector2 = -roll1/3 + roll2 + roll3/3
            # get the diff between the two vectors
            diff = vector - vector2
            # calculate the sigma of the diff
            nsig = mp.estimate_sigma(diff)
            # normalize by sqrt(20 / 9.0) and return
            return nsig / np.sqrt(20 / 9.0)

    def get_effective_wavelength(self) -> Tuple[float, float]:
        """
        Get the effective wavelength factors (photon and energy weights)

        :return: tuple, 1. the photon weighted mean, 2. the energy weighted mean
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_effective_wavelength()'
        # get the median file
        medfile = self.get_variable('MEIDAN_IMAGE_FILE', func_name)
        # load the median from disk
        med = io.load_fits(medfile)
        # get the tracemap
        tracemap = self.get_trace_map()
        # get the wave grid
        wavegrid = self.get_wavegrid(order=1)
        # ---------------------------------------------------------------------
        # find the domain that has spectra
        with warnings.catch_warnings(record=True) as _:
            sp_domain = np.nanmean(tracemap * med, axis=0)
        # work out the spectral energy distribution
        sp_energy = sp_domain / wavegrid
        # ---------------------------------------------------------------------
        # work out the mean photon weighted wavegrid
        part1a = np.nansum(sp_domain * wavegrid)
        part2a = np.nansum(sp_domain * np.isfinite(wavegrid))
        mean_photon_weighted = part1a / part2a
        # ---------------------------------------------------------------------
        # work out the mean energy weighted wavegrid
        part1b = np.nansum(sp_energy * wavegrid)
        part2b = np.nansum(sp_energy * np.isfinite(wavegrid))
        mean_energy_weighted = part1b / part2b
        # ---------------------------------------------------------------------
        # deal with WLC domain
        if self.params['WLC_DOMAIN'] is not None:
            msg = 'Domain:\t {0:.3f} -- {1:.3f} um'
            margs = self.params['WLO_DOMAIN']
            misc.printc(msg.format(*margs), 'number')
        else:
            msg = 'Full domain included, parameter WLC_DOMAIN not defined'
            misc.printc(msg, 'warning')
        # ---------------------------------------------------------------------
        # print the energy-weighted mean
        msg = 'Energy-weighted mean:\t{0:.3f} um'
        misc.printc(msg.format(mean_energy_weighted), 'number')
        # print the photon-weighted mean
        msg = 'Photon-weighted mean:\t{0:.3f} um'
        misc.printc(msg.format(mean_photon_weighted), 'number')
        # ----------------------------------------------------------------
        # return the effective wavelength factors
        return mean_photon_weighted, mean_energy_weighted

    # ==========================================================================
    # Spectral Extraction functionality
    # ==========================================================================
    def create_sed(self, trace_order: int) -> Table:
        # set the function name
        func_name = f'{__NAME__}.{self.name}.create_sed()'
        # load the median image
        med_file = self.get_variable('MEIDAN_IMAGE_FILE', func_name)
        med = io.load_fits(med_file)
        # load the residuals
        res_file = self.get_variable('WLC_RES_FILE', func_name)
        residual = io.load_fits(res_file)
        # for future reference in the code, we keep track of data size
        self.set_variable('DATA_X_SIZE', residual.shape[2], func_name)
        self.set_variable('DATA_Y_SIZE', residual.shape[1], func_name)
        self.set_variable('DATA_N_FRAMES', residual.shape[0], func_name)
        # ---------------------------------------------------------------------
        # get the trace position
        posmax, throughput = self.get_trace_pos(order_num=trace_order)
        # get wave grid
        wavegrid = self.get_wavegrid(order_num=trace_order)
        # get clean median trace for spectrum
        dx, dy, rotxy, ddy, med_clean = self.get_gradients(med)
        # construct the sed
        sp_sed = np.zeros(med.shape[1])
        # get a ribbon on the trace that extends over the input width
        for ix in range(med.shape[1]):
            # get width
            width = self.params['TRACE_WIDTH_EXTRACTION'] // 2
            # get start and end positions
            ystart = posmax[ix] - width
            yend = posmax[ix] + width
            # sum to get the spectrum SED
            sp_sed[ix] = np.nansum(med_clean[ystart:yend, ix])
        # ---------------------------------------------------------------------
        # plot the SED
        plots.plot_sed(self.params, wavegrid, sp_sed / throughput, trace_order)
        # ---------------------------------------------------------------------
        # construct sed table name
        sed_table_name = self.get_variable('SED_TABLE_FILE', func_name)
        sed_table_name = sed_table_name.format(self.params['OBJECTNAME'],
                                               trace_order)
        # construct SED table
        sed_table = Table()
        sed_table['wavelength'] = wavegrid
        sed_table['flux'] = sp_sed / throughput
        sed_table['raw flux'] = sp_sed
        sed_table['throughput'] = throughput
        # save table to disk
        io.save_table(sed_table_name, sed_table)
        # return this table
        return sed_table

    def load_model(self) -> np.ndarray:
        """
        load the model (and deal with masking order zero if required)

        :return:
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.create_model()'
        # load the median image
        med_file = self.get_variable('MEIDAN_IMAGE_FILE', func_name)
        med = io.load_fits(med_file)
        # load the residuals
        recon_file = self.get_variable('WLC_RECON_FILE', func_name)
        model = io.load_fits(recon_file)
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # load the tracemap
        tracemap = self.get_trace_map()
        # ---------------------------------------------------------------------
        # deal with masking order zero
        if self.params['MASK_ORDER_0']:
            # load the mask trace position
            mask_trace_pos, _, _, _, _ = self.get_mask_trace_pos(med, tracemap)
            # need to re-get the mask order zero
            mask_order0, xpos, ypos = self.get_mask_order0(mask_trace_pos,
                                                           tracemap)
            # loop around frames and mask out order zero (with NaNs)
            for iframe in tqdm(nbframes, leave=False):
                # set the order zero values to nan
                model[iframe][mask_order0] = np.nan
        # ---------------------------------------------------------------------
        # return the model
        return model

    def ratio_residual_to_trace(self, model: np.ndarray, trace_order: int
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        We find the ratio of residual to trace.

        The logic here is that  we want to know the mean ratio of the
        residual to the model trace to do this, we divide the residual by the
        trace and take the weighted mean.

        The thing is that the trace has a very structured shape, so this
        needs to include a propagation of errors. There error of pixels along
        the trace profile are the input errors divided by the model trace.
        Pixels with a very low trace value have correspondingly larger errors.

        :param model:
        :param trace_order:

        :return:
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.ratio_residual_to_trace()'
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # ---------------------------------------------------------------------
        # get the trace position
        posmax, throughput = self.get_trace_pos(order_num=trace_order)
        # load the residuals
        res_file = self.get_variable('WLC_RES_FILE', func_name)
        residual = io.load_fits(res_file)
        # load the error file
        err_file = self.get_variable('WLC_ERR_FILE', func_name)
        err = io.load_fits(err_file)
        # ---------------------------------------------------------------------
        # placeholder for the cube spectra
        spec = np.full([nbframes, nbxpix], np.nan)
        spec_err = np.full([nbframes, nbxpix], np.nan)
        # loop through observations and spectral bins
        for iframe in tqdm(range(nbframes), leave=False):
            for ix in range(nbxpix):
                # get width
                width = self.params['TRACE_WIDTH_EXTRACTION'] // 2
                # get start and end positions
                ystart = posmax[ix] - width
                yend = posmax[ix] + width
                # model of the trace for that observation
                v0 = model[iframe, ystart:yend, ix]
                # residual of the trace
                v1 = residual[iframe, ystart:yend, ix]
                # corresponding error
                v2 = err[iframe, ystart:yend, ix]
                # calculate the ratio
                with warnings.catch_warnings(record=True) as _:
                    try:
                        ratio, err_ratio = mp.odd_ratio_mean(v1/v0, v2/v0)
                    except Exception as _:
                        ratio, err_ratio = np.nan, 0
                # the above code does the eqiuvalent of a sigma-clipped mean
                # and returns the uncertainty
                if err_ratio != 0:
                    # update the cube
                    spec[iframe, ix] = ratio
                    spec_err[iframe, ix] = err_ratio
        # ---------------------------------------------------------------------
        # return the spectrum and corresponding error
        return spec, spec_err

    def remove_trend(self, spec: np.ndarray):
        # set function name
        func_name = f'{__NAME__}.{self.name}.binary_transit_masks()'
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # get the polynomial degree for trace baseline
        polydeg = self.params['TRACE_BASELINE_POLYORD']
        # get the out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # deal with no out-of-transit domain defined
        if not has_oot:
            wmsg = ('Cannot remove trend without '
                    'out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to remove_trend.')
            misc.printc(wmsg, 'warning')
        # ---------------------------------------------------------------------
        # loop around
        for ix in range(nbxpix):
            # get the slide without the transit
            v1 = spec[oot_domain, ix]
            # find valid pixels
            valid = np.isfinite(v1)
            # if we don't have enough good pixels skip removing trend for this
            #  row
            if np.sum(valid) < 2:
                continue
            # get the valid xpix for this row
            index = np.arange(nbframes)[oot_domain]
            # fit the trend
            tfit = np.polyfit(index[valid], v1[valid], polydeg)
            # remove the trend and update the spectrum
            spec[:, ix] -= np.polyval(tfit, np.arange(nbframes))
        # ---------------------------------------------------------------------
        # return the spectrum
        return spec





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
