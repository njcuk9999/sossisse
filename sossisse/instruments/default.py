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
from tqdm import tqdm
from scipy.ndimage import shift
from scipy.signal import convolve2d
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from wpca import EMPCA

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.core import math as mp

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
        self._variables['TEMP_PCAS'] = None
        self._variables['TEMP_TRANSIT_IN_VS_OUT'] = None
        # true/false flags
        self._variables['DO_BACKGROUND'] = None
        self._variables['FLAG_CDS'] = None
        # meta data
        self._variables['TAG1'] = None
        self._variables['TAG2'] = None
        self._variables['META'] = None
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
        # temp files
        self.vsources['TEMP_INI_CUBE'] = f'{self.name}.load_data_with_dq()'
        self.vsources['TEMP_INI_ERR'] = f'{self.name}.load_data_with_dq()'
        self.vsources['TEMP_CLEAN_NAN'] = f'{self.name}.patch_isolated_bads()'
        self.vsources['MEIDAN_IMAGE_FILE'] = f'{self.name}.clean_1f()'
        self.vsources['CLEAN_CUBE_FILE'] = f'{self.name}.clean_1f()'
        self.vsources['TEMP_BEFORE_AFTER_CLEAN1F'] = f'{self.name}.clean_1f()'
        self.vsources['TEMP_PCAS'] = f'{self.name}.clean_1f()'
        self.vsources['TEMP_TRANSIT_IN_VS_OUT'] = f'{self.name}.clean_1f()'
        # true/false flags
        self.vsources['DO_BACKGROUND'] = f'{self.name}.remove_background()'
        self.vsources['FLAG_CDS'] = f'{self.name}.load_data_with_dq()'
        # meta data
        self.vsources['TAG1'] = f'{self.name}.update_meta_data()'
        self.vsources['TAG2'] = f'{self.name}.update_meta_data()'
        self.vsources['META'] = f'{self.name}.update_meta_data()'
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

    def set_variable(self, key, value):
        """
        Set a variable in the variables dictionary

        :param key: str, the key to set
        :param value: any, the value to set
        :return:
        """
        if key in self._variables and key in self.vsources:
            self._variables[key] = value
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

    def load_data(self, filename: str, ext: int = None, extname: str = None):
        """
        Load the data from a file

        :param filename: str, the filename to load
        :param ext: int, the extension number to load
        :param extname: str, the extension name to load

        :return: data, the loaded data
        """
        _ = self
        # try to get data from filename
        try:
            data = fits.getdata(filename, ext, extname)
        except Exception as e:
            emsg = 'Error loading data from file: {0}\n\t{1}: {2}'
            eargs = [filename, type(e), str(e)]
            raise exceptions.SossisseFileException(emsg.format(*eargs))
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
        try:
            data = Table.read(filename, ext)
        except Exception as e:
            emsg = 'Error loading table from file: {0}\n\t{1}: {2}'
            eargs = [filename, type(e), str(e)]
            raise exceptions.SossisseFileException(emsg.format(*eargs))
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
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = 'temporary_initial_cube.fits'
        temp_ini_cube = os.path.join(self.params['TEMP_PATH'], temp_ini_cube)
        temp_ini_err = 'temporary_initial_err.fits'
        temp_ini_err = os.path.join(self.params['TEMP_PATH'], temp_ini_err)
        # save these for later
        self.set_variable('TEMP_INI_CUBE', temp_ini_cube)
        self.set_variable('TEMP_INI_ERR', temp_ini_err)
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
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_clean_nan = 'temporary_cleaned_isolated.fits'
        temp_clean_nan = os.path.join(self.params['TEMP_PATH'], temp_clean_nan)
        # save these for later
        self.set_variable('TEMP_CLEAN_NAN', temp_clean_nan)
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

    def get_trace_positions(self) -> np.ndarray:
        raise NotImplementedError('get_trace_pos() must be implemented in '
                                  'child Instrument class')


    def get_trace_map(self) -> np.ndarray:
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
                     order_num: Union[int, None] = None):
        """
        Get the wave grid for the instrument

        :param source: str, the source of the wave grid
        :param order_num: int, the order number to use (if source is pos)

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
        # return the wave grid
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
        # update meta data
        self.update_meta_data()
        # get tag1
        tag1 = self.get_variable('TAG1', func_name)
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # ---------------------------------------------------------------------
        # construct temporary file names
        median_image_file = 'median{0}.fits'.format(tag1)
        median_image_file = os.path.join(self.params['TEMP_PATH'],
                                         median_image_file)
        clean_cube_file = 'cleaned_cube{0}.fits'.format(tag1)
        clean_cube_file = os.path.join(self.params['TEMP_PATH'],
                                        clean_cube_file)
        tmp_before_after_clean1f = 'temporary_before_after_clean1f.fits'
        tmp_before_after_clean1f = os.path.join(self.params['TEMP_PATH'],
                                                tmp_before_after_clean1f)
        tmp_pcas = 'temporary_pcas.fits'.format(tag1)
        tmp_pcas = os.path.join(self.params['TEMP_PATH'], tmp_pcas)


        temp_transit_invsout = 'temporary_transit_in_vs_out.fits'
        tmp_transit_invsout = os.path.join(self.params['TEMP_PATH'],
                                           temp_transit_invsout)
        # ---------------------------------------------------------------------
        # save these for later
        self.set_variable('MEDIAN_IMAGE_FILE', median_image_file)
        self.set_variable('CLEAN_CUBE_FILE', clean_cube_file)
        self.set_variable('TEMP_BEFORE_AFTER_CLEAN1F', tmp_before_after_clean1f)
        self.set_variable('TEMP_PCAS', tmp_pcas)
        self.set_variable('TEMP_TRANSIT_IN_VS_OUT', tmp_transit_invsout)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            # make sure all required files exist
            cond = os.path.exists(median_image_file)
            cond &= os.path.exists(clean_cube_file)
            cond &= os.path.exists(tmp_before_after_clean1f)
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
                misc.printc('\tReading: {0}'.format(tmp_before_after_clean1f),
                            'info')
                before_after_clean1f = self.load_data(tmp_before_after_clean1f)
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
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        oot_domain_before = self.get_variable('OOT_DOMAIN_BEFORE', func_name)
        oot_domain_after = self.get_variable('OOT_DOMAIN_AFTER', func_name)
        int_domain = self.get_variable('INT_DOMAIN', func_name)
        # get flag for median out of transit
        med_oot = self.params['MEDIAN_OOT']
        # ---------------------------------------------------------------------
        # deal  with creating median
        with warnings.catch_warnings:
            if med_oot:
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
        # get the number of frames
        data_n_frames = self.get_variable('DATA_N_FRAMES', func_name)
        # get the contact points
        cframes = self.params['CONTACT_FRAMES']
        # get the rejection domain
        rej_domain = self.params['REJECT_DOMAIN']
        # deal with value already found
        if oot_domain is not None:
            return
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
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        # only fit the pca to the flux in the trace (nan everything else)
        nanmask = np.ones_like(tracemap, dtype=float)
        nanmask[~tracemap] = np.nan
        # copy the normalized cube
        cube3ini = cube2[oot_domain]
        # subtract off the median
        for iframe in tqdm(range(cube3ini.shape[0]), leave=False):
            cube3ini[iframe] -= med
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
        # find out the regions that are valid at least 95% of the time
        with warnings.catch_warnings(record=True) as _:
            valid = np.where(np.nanmean(weights != 0, axis=0) > 0.95)[0]
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

        #

        # get the components



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
