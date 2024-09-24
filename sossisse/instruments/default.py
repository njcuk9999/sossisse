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
        self._variables['TEMP_INI_CUBE_BKGRND'] = None
        self._variables['TEMP_INI_ERR_BKGRND'] = None
        self._variables['TEMP_CLEAN_NAN'] = None
        self._variables['MEDIAN_IMAGE_FILE'] = None
        self._variables['CLEAN_CUBE_FILE'] = None
        self._variables['TEMP_BEFORE_AFTER_CLEAN1F'] = None
        self._variables['TEMP_PCA_FILE'] = None
        self._variables['TEMP_TRANSIT_IN_VS_OUT'] = None
        # WLC files
        self._variables['WLC_ERR_FILE'] = None
        self._variables['WLC_RES_FILE'] = None
        self._variables['WLC_RECON_FILE'] = None
        self._variables['WLC_LTBL_FILE'] = None
        # spectral extraction files
        self._variables['SPE_SED_TBL'] = None
        self._variables['RES_NO_GREY_ORD'] = None
        self._variables['RES_GREY_ORD'] = None
        self._variables['SPECTRA_ORD'] = None
        self._variables['WAVE_ORD'] = None
        self._variables['TSPEC_ORD'] = None
        self._variables['TSPEC_ORD_BIN'] = None
        self._variables['EUREKA_FILE'] = None
        # true/false flags
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
        self._variables['PHOTO_WEIGHTED_MEAN'] = None
        self._variables['ENERGY_WEIGHTED_MEAN'] = None
        # ---------------------------------------------------------------------
        # define source for variables
        self.vsources = dict()
        # array sizes
        self.vsources['DATA_X_SIZE'] = f'{self.name}.load_data_with_dq()'
        self.vsources['DATA_Y_SIZE'] = f'{self.name}.load_data_with_dq()'
        self.vsources['DATA_N_FRAMES'] = f'{self.name}.load_data_with_dq()'
        # construct temporary file name function
        define_func = f'{self.name}.define_filenames()'
        # temp files
        self.vsources['TEMP_INI_CUBE'] = define_func
        self.vsources['TEMP_INI_ERR'] = define_func
        self.vsources['TEMP_INI_CUBE_BKGRND'] = define_func
        self.vsources['TEMP_INI_ERR_BKGRND'] = define_func
        self.vsources['TEMP_CLEAN_NAN'] = define_func
        self.vsources['MEDIAN_IMAGE_FILE'] = define_func
        self.vsources['CLEAN_CUBE_FILE'] = define_func
        self.vsources['TEMP_BEFORE_AFTER_CLEAN1F'] = define_func
        self.vsources['TEMP_PCA_FILE'] = define_func
        self.vsources['TEMP_TRANSIT_IN_VS_OUT'] = define_func
        # WLC files
        self.vsources['WLC_ERR_FILE'] = define_func
        self.vsources['WLC_RES_FILE'] = define_func
        self.vsources['WLC_RECON_FILE'] = define_func
        self.vsources['WLC_LTBL_FILE'] = define_func
        # spectral extraction files
        self.vsources['SPE_SED_TBL'] = define_func
        self.vsources['RES_NO_GREY_ORD'] = define_func
        self.vsources['RES_GREY_ORD'] = define_func
        self.vsources['SPECTRA_ORD'] = define_func
        self.vsources['WAVE_ORD'] = define_func
        self.vsources['TSPEC_ORD'] = define_func
        self.vsources['TSPEC_ORD_BIN'] = define_func
        self.vsources['EUREKA_FILE'] = define_func
        # true/false flags
        self.vsources['FLAG_CDS'] = f'{self.name}.id_image_shape()'
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
        self.vsources['PHOTO_WEIGHTED_MEAN'] = f'{self.name}.get_effective_wavelength()'
        self.vsources['ENERGY_WEIGHTED_MEAN'] = f'{self.name}.get_effective_wavelength()'

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
        if self._variables['FLAG_CDS'] is not None:
            # only if true
            if self._variables['FLAG_CDS']:
                # deal with FLAG_CDS = True but CDS_IDS None
                if self.params['CDS_IDS'] is None:
                    emsg = ('CDS_IDS not set but CDS flag is set to True.'
                            'If input data is not a CDS please set CDS_IDS.')
                    raise exceptions.SossisseConstantException(emsg)
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
        if self.params['DO_BACKGROUND']:
            # get variable
            do_background = self.params['DO_BACKGROUND']
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
        # get photo weighted mean and energy weighted mean
        # ---------------------------------------------------------------------
        phot_wmn = self._variables['PHOTO_WEIGHTED_MEAN']
        ener_wmn = self._variables['ENERGY_WEIGHTED_MEAN']
        # Add the photo weighted mean 
        if phot_wmn is not None:
            meta_data['PHOT_WMN'] = (phot_wmn, 'Photo weighted mean in um')
        # Add the energy weighted mean
        if ener_wmn is not None:
            meta_data['ENER_WMN'] = (ener_wmn, 'Energy weighted mean in um')
        # ---------------------------------------------------------------------
        # push tag1, tag2 and meta data to variables
        self.set_variable('TAG1', tag1)
        self.set_variable('TAG2', tag2)
        self.set_variable('META', meta_data)

    def define_filenames(self):
        # set function name
        func_name = f'{__NAME__}.define_filenames()'
        # update meta data
        self.update_meta_data()
        # get tag1
        tag1 = self.get_variable('TAG1', func_name)
        tag2 = self.get_variable('TAG2', func_name)
        # get file paths
        temppath = self.params['TEMP_PATH']
        otherpath = self.params['OTHER_PATH']
        fitspath = self.params['FITS_PATH']
        # ---------------------------------------------------------------------
        # construct temporary file names
        # ---------------------------------------------------------------------
        median_image_file = 'median{0}.fits'.format(tag1)
        median_image_file = os.path.join(temppath, median_image_file)
        # ---------------------------------------------------------------------
        clean_cube_file = 'cleaned_cube{0}.fits'.format(tag1)
        clean_cube_file = os.path.join(temppath, clean_cube_file)
        # ---------------------------------------------------------------------
        tmp_before_after_clean1f = 'temporary_before_after_clean1f.fits'
        tmp_before_after_clean1f = os.path.join(temppath,
                                                tmp_before_after_clean1f)
        # ---------------------------------------------------------------------
        tmp_pcas = 'temporary_pcas.fits'.format(tag1)
        tmp_pcas = os.path.join(temppath, tmp_pcas)
        # ---------------------------------------------------------------------
        temp_transit_invsout = 'temporary_transit_in_vs_out.fits'
        tmp_transit_invsout = os.path.join(temppath, temp_transit_invsout)
        # ---------------------------------------------------------------------
        temp_clean_nan = 'temporary_cleaned_isolated.fits'
        temp_clean_nan = os.path.join(temppath, temp_clean_nan)
        # ---------------------------------------------------------------------
        temp_ini_cube = 'temporary_initial_cube.fits'
        temp_ini_cube = os.path.join(temppath, temp_ini_cube)
        # ---------------------------------------------------------------------
        temp_ini_err = 'temporary_initial_err.fits'
        temp_ini_err = os.path.join(temppath, temp_ini_err)
        # ---------------------------------------------------------------------
        tmp_ini_cube_bkgrnd = 'temporary_initial_cube_bkgrnd.fits'
        tmp_ini_cube_bkgrnd = os.path.join(temppath, tmp_ini_cube_bkgrnd)
        # ---------------------------------------------------------------------
        tmp_ini_err_bkgrnd = 'temporary_initial_err_bkgrnd.fits'
        tmp_ini_err_bkgrnd = os.path.join(temppath, tmp_ini_err_bkgrnd)
        # ---------------------------------------------------------------------
        errfile = os.path.join(temppath, 'errormap{}.fits'.format(tag1))
        # ---------------------------------------------------------------------
        resfile = os.path.join(temppath, 'residual{}.fits'.format(tag1))
        # ---------------------------------------------------------------------
        reconfile = os.path.join(temppath, 'recon{}.fits'.format(tag1))
        # ---------------------------------------------------------------------
        ltbl_file = os.path.join(otherpath, 'stability{}.csv'.format(tag1))
        # ---------------------------------------------------------------------
        sed_table = os.path.join(otherpath, 'sed_{objname}_ord{trace_order}.csv')
        # ---------------------------------------------------------------------
        res_no_grey_ord = 'residual_no_grey_ord{trace_order}' + tag2 + '.fits'
        res_no_grey_ord = os.path.join(fitspath, res_no_grey_ord)
        # ---------------------------------------------------------------------
        res_grey_ord = 'residual_grey_ord{trace_order}' + tag2 + '.fits'
        res_grey_ord = os.path.join(fitspath, res_grey_ord)
        # ---------------------------------------------------------------------
        spectra_ord = 'spectra_ord{trace_order}' + tag2 + '.fits'
        spectra_ord = os.path.join(fitspath, spectra_ord)
        # ---------------------------------------------------------------------
        waveord_file = 'wavelength_ord{trace_order}' + tag2 + '.fits'
        waveord_file = os.path.join(fitspath, waveord_file)
        # ---------------------------------------------------------------------
        tspec_ord = 'tspec_ord{trace_order}' + tag2 + '.csv'
        tspec_ord = os.path.join(otherpath, tspec_ord)
        # ---------------------------------------------------------------------
        tspec_ord_bin = 'tspec_ord_{trace_order}_bin{res}' + tag2 + '.csv'
        tspec_ord_bin = os.path.join(otherpath, tspec_ord_bin)
        # ---------------------------------------------------------------------
        eureka_file = 'spectra_ord{trace_order}' + tag2 + '.h5'
        # ---------------------------------------------------------------------
        # temp files
        self.set_variable('MEDIAN_IMAGE_FILE', median_image_file)
        self.set_variable('CLEAN_CUBE_FILE', clean_cube_file)
        self.set_variable('TEMP_BEFORE_AFTER_CLEAN1F', tmp_before_after_clean1f)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        self.set_variable('TEMP_TRANSIT_IN_VS_OUT', tmp_transit_invsout)
        self.set_variable('TEMP_CLEAN_NAN', temp_clean_nan)
        self.set_variable('TEMP_INI_CUBE', temp_ini_cube)
        self.set_variable('TEMP_INI_ERR', temp_ini_err)
        self.set_variable('TEMP_INI_CUBE_BKGRND', tmp_ini_cube_bkgrnd)
        self.set_variable('TEMP_INI_ERR_BKGRND', tmp_ini_err_bkgrnd)
        # WLC files
        self.set_variable('WLC_ERR_FILE', errfile)
        self.set_variable('WLC_RES_FILE', resfile)
        self.set_variable('WLC_RECON_FILE', reconfile)
        self.set_variable('WLC_LTBL_FILE', ltbl_file)
        self.set_variable('SPE_SED_TBL', sed_table)
        # spectral extraction files
        self.set_variable('RES_NO_GREY_ORD', res_no_grey_ord)
        self.set_variable('RES_GREY_ORD', res_grey_ord)
        self.set_variable('SPECTRA_ORD', spectra_ord)
        self.set_variable('WAVE_ORD', waveord_file)
        self.set_variable('TSPEC_ORD', tspec_ord)
        self.set_variable('TSPEC_ORD_BIN', tspec_ord_bin)
        self.set_variable('EUREKA_FILE', eureka_file)

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

    def load_table(self, filename: str, fmt: str = None, ext: int = None):
        """
        Load the table from a file

        :param filename: str, the filename to load
        :param fmt: str, the format of the file
        :param ext: int, the extension number to load

        :return: data, the loaded data
        """
        _ = self
        # default is to just load the table file
        data = io.load_table(filename, fmt=fmt, hdu=ext)
        # return the data
        return data

    def load_cube(self, n_slices: int, image_shape: Tuple[int, int, int],
                  flag_cds: bool
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # create the containers for the cube of science data,
        # the error cube, and the DQ cube
        cube = np.zeros([n_slices, image_shape[0], image_shape[1]])
        err = np.zeros([n_slices, image_shape[0], image_shape[1]])
        dq = np.zeros([n_slices, image_shape[0], image_shape[1]])
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

    def load_data_with_dq(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                # print that we are reading files
                misc.printc('Reading temporary file: {0}'.format(temp_ini_cube),
                            'info')
                misc.printc('Reading temporary file: {0}'.format(temp_ini_err),
                            'info')
                # load the data
                cube = self.load_data(temp_ini_cube)
                err = self.load_data(temp_ini_err)
                # for future reference in the code, we keep track of data size
                self.set_variable('DATA_X_SIZE', cube.shape[2])
                self.set_variable('DATA_Y_SIZE', cube.shape[1])
                self.set_variable('DATA_N_FRAMES', cube.shape[0])
                # return
                return cube, err, None
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
            bin_shape = self.bin_cube(tmp_data, get_shape=True)
            # add to the number of slices
            n_slices += bin_shape[0]
            # store the shape of the raw data
            raw_shapes.append(bin_shape)
            # make sure tmp data is deleted
            del tmp_data
        # ---------------------------------------------------------------------
        # identify cds and get iamge shape
        # ---------------------------------------------------------------------
        image_shape, flag_cds = self.id_image_shape(raw_shapes)
        # ---------------------------------------------------------------------
        # recalculate tags
        self.update_meta_data()
        # ---------------------------------------------------------------------
        # get flat
        flat, no_flat = self.get_flat(image_shape)
        # ---------------------------------------------------------------------
        # load and bin the cube
        cube, err, dq = self.load_cube(n_slices, image_shape, flag_cds)
        # ---------------------------------------------------------------------
        # apply the flat (may be ones)
        if not no_flat:
            # print progress
            misc.printc('Applying flat field to data', 'info')
            # apply the flat field
            for iframe in tqdm(range(cube.shape[0])):
                cube[iframe] /= flat
                err[iframe] /= flat
        # ---------------------------------------------------------------------
        # patch to avoid annoying zeros in error map
        # Question: All instruments?
        with warnings.catch_warnings(record=True) as _:
            err[err == 0] = np.nanmin(err[err != 0])
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
        return cube, err, dq

    def id_image_shape(self, raw_shapes: List[List[int]]
                       ) -> Tuple[List[int], bool]:
        """
        Identify the image shape and if the data is in CDS format
        and return the shape of the images (without the number of slices) and
        with out the number of raw frames

        :param raw_shapes: list of list of ints, the raw shapes of the
                           input data files
        :return:
        """
        # flag for all cds
        all_cds = []
        # storage of image shapes
        image_shapes = []
        # loop around raw shapes
        for raw_shape in raw_shapes:
            # raw data might be a cds - flag this and adjust raw shape accordingly
            if len(raw_shape) == 4:
                flag_cds = True
                raw_shape = raw_shape[:-1]
            else:
                flag_cds = False
            # append flag
            all_cds.append(flag_cds)
            # get the image shape
            image_shapes.append(tuple(raw_shape[1:]))
        # ---------------------------------------------------------------------
        # check if all image shapes are the same
        if len(set(image_shapes)) != 1:
            emsg = 'Inconsistent image shapes:'
            for f_it, filename in enumerate(self.params['FILES']):
                emsg += f'\n\t{filename}: {image_shapes[f_it]}'
            raise exceptions.SossisseFileException(emsg)
        # ---------------------------------------------------------------------
        # check that all are CDS or all are not CDS
        if len(set(all_cds)) != 1:
            emsg = 'Inconsistent CDS format (Eiher all CDS or not):'
            for f_it, filename in enumerate(self.params['FILES']):
                emsg += f'\n\t{filename}: {raw_shapes[f_it]}'
            raise exceptions.SossisseFileException(emsg)
        else:
            flag_cds = all_cds[0]
            # store the flag
            self.set_variable('FLAG_CDS', flag_cds)
        # ---------------------------------------------------------------------
        # return the image shape (as all should be the same now)
        return list(image_shapes[0]), flag_cds



    def bin_cube(self, cube: np.ndarray, bin_type: str = 'Flux',
                 get_shape: bool = False
                 ) -> Union[np.ndarray, List[int]]:
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
            if get_shape:
                return list(cube.shape)
            else:
                return cube
        # don't bin if the number of frames in each bin is 1
        if self.params['DATA_BIN_SIZE'] == 1:
            if get_shape:
                return list(cube.shape)
            else:
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
        bin_shape = [n_bins, cube.shape[1], cube.shape[2]]
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
        for it in tqdm(range(n_bins)):
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

    def get_flat(self, image_shape: List[int]) -> Tuple[np.ndarray, bool]:
        """
        Get the flat field

        :param image_shape: tuple, the shape of the image

        :return: tuple, 1. the flat field, 2. a boolean indicating if the flat
                 field is all ones
        """
        if self.params['FLATFILE'] is None:
            # flat field is a single frame
            return np.ones(image_shape), True
        else:
            # load the flat field
            flat = self.load_data(self.params['FLATFILE'])
            # check the shape of the flat field
            if flat.shape != image_shape:
                emsg = 'Flat field shape does not match data frame shape'
                raise exceptions.SossisseInstException(emsg, self.name)
            # some sanity checks in flat
            flat[flat == 0] = np.nan
            flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
            flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
            # return the flat field
            return flat, False

    def remove_background(self, cube: np.ndarray, err: np.ndarray,
                          dq: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes the background with a 3 DOF model. It's the background image
        times a slope + a DC offset (1 amp, 1 slope, 1 DC)

        Background file is loaded from params['BKGFILE']

        :param cube: np.ndarray, the cube to remove the background from
        :param err: np.ndarray, the error cube
        :param dq: np.ndarray, the data quality cube

        :return: tuple, 1. np.ndarray, the background corrected cube
                        2. np.ndarray, the background corrected error cube
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.remove_background()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = self.get_variable('TEMP_INI_CUBE_BKGRND', func_name)
        temp_ini_err = self.get_variable('TEMP_INI_ERR_BKGRND', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            if os.path.exists(temp_ini_cube) and os.path.exists(temp_ini_err):
                # print that we are reading files
                misc.printc('Reading temporary file: {0}'.format(temp_ini_cube),
                            'info')
                misc.printc('Reading temporary file: {0}'.format(temp_ini_err),
                            'info')
                # load the data
                cube = self.load_data(temp_ini_cube)
                err = self.load_data(temp_ini_err)
                # return
                return cube, err
        # force the cube to be floats (it should be)
        cube = cube.astype(float)
        # deal with not doing background correction
        if not self.params['DO_BACKGROUND']:
            # print progress
            msg = 'We do not clean background. BKGFILE is not set.'
            misc.printc(msg, 'warning')
            # return the cube (unchanged)
            return cube, err
        # update the meta data
        self.update_meta_data()
        # ---------------------------------------------------------------------
        # trick to get a mask where True is valid
        cube_mask = np.zeros_like(cube, dtype=bool)
        for valid_dq in self.params['VALID_DQ']:
            # print DQ values
            misc.printc(f'Accepting DQ = {valid_dq}', 'number')
            # get the mask
            cube_mask[dq == valid_dq] = True
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
        for ishift in tqdm(range(len(bgnd_shifts))):
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
            residuals = mcube - np.polyval(bfit, background2)
            # get the median of the residuals
            med_res = np.nanmedian(residuals[box[2]:box[3], box[0]:box[1]],
                                   axis=0)
            # get the rms
            rms[ishift] = np.std(med_res)

        # find the optimal shift
        rms_min = np.argmin(rms)
        # fit this rms
        bstart = rms_min - 1
        bend = rms_min + 2
        # TODO: Deal with pooly conditioned polyfit (RankWarning)
        with warnings.catch_warnings(record=True) as _:
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
        for frame in tqdm(range(cube.shape[0])):
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
        for iframe in tqdm(range(cube.shape[0])):
            with warnings.catch_warnings(record=True) as _:
                med = np.nanmedian(cube[iframe] * mean_mask, axis=0)
                lowp = mp.lowpassfilter(med, 25)
            # subtract the low pass filter from this frame of the cube
            cubetile = np.tile(lowp, mcube.shape[0]).reshape(mcube.shape)
            cube[iframe] -= cubetile
        # ---------------------------------------------------------------------
        # mask the values in cube_mask
        cube[~cube_mask] = np.nan
        err[~cube_mask] = np.inf
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then save them
        # we re-save these with the background removed
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
        # return the background corrected cube
        return cube, err

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
                       'speed things up. \nReading: {0}')
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
        for iframe in tqdm(range(cube.shape[0])):
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

    def get_trace_positions(self, log: bool = True):
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
        # print progress
        if log:
            misc.printc('Getting the trace map', 'info')
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

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.get_wavegrid()'
        # get x and y size from cube
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        xoffset = self.params['X_TRACE_OFFSET']
        yoffset = self.params['Y_TRACE_OFFSET']
        trace_wid_mask = self.params['TRACE_WIDTH_MASKING']
        # load the trace position
        tbl_ref = self.load_table(self.params['POS_FILE'], ext=order_num)
        # ---------------------------------------------------------------------
        # get columns
        xpos = np.array(tbl_ref['X'])
        ypos = np.array(tbl_ref['Y'])
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
            throughput = np.array(tbl_ref['THROUGHPUT'])
        # ---------------------------------------------------------------------
        # get the valid trace positions
        valid = (xpos > 0) & (xpos < xsize - 1)
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
        posmax = np.array(spline_y(rxpos) - 0.5, dtype=dtype)
        throughput = np.array(spline_throughput(rxpos), dtype=float)
        # ---------------------------------------------------------------------
        # deal with map2d
        if map2d:
            # make a mask of the image
            posmap = np.zeros([ysize, xsize], dtype=bool)
            # loop around pixels in the x direction
            for ix_pix in range(xsize):
                # get the top and bottom of the trace
                bottom = int(posmax[ix_pix] - trace_wid_mask // 2)
                top = int(posmax[ix_pix] + trace_wid_mask // 2)
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

    def get_wavegrid(self, order_num: Union[int, None] = None,
                     return_xpix: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the wave grid for the instrument

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
            tbl_ref = self.load_table(self.params['POS_FILE'], ext=order_num)
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
        # save these for later
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        clean_cube_file = self.get_variable('CLEAN_CUBE_FILE', func_name)
        tmp_before_after_c1f = self.get_variable('TEMP_BEFORE_AFTER_CLEAN1F',
                                                 func_name)
        tmp_pcas = self.get_variable('TEMP_PCA_FILE', func_name)
        tmp_transit_invsout = self.get_variable('TEMP_TRANSIT_IN_VS_OUT',
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
        # print progress
        misc.printc('Cleaning 1/f noise', 'info')
        # create a copy of the cube, we will normalize the amplitude of each
        # trace
        cube2 = np.array(cube, dtype=float)
        # first estimate of the trace amplitude
        misc.printc('\tFirst median of cube to create trace esimate', 'info')
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
        with warnings.catch_warnings(record=True) as _:
            if med_oot and has_oot:
                med = np.nanmedian(cube2[oot_domain], axis=0)
            else:
                med = np.nanmedian(cube2, axis=0)
        # ---------------------------------------------------------------------
        # do a dot product of each trace to the median and adjust amplitude
        #   so that they all match
        amps = np.zeros(nframes)
        # loop around frames
        for iframe in tqdm(range(nframes)):
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
        misc.printc('\tSecond median of cube with proper normalization', 'info')
        # normalize cube
        with warnings.catch_warnings(record=True) as _:
            if med_oot:
                med = np.nanmedian(cube2[oot_domain], axis=0)
                before = np.nanmedian(cube2[oot_domain_before], axis=0)
                after = np.nanmedian(cube2[oot_domain_after], axis=0)
                # get the median difference
                med_diff = before - after
                # low pass the median difference
                for frame in tqdm(range(med_diff.shape[0])):
                    med_diff[frame] = mp.lowpassfilter(med_diff[frame], 15)
                # get the square ratio between median and med diff
                ratio = np.sqrt(np.nansum(med ** 2) / np.nansum(med_diff ** 2))
                # scale the median difference
                med_diff *= ratio
            else:
                med = np.nanmedian(cube2, axis=0)
        # ---------------------------------------------------------------------
        # also keep track of the in vs out-of-transit 2D image.
        with warnings.catch_warnings(record=True) as _:
            med_out = np.nanmedian(cube2[oot_domain], axis=0)
            med_in = np.nanmedian(cube2[int_domain], axis=0)
        # get the diff in vs out
        transit_invsout = med_in - med_out
        # ---------------------------------------------------------------------
        # print progress
        misc.printc('\nCalculating residuals', 'info')
        # work out the residuals
        residuals = np.zeros_like(cube)
        # loop around frames
        for iframe in tqdm(range(nframes)):
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
            # write the before after clean 1f
            misc.printc('\tWriting: {0}'.format(tmp_before_after_c1f),
                        'info')
            fits.writeto(tmp_before_after_c1f, med_diff, overwrite=True)
        # ---------------------------------------------------------------------
        # return the cleaned cube, the median image, the median difference
        return_list = [cube, med, med_diff, transit_invsout, pcas]
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
        # note +1 as the last point of contact is deemed part of the transit
        valid_oot[cframes[0]:cframes[3] + 1] = False
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
        # note +1 as the last point of contact is deemd part of the transit
        valid_oot_after[:cframes[3] + 1] = False
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
        # print progress
        misc.printc('\tSubtracting 1/f noise', 'info')
        # get the degree for the 1/f polynomial fit
        degree_1f_corr = self.params['DEGREE_1F_CORR']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # deal with no poly fit of the 1/f noise
        if degree_1f_corr == 0:
            # get the median noise contribution
            with warnings.catch_warnings(record=True) as _:
                noise_1f = np.nanmedian(residuals, axis=1)
            # subtract this off the cube frame-by-frame
            for iframe in tqdm(range(nframes)):
                # we subtract the 1/f noise off each column
                for col in range(nbxpix):
                    cube[iframe, :, col] -= noise_1f[iframe, col]
        # otherwise we fit the 1/f noise
        else:
            # loop around every frame
            for iframe in tqdm(range(nframes)):
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
                    # noinspection PyBroadException
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
                med: np.ndarray, tracemap: np.ndarray
                ) -> Union[None, np.ndarray]:
        """
        Fit the PCA to the tracemap

        :param cube2: np.ndarray, the cube to fit the PCA to
        :param err: np.ndarray, the error cube
        :param med: np.ndarray, the median image
        :param tracemap: np.ndarray, the trace map

        :return: None if unable to do the PCA otherwise np.ndarray:
                 the principle components
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
        # print progress
        misc.printc('Fitting PCA', 'info')
        # ---------------------------------------------------------------------
        # only fit the pca to the flux in the trace (nan everything else)
        nanmask = np.ones_like(tracemap, dtype=float)
        nanmask[~tracemap] = np.nan
        # copy the normalized cube
        cube3ini = cube2[oot_domain]
        # subtract off the median
        for iframe in tqdm(range(cube3ini.shape[0])):
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
        plots.pca_plot(self, n_comp, pcas, variance_ratio)
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
        # print progress
        msg = 'Recentering trace position'
        misc.printc(msg, 'info')
        # ---------------------------------------------------------------------
        # get some parameters from instrument
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # ---------------------------------------------------------------------
        # print what we are doing
        msg = '\tScan to optimize position of trace'
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
        for ix in tqdm(range(len(dxs))):
            for iy in range(len(dys)):
                # update the x and y positions
                self.params['X_TRACE_OFFSET'] = dxs[ix]
                self.params['Y_TRACE_OFFSET'] = dys[iy]
                # re-gen the trace map (without logging) using new x/y trace
                #  offset
                tracemap = self.get_trace_map(log=False)
                sums[ix, iy] = np.nansum(tracemap * med)
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
        plots.gradient_plot(self, dx, dy, rotxy)
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
        # if we don't have a mask, we set dummy values for the plot
        # later in the code
        x_order0 = [np.nan]
        y_order0 = [np.nan]
        # deal with masking order zero
        if self.params['MASK_ORDER_ZERO']:
            # adding the masking of order 0
            mo0out = self.get_mask_order0(mask_trace_pos, tracemap)
            # get return from get_mask_order0
            mask_order0, x_order0, y_order0 = mo0out
            # set the values in the mask where order zero to 0
            mask_trace_pos[mask_order0] = 0
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
        _ = self, tracemap
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
                output_names.append(f'PCA{icomp + 1}')
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
        valid_arr = np.ones_like(cube).astype(float)
        # ---------------------------------------------------------------------
        # storage a trace correction array
        trace_corr = np.zeros(nframes, dtype=float)
        # loop around the frames
        for iframe in tqdm(range(cube.shape[0])):
            # find the best combination of scale/dx/dy/rotation
            # amps is a vector with the amplitude of all fitted terms
            # -----------------------------------------------------------------
            # set up a mask of valid pixels
            valid = np.isfinite(cube[iframe]).astype(float)
            valid[valid != 1] = np.nan
            valid[mask_trace_pos == 0] = np.nan
            # -----------------------------------------------------------------
            # calculate the sum of the trace
            with warnings.catch_warnings(record=True) as _:
                # work out the sum and error on the sum of the trace
                sum_trace = np.nansum(cube[iframe] * valid)
                err_sum_trace = np.sqrt(np.nansum(err[iframe] ** 2 * valid))
                # push into outputs
                outputs['sum_trace'][iframe] = sum_trace
                outputs['sum_trace_error'][iframe] = err_sum_trace
            # -----------------------------------------------------------------
            # calculate the amplitude when there is no model applied
            with warnings.catch_warnings(record=True) as _:
                # normalize cube by median
                tmp_slice = valid * (cube[iframe] / med)
                tmp_slice_err = valid * (err[iframe] / med)
                # get the odd ratio mean
                amp0 = mp.odd_ratio_mean(tmp_slice.ravel(),
                                         tmp_slice_err.ravel())
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
                recon -= amp_model[output_names.index('zeropoint')]
            # -----------------------------------------------------------------
            # calculate the trace correction
            part1 = np.nansum(recon * valid / err[iframe] ** 2)
            part2 = np.nansum(med * amp_model[0] * valid / err[iframe] ** 2)
            trace_corr[iframe] = part1 / part2
            # -----------------------------------------------------------------
            # plot the trace correction sample plot
            if iframe == 0:
                plots.trace_correction_sample(self, iframe,
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
        plots.aperture_correction_plot(self, outputs, trace_corr)
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
        :param valid_arr: np.ndarray, the valid pixels

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
        # Precompute finite mask for the entire cube
        finite_mask_cube = np.isfinite(cube)
        # ---------------------------------------------------------------------
        # loop around x pix
        for ix in tqdm(range(nbxpix)):
            # get the slice of the cube
            cube_slice = cube[:, :, ix]
            valid_slice = valid_arr[:, :, ix]
            finite_mask_slice = finite_mask_cube[:, :, ix]
            # Skip if no valid pixels
            if not np.any(finite_mask_slice):
                continue
            # loop around y pix
            for iy in range(nbypix):
                # Skip if slice  if no pixels are finite
                if np.sum(np.isfinite(valid_slice[:, iy])) == 0:
                    continue
                # get the sample column
                sample = cube_slice[:, iy]
                # get the out of transit domain in the sample
                sample_oot = sample[oot_domain]
                # get the indices of the out of transit domain
                frames_oot = frames[oot_domain]
                # find any nans in the oot sample
                finite_mask = finite_mask_slice[:, iy][oot_domain]
                # -------------------------------------------------------------
                # only fit the polynomial if we have enough points
                if np.sum(finite_mask) <= poly_order:
                    continue
                # -------------------------------------------------------------
                # deal with having NaNs in the sample
                if False in finite_mask:
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
        _ = self
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
            vector2 = -roll1 / 3 + roll2 + roll3 / 3
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
        medfile = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        # load the median from disk
        med = io.load_fits(medfile)
        # get the tracemap
        tracemap = self.get_trace_map()
        # get the wave grid
        wavegrid = self.get_wavegrid(order_num=1)
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
        # Add to variables
        self.set_variable('PHOTO_WEIGHTED_MEAN', mean_photon_weighted)
        self.set_variable('ENERGY_WEIGHTED_MEAN', mean_energy_weighted)
        # ----------------------------------------------------------------
        # return the effective wavelength factors
        return mean_photon_weighted, mean_energy_weighted

    def save_wlc_results(self, cube: np.ndarray, err: np.ndarray, 
                         lrecon: np.ndarray, ltable: Table):
        # set function name
        func_name = f'{__NAME__}.{self.name}.save_wlc_results()'
        # ---------------------------------------------------------------------
        # update the meta data
        self.update_meta_data()
        # get the meta data
        meta_data = self.get_variable('META', func_name)
        # -------------------------------------------------------------------------
        # write the error map
        errfile = self.get_variable('WLC_ERR_FILE', func_name)
        io.save_fits(errfile, datalist=[err], datatypes=['image'],
                     datanames=['err'], meta=meta_data)
        # -------------------------------------------------------------------------
        # write the residual map
        resfile = self.get_variable('WLC_RES_FILE', func_name)
        io.save_fits(resfile, datalist=[cube], datatypes=['image'],
                     datanames=['residual'], meta=meta_data)
        # -------------------------------------------------------------------------
        # write the recon
        reconfile = self.get_variable('WLC_RECON_FILE', func_name)
        io.save_fits(reconfile, datalist=[lrecon], datatypes=['image'],
                     datanames=['recon'], meta=meta_data)
        # -------------------------------------------------------------------------
        # write the table to the csv path
        ltbl_file = self.get_variable('WLC_LTBL_FILE', func_name)
        io.save_table(ltbl_file, ltable, fmt='csv')

    # ==========================================================================
    # Spectral Extraction functionality
    # ==========================================================================
    def create_sed(self, med: np.ndarray, residual: np.ndarray,
                   wavegrid: np.ndarray, posmax: np.ndarray,
                   throughput: np.ndarray, med_clean: np.ndarray,
                   trace_order: int) -> np.ndarray:
        # set the function name
        func_name = f'{__NAME__}.{self.name}.create_sed()'
        # for future reference in the code, we keep track of data size
        self.set_variable('DATA_X_SIZE', residual.shape[2], func_name)
        self.set_variable('DATA_Y_SIZE', residual.shape[1], func_name)
        self.set_variable('DATA_N_FRAMES', residual.shape[0], func_name)
        # ---------------------------------------------------------------------
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
        plots.plot_sed(self, wavegrid, sp_sed / throughput, trace_order)
        # ---------------------------------------------------------------------
        # return this table
        return sp_sed

    def load_model(self, recon: np.ndarray, med: np.ndarray) -> np.ndarray:
        """
        load the model (and deal with masking order zero if required)

        :return:
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.create_model()'
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # load the tracemap
        tracemap = self.get_trace_map()
        # ---------------------------------------------------------------------
        # the model starts as the recon
        model = np.array(recon)
        # deal with masking order zero
        if self.params['MASK_ORDER_ZERO']:
            # load the mask trace position
            mask_trace_pos, _, _, _, _ = self.get_mask_trace_pos(med, tracemap)
            # need to re-get the mask order zero
            mask_order0, xpos, ypos = self.get_mask_order0(mask_trace_pos,
                                                           tracemap)
            # loop around frames and mask out order zero (with NaNs)
            for iframe in tqdm(range(nbframes)):
                # set the order zero values to nan
                model[iframe][mask_order0] = np.nan
        # ---------------------------------------------------------------------
        # return the model
        return model

    def ratio_residual_to_trace(self, model: np.ndarray, err: np.ndarray,
                                residual: np.ndarray, posmax: np.ndarray
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

        :param model: np.ndarray, the model
        :param err: np.ndarray, the error cube
        :param residual: np.ndarray, the residual cube
        :param posmax: np.ndarray, the position of the trace

        :return: tuple, 1. the spectrum, 2. the spectrum error
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.ratio_residual_to_trace()'
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # ---------------------------------------------------------------------
        # placeholder for the cube spectra
        spec = np.full([nbframes, nbxpix], np.nan)
        spec_err = np.full([nbframes, nbxpix], np.nan)
        # loop through observations and spectral bins
        for iframe in tqdm(range(nbframes)):
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
                    # noinspection PyBroadException
                    try:
                        ratio, err_ratio = mp.odd_ratio_mean(v1 / v0, v2 / v0)
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

    def remove_trend(self, spec: np.ndarray, ltable: Table
                     ) -> Tuple[np.ndarray, Table]:
        """
        Remove the out-of-transit trend from the spectrum and the linear fit
        table

        :param spec: np.ndarray, the spectrum
        :param ltable: Table, the linear fit table
        :return:
        """
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
            # return the spec and ltable without removing trend
            return spec, ltable
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
        # do the same for the photometric time series
        # ---------------------------------------------------------------------
        # to the same as we did on the spectrum on the photometric time series
        v1 = np.array(ltable['amplitude'])
        # get an index array for v1
        index = np.arange(spec.shape[0])
        # fit the out-of-transit trend
        tfit = np.polyfit(index[oot_domain], v1[oot_domain], 1)
        # remove this off the ampliduteds
        ltable['amplitude'] -= np.polyval(tfit, index)
        # ---------------------------------------------------------------------
        # return the updated spec and ltable
        return spec, ltable

    def get_transit_depth(self, ltable: Table) -> Union[float, None]:
        """
        Get the transit depth (either user defined or calculate)

        :param ltable: Table, the linear fit table (from WLC)

        :raises SossisseConstantException: if TDEPTH_MODE is set to compute
                                             and TDEPTH is not set or TDEPTH
                                             is not a valid float
        :return: None if out-of-transit domain not set, otherwise the transit
                 depth
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_transit_depth()'
        # deal with the case where we are not in "compute" mode
        if self.params['TDEPTH_MODE'] != 'compute':
            # user must set the transit depth if this is the case
            if self.params['TDEPTH'] is None:
                emsg = 'TDEPTH_MODE is not set to compute, please set TDEPTH'
                raise exceptions.SossisseConstantException(emsg)
            # return the transit depth defined by user
            else:
                try:
                    return float(self.params['TDEPTH'])
                except Exception as e:
                    emsg = 'TDEPTH value is not valid\n\t{0}:{1}'
                    emsg = emsg.format(type(e), e)
                    raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # otherwise we are in compute mode
        # ---------------------------------------------------------------------
        # get the out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        int_domain = self.get_variable('INT_DOMAIN', func_name)
        # deal with out of transit domain not set
        if not has_oot:
            wmsg = ('Cannot calculate transit depth trend without '
                    'out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to remove_trend.')
            misc.printc(wmsg, 'warning')
            # return the spec and ltable without removing trend
            return None
        # ---------------------------------------------------------------------
        # get the transit depth
        with warnings.catch_warnings(record=True) as _:
            part1 = np.nanmedian(ltable['amplitude'][oot_domain])
            part2 = np.nanmean(ltable['amplitude'][int_domain])
            # transit depth is the median out-of-transit amplitudes
            # minus the mean of the in transit amplitudes
            transit_depth = part1 - part2
        # ---------------------------------------------------------------------
        # return the transit depth
        return transit_depth

    IntransitSpectrum = Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                              Tuple[None, None, None]]

    def intransit_spectrum(self, spec: np.ndarray, spec_err: np.ndarray
                           ) -> IntransitSpectrum:
        """
        Construct the in-transit spectrum

        :param spec: np.ndarray, the spectrum
        :param spec_err: np.ndarray, the spectrum error

        :return: tuple, 1. the in-transit spectrum, 2. the in-transit spectrum
                    error, 3. the out-of-transit spectrum error,
                    if out-of-transit if not defined returns None, None, None
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.intransit_spectrum()'
        # get the out-of-transit domain
        self.get_valid_oot()
        has_oot = self.get_variable('HAS_OOT', func_name)
        oot_domain = self.get_variable('OOT_DOMAIN', func_name)
        int_domain = self.get_variable('INT_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # deal with no out-of-transit domain defined
        if not has_oot:
            wmsg = ('Cannot get in-transit spectrum without '
                    'out-of-transit domain.'
                    '\n\tPlease set CONTACT_FRAMES to remove_trend.')
            misc.printc(wmsg, 'warning')
            # return the spec and ltable without removing trend
            return None, None, None
        # ---------------------------------------------------------------------
        # weights of each point from uncertainties
        weight = 1 / spec_err ** 2
        # in transit spectrum and error
        with warnings.catch_warnings(record=True) as _:
            # calculate the weighted sum of the spectrum - in transit
            sumspec_in = np.nansum(spec[int_domain] * weight[int_domain],
                                   axis=0)
            # calculate the sum of the weights - in transit
            sumweight_in = np.nansum(weight[int_domain], axis=0)
            # calculate the in transit spectrum
            spec_in = sumspec_in / sumweight_in
            # calculate the in transit spectrum error
            spec_err_in = np.sqrt(1 / sumweight_in ** 2)

        with warnings.catch_warnings(record=True) as _:
            # calculate the sum of the weights - out of transit
            sumweight_out = np.nansum(weight[oot_domain], axis=0)
            # calculate the out-of-transit spectrum error
            spec_err_out = 1 / np.sqrt(sumweight_out ** 2)
        # ---------------------------------------------------------------------
        # if we have removed a trend, we need to add in quadrature
        #  out-of-transit  errors to in-transit
        if self.params['REMOVE_TREND']:
            spec_err_in = np.sqrt(spec_err_in ** 2 + spec_err_out ** 2)
        # ---------------------------------------------------------------------
        # chane infinite values to nan
        spec_err_in[~np.isfinite(spec_err_in)] = np.nan
        spec_err_out[~np.isfinite(spec_err_out)] = np.nan
        # ---------------------------------------------------------------------
        return spec_in, spec_err_in, spec_err_out

    def bin_spectrum(self, wavegrid: np.ndarray, spec_in: np.ndarray,
                     spec_err_in: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin the spectrum
        
        :param wavegrid: np.ndarray, the wavelength grid
        :param spec_in: np.ndarray, the in-transit
        :param spec_err_in: np.ndarray, the in-transit error
        
        """
        # get the wavelength bins
        with warnings.catch_warnings(record=True) as _:
            # log the wavelength
            logwave = np.log(wavegrid / np.nanmin(wavegrid))
            # get the wavelength binning
            wbin = np.floor(logwave) * self.params['RESOLUTION_BIN']
        # create a wavebin, fluxbin and corresponding error vectors for output
        wave_bin = np.array(list(set(wbin)))
        flux_bin = np.zeros_like(wave_bin)
        err_bin = np.zeros_like(wave_bin)
        # loop around bins in wavelength
        for ibin in range(len(wave_bin)):
            # find all wavelengths in this bin
            valid = wbin == wave_bin[ibin]
            # get the flux and error using odd_ratio_mean
            flux, error = mp.odd_ratio_mean(spec_in[valid], spec_err_in[valid])
            # push into array
            flux_bin[ibin] = flux
            err_bin[ibin] = error
        # return the binned data
        return wave_bin, flux_bin, err_bin

    def save_spe_results(self, storage: Dict[str, Any], trace_order: int):
        """
        Save the results to disk
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.save_results()'
        # deal with not saving results --> return
        if not self.params['SAVE_RESULTS']:
            # print message saving results are not saved 
            msg = 'Results not saved as SAVE_RESUILTS is False'
            misc.printc(msg, 'info')
            return

        # get the meta data
        meta_data = self.get_variable('META', func_name)

        # get variables from storage
        wavegrid = storage['wavegrid']
        sp_sed = storage['sp_sed']
        throughput = storage['throughput']
        spec = storage['spec']
        spec_err = storage['spec_err']
        amp_image = storage['amp_image']
        wavegrid_2d = storage['wavegrid_2d']
        spec_in = storage['spec_in']
        spec_err_in = storage['spec_err_in']
        transit_depth = storage['transit_depth']
        wave_bin = storage['wave_bin']
        flux_bin = storage['flux_bin']
        flux_bin_err = storage['flux_bin_err']
        # ---------------------------------------------------------------------
        # Save the SED table
        # ---------------------------------------------------------------------
        # construct sed table name
        sed_table_name = self.get_variable('SPE_SED_TBL', func_name)
        sed_table_name = sed_table_name.format(objname=self.params['OBJECTNAME'],
                                               trace_order=trace_order)
        # construct SED table
        sed_table = Table()
        sed_table['wavelength'] = wavegrid
        sed_table['flux'] = sp_sed / throughput
        sed_table['raw flux'] = sp_sed
        sed_table['throughput'] = throughput
        # save table to disk
        io.save_table(sed_table_name, sed_table)
        # ---------------------------------------------------------------------
        # Save the residual no grey order file
        # ---------------------------------------------------------------------
        # get file name
        res_no_grey_ord = self.get_variable('RES_NO_GREY_ORD', func_name)
        res_no_grey_ord = res_no_grey_ord.format(trace_order=trace_order)
        # write the residual no grey order tile
        io.save_fits(res_no_grey_ord, datalist=[spec, spec_err],
                     datatypes=['image', 'image'],
                     datanames=['spec', 'spec_err'],
                     meta=meta_data)
        # ---------------------------------------------------------------------
        # Save the residual grey order file (from amplitudes
        # ---------------------------------------------------------------------
        # get file name
        res_grey_ord = self.get_variable('RES_GREY_ORD', func_name)
        res_grey_ord = res_grey_ord.format(trace_order=trace_order)
        # write the residual grey order tile
        io.save_fits(res_grey_ord, datalist=[amp_image, spec_err],
                     datatypes=['image', 'image'],
                     datanames=['spec', 'spec_err'],
                     meta=meta_data)
        # ---------------------------------------------------------------------
        # Save the spectra for this trace order to file
        # ---------------------------------------------------------------------
        # get file name
        spec_file = self.get_variable('SPECTRA_ORD', func_name)
        spec_file = spec_file.format(trace_order=trace_order)
        # get the data list
        datalist = [wavegrid_2d, amp_image, spec_err]
        # get the data type list
        datatypes = ['image', 'image', 'image']
        # get the data names
        datanames = ['WAVELENGTH', 'RELFLUX', 'RELFLUX_ERROR']
        # save the fits file
        io.save_fits(spec_file, datalist=datalist, datatypes=datatypes,
                     datanames=datanames, meta=meta_data)

        # ---------------------------------------------------------------------
        # Save the wavelength for this trace order to file
        # ---------------------------------------------------------------------
        # get file name
        wave_file = self.get_variable('WAVE_ORD', func_name)
        wave_file = wave_file.format(trace_order=trace_order)
        # write the wavelength file
        io.save_fits(wave_file, datalist=[wavegrid], datatypes=['image'],
                     datanames=['WAVELENGTH'], meta=meta_data)

        # ---------------------------------------------------------------------
        # Save the transit spectrum for this trace order to file
        # ---------------------------------------------------------------------
        # get file name
        transit_file = self.get_variable('TSPEC_ORD', func_name)
        transit_file = transit_file.format(trace_order=trace_order)

        # make transit table
        t_table = Table()
        t_table['wavelength'] = wavegrid
        t_table['flux'] = (spec_in + transit_depth) * 1e6
        t_table['flux_err'] = spec_err_in * 1e6
        # save table
        io.save_table(transit_file, t_table, fmt='csv')

        # ---------------------------------------------------------------------
        # Save the transit spectrum for this trace order (binned) to file
        # ---------------------------------------------------------------------
        # get the resolution
        res = self.params['RESOLUTION_BIN']
        # get file name
        transit_bin_file = self.get_variable('TSPEC_ORD_BIN', func_name)
        transit_bin_file = transit_bin_file.format(trace_order=trace_order,
                                                   res=res)
        # make transit table
        tb_table = Table()
        tb_table['wavelength'] = wave_bin
        tb_table['flux'] = (flux_bin + transit_depth) * 1e6
        tb_table['flux_err'] = flux_bin_err * 1e6
        # save table
        io.save_table(transit_bin_file, tb_table, fmt='csv')

    def to_eureka(self, storage: Dict[int, Dict[str, Any]]):
        # set function name
        func_name = f'{__NAME__}.{self.name}.to_eureka()'
        # get the trace orders from the
        trace_orders = list(storage.keys())
        # print progress
        msg = 'Converting to Eureka format'
        misc.printc(msg, 'info')
        # loop around trace orders
        for trace_order in trace_orders:
            # ----------------------------------------------------------------
            # print progress
            msg = f'Processing order {trace_order}'
            misc.printc(msg, 'info')
            # get variables from storage
            flux = np.array(storage[trace_order]['amp_image'])
            flux_err = np.array(storage[trace_order]['spec_err'])
            wavegrid = np.array(storage[trace_order]['wavegrid'])
            # ----------------------------------------------------------------
            # print progress
            msg = 'Sorting in decreasing wavelength order'
            misc.printc(msg, 'info')
            # sort by decreasing wavelength order
            sortmask = np.argsort(wavegrid)[::-1]
            # apply sort mask
            wavegrid = wavegrid[sortmask]
            flux = flux[:, sortmask]
            flux_err = flux_err[:, sortmask]
            # -----------------------------------------------------------------
            # print progress
            msg = 'Reading raw file(s) to get time array'
            misc.printc(msg, 'info')
            # storage for time info
            time_arr = []
            # loop around raw files
            for filename in self.params['FILES']:
                # get time array data
                tmp_table = io.load_table(filename, hdu='INT_TIMES')
                # get the int_mid_bjd_tdb column
                time_arr.append(np.array(tmp_table['int_mid_BJD_TDB']))
            # convert time_arr into a numpy array
            time_arr = np.concatenate(time_arr)
            # -----------------------------------------------------------------
            # get the filename
            filename = self.get_variable('EUREKA_FILE', func_name)
            filename = filename.format(trace_order=trace_order)
            # -----------------------------------------------------------------
            # save eureka format file
            io.save_eureka(filename, flux, flux_err, wavegrid, time_arr)


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
