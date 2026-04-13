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
import sys
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numexpr as ne
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

from aperocore import math as mp
from aperocore.constants import param_functions
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.core import misc
from sossisse.general import plots

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.default'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# Get parmaeter dictionary
ParamDict = param_functions.ParamDict
# cache for param table
PARAM_TABLE = dict()
# Customize LaTeX output for tex out table
LATEX_DICT = {
    # outer environment
    'tabletype': 'table',
    # column format with vertical lines
    'col_align': '|l|r|',
    # before header row
    'header_start': r'\hline',
    # after header row
    'header_end': r'\hline',
    # after all data
    'data_end': r'\hline',
    # caption text
    'caption': f'SOSSISSE ({__version__}) parameters used for this run',
}


# =============================================================================
# Define instrument class
# =============================================================================
class Instrument:
    """
    Base instrument class - must have all methods given in
    any child class
    """

    def __init__(self, params: ParamDict):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        self.name = 'Default'
        # set the parameters from input
        self.params = copy.deepcopy(params)
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
        self._variables['TEMP_INI_DQ'] = None
        self._variables['TEMP_FF_CUBE'] = None
        self._variables['TEMP_FF_ERR'] = None
        self._variables['TEMP_FF_DQ'] = None
        self._variables['TEMP_CUBE_POST_COSMICS'] = None
        self._variables['TEMP_INI_CUBE_BKGRND'] = None
        self._variables['TEMP_INI_ERR_BKGRND'] = None
        self._variables['TEMP_INI_CUBE_LOWPASS'] = None
        self._variables['TEMP_INI_ERR_LOWPASS'] = None
        self._variables['TEMP_CLEAN_NAN'] = None
        self._variables['TEMP_CLEAN_NAN_ERR'] = None
        self._variables['BADPIXEL_CUTOUT_DIR'] = None
        self._variables['MEDIAN_IMAGE_FILE'] = None
        self._variables['TEMP_AMP_FILE'] = None
        self._variables['CLEAN_CUBE_FILE'] = None
        self._variables['TEMP_PCA_FILE'] = None
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
        self._variables['HAS_BASELINE'] = None
        # meta data
        self._variables['META'] = None
        self._variables['OUTPUT_NAMES'] = None
        self._variables['OUTPUT_UNITS'] = None
        self._variables['OUTPUT_FACTOR'] = None
        # simple vectors
        self._variables['BASELINE_INTS'] = None
        self._variables['BASELINE_DOMAIN'] = None
        self._variables['PHOTO_WEIGHTED_MEAN'] = None
        self._variables['ENERGY_WEIGHTED_MEAN'] = None
        # final output files
        self._variables['OUT_SPEC_LC_FILE'] = None
        self._variables['OUT_WLC_FILE'] = None
        self._variables['OUT_TEX_FILE'] = None
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
        self.vsources['TEMP_INI_DQ'] = define_func
        self.vsources['TEMP_FF_CUBE'] = define_func
        self.vsources['TEMP_FF_ERR'] = define_func
        self.vsources['TEMP_FF_DQ'] = define_func
        self.vsources['TEMP_CUBE_POST_COSMICS'] = define_func
        self.vsources['TEMP_INI_CUBE_BKGRND'] = define_func
        self.vsources['TEMP_INI_ERR_BKGRND'] = define_func
        self.vsources['TEMP_INI_CUBE_LOWPASS'] = define_func
        self.vsources['TEMP_INI_ERR_LOWPASS'] = define_func
        self.vsources['TEMP_CLEAN_NAN'] = define_func
        self.vsources['TEMP_CLEAN_NAN_ERR'] = define_func
        self.vsources['BADPIXEL_CUTOUT_DIR'] = define_func
        self.vsources['MEDIAN_IMAGE_FILE'] = define_func
        self.vsources['TEMP_AMP_FILE'] = define_func
        self.vsources['CLEAN_CUBE_FILE'] = define_func
        self.vsources['TEMP_PCA_FILE'] = define_func
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
        self.vsources['HAS_BASELINE'] = f'{self.name}.get_baseline_params()'
        # meta data
        self.vsources['META'] = f'{self.name}.update_meta_data()'
        self.vsources['OUTPUT_NAMES'] = f'{self.name}.setup_linear_reconstruction()'
        self.vsources['OUTPUT_UNITS'] = f'{self.name}.setup_linear_reconstruction()'
        self.vsources['OUTPUT_FACTOR'] = f'{self.name}.setup_linear_reconstruction()'
        # simple vectors
        self.vsources['BASELINE_INTS'] = f'{self.name}.get_baseline_params()'
        self.vsources['BASELINE_DOMAIN'] = f'{self.name}.get_baseline_params()'
        self.vsources['PHOTO_WEIGHTED_MEAN'] = f'{self.name}.get_effective_wavelength()'
        self.vsources['ENERGY_WEIGHTED_MEAN'] = f'{self.name}.get_effective_wavelength()'
        # final output files
        self.vsources['OUT_SPEC_LC_FILE'] = define_func
        self.vsources['OUT_WLC_FILE'] = define_func
        self.vsources['OUT_TEX_FILE'] = define_func

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
                    'Please run {1} before running {2}.')
            eargs = [key, self.vsources[key], func_name]
            raise exceptions.SossisseInstException(emsg.format(*eargs),
                                                   self.name)
        # otherwise return the variable
        return self._variables[key]

    def update_meta_data(self):
        """
        Update the meta data dictionary

        :return:
        """
        # set function name
        # func_name = f'{__NAME__}.update_meta_data()'
        # set up storage
        meta_data = dict()
        # get wlc general properties
        wlc_gen_params = self.params.get('WLC.GENERAL')
        # get cds_ids
        cds_ids = self.params['WLC.INPUTS.CDS_IDS']
        # ---------------------------------------------------------------------
        # deal with CDS data
        # ---------------------------------------------------------------------
        if self._variables['FLAG_CDS'] is not None:
            # only if true
            if self._variables['FLAG_CDS']:
                # deal with FLAG_CDS = True but CDS_IDS None
                if cds_ids is None:
                    emsg = ('WLC.INPUTS.CDS_IDS not set but CDS flag is '
                            'set to True. If input data is not a CDS please '
                            'set WLC.INPUTS.CDS_IDS.')
                    raise exceptions.SossisseConstantException(emsg)
                # add to meta data
                meta_data['CDS'] = (True, 'Whether data is calculated from '
                                          'a CDS')
                meta_data['CDS_FIRST'] = (cds_ids[0], 'First frame of CDS')
                meta_data['CDS_LAST'] = (cds_ids[1], 'Last frame of CDS')
        # ---------------------------------------------------------------------
        # deal with wlc domain
        # ---------------------------------------------------------------------
        wlc_domain = self.params['GENERAL.WLC_DOMAIN']

        if wlc_domain is not None:
            # add to meta data
            meta_data['WLC_DOM0'] = (wlc_domain[0],
                                     'Wavelength domain start [um]')
            meta_data['WLC_DOM1'] = (wlc_domain[1],
                                     'Wavelength domain end [um]')
        # ---------------------------------------------------------------------
        # deal with median out of transit
        # ---------------------------------------------------------------------
        # add to meta data
        meta_data['BASELINE_MED'] = (wlc_gen_params['MEDIAN_BASELINE'],
                                     'Median out of transit')
        # get the linear model params
        lm_params = self.params.get('WLC.LMODEL')
        # ---------------------------------------------------------------------
        # deal with background correction
        # ---------------------------------------------------------------------
        # deal with DO_BACKGROUND set
        if self.params['GENERAL.DO_BACKGROUND']:
            # get variable
            do_background = self.params['GENERAL.DO_BACKGROUND']
            # add to meta data
            meta_data['DO_BKG'] = (int(do_background),
                                   'Background was removed')
        # ---------------------------------------------------------------------
        # deal with degree of the polynomial for the 1/f correction
        # ---------------------------------------------------------------------
        # add to meta data
        meta_data['DEG1FCORR'] = (wlc_gen_params['DEGREE_1F_CORR'],
                                  'Degree of polynomial for 1/f correction')
        # ---------------------------------------------------------------------
        # deal with whether we fit the rotation in the linear model
        # ---------------------------------------------------------------------
        # add to meta data
        meta_data['FITROT'] = (lm_params['FIT_ROTATION'],
                               'Rotation was fitted in linear model')
        # ---------------------------------------------------------------------
        # deal with whether we fit the zero point offset in the linear model
        # ---------------------------------------------------------------------
        # add to meta data
        meta_data['FITZP'] = (lm_params['FIT_ZERO_POINT_OFFSET'],
                              'Zero point offset was fitted in linear model')
        # ---------------------------------------------------------------------
        # deal with whether we fit the 2nd derivative in y
        # ---------------------------------------------------------------------
        # add to meta data
        meta_data['FITDDY'] = (lm_params['FIT_DDY'],
                               '2nd derivative was fitted in y')
        # ---------------------------------------------------------------------
        # add the baseline integrations
        # ---------------------------------------------------------------------
        # get baseline integrations
        baseline_ints = self._variables['BASELINE_INTS']
        # add to meta data
        if baseline_ints is None:
            # set the number of baselines
            meta_data[f'BL_INTS'] = (0, 'Number of baseline integration groups')
        elif isinstance(baseline_ints, list):
            # set the number of baselines
            meta_data[f'BL_INTS'] = (len(baseline_ints), 
                                     'Number of baseline integration groups')
            # loop around baselines and add them to meta data
            for it in range(len(baseline_ints)):
                for jt in range(2):
                    # assume we don't have more than 1000 baselines
                    itstr = f'{str(it):03s}'
                    # add to meta data
                    if jt == 0:
                        meta_data[f'BL{itstr}_0'] = (baseline_ints[it][jt],
                                                     f'Baseline integration '
                                                     f'[{it}] start')
                    else:
                        meta_data[f'BL{itstr}_{jt}'] = (baseline_ints[it][jt],
                                                      f'Baseline integration '
                                                      f'[{it}] end')
        else:
            emsg = 'BASELINE_INTS must be a list of lists'
            raise exceptions.SossisseConstantException(emsg)

        # ---------------------------------------------------------------------
        # deal with removing trend from out-of-transit data
        # ---------------------------------------------------------------------
        # get the spec extraction params
        spec_ext_params = self.params.get('SPEC_EXT')
        # add to meta data
        meta_data['RMVTREND'] = (spec_ext_params['REMOVE_TREND'],
                                 'Trend was removed from out-of-transit')
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
        # push meta data to variables
        self.set_variable('META', meta_data)

    def define_filenames(self):
        # set function name
        func_name = f'{__NAME__}.define_filenames()'
        # update meta data
        self.update_meta_data()
        # get file paths
        temppath = self.params['PATHS.TEMP_PATH']
        otherpath = self.params['PATHS.OTHER_PATH']
        fitspath = self.params['PATHS.FITS_PATH']
        outpath = self.params['PATHS.OUT_PATH']
        # ---------------------------------------------------------------------
        # Whatever happens raw input files must exist
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # construct temporary file names
        # ---------------------------------------------------------------------
        # amplitude file
        tmp_amp_file = 'temporary_amp.fits'
        tmp_amp_file = os.path.join(temppath, tmp_amp_file)
        # ---------------------------------------------------------------------
        clean_cube_file = 'cleaned_cube.fits'
        clean_cube_file = os.path.join(temppath, clean_cube_file)
        # ---------------------------------------------------------------------
        tmp_pcas = 'temporary_pcas.fits'
        tmp_pcas = os.path.join(temppath, tmp_pcas)
        # ---------------------------------------------------------------------
        temp_clean_nan = 'temporary_cleaned_isolated.fits'
        temp_clean_nan = os.path.join(temppath, temp_clean_nan)
        # ---------------------------------------------------------------------
        temp_clean_nan_err = 'temporary_cleaned_isolated_err.fits'
        temp_clean_nan_err = os.path.join(temppath, temp_clean_nan_err)
        # ---------------------------------------------------------------------
        badpix_cutout_dir = os.path.join(temppath, 'badpix_cutouts')
        if not os.path.exists(badpix_cutout_dir):
            os.makedirs(badpix_cutout_dir)
        # ---------------------------------------------------------------------
        temp_ini_cube = 'temporary_initial_cube.fits'
        temp_ini_cube = os.path.join(temppath, temp_ini_cube)
        # ---------------------------------------------------------------------
        temp_ini_err = 'temporary_initial_err.fits'
        temp_ini_err = os.path.join(temppath, temp_ini_err)
        # ---------------------------------------------------------------------
        temp_ini_dq = 'temporary_initial_dq.fits'
        temp_ini_dq = os.path.join(temppath, temp_ini_dq)
        # ---------------------------------------------------------------------
        temp_ff_cube = 'temporary_ff_cube.fits'
        temp_ff_cube = os.path.join(temppath, temp_ff_cube)
        # ---------------------------------------------------------------------
        temp_ff_err = 'temporary_ff_err.fits'
        temp_ff_err = os.path.join(temppath, temp_ff_err)
        # ---------------------------------------------------------------------
        temp_ff_dq = 'temporary_ff_dq.fits'
        temp_ff_dq = os.path.join(temppath, temp_ff_dq)
        # ---------------------------------------------------------------------
        temp_cube_post_cosmics = 'temporary_cube_post_cosmics.fits'
        temp_cube_post_cosmics = os.path.join(temppath, temp_cube_post_cosmics)
        # ---------------------------------------------------------------------
        tmp_ini_cube_bkgrnd = 'temporary_initial_cube_bkgrnd1.fits'
        tmp_ini_cube_bkgrnd = os.path.join(temppath, tmp_ini_cube_bkgrnd)
        # ---------------------------------------------------------------------
        tmp_ini_err_bkgrnd = 'temporary_initial_err_bkgrnd1.fits'
        tmp_ini_err_bkgrnd = os.path.join(temppath, tmp_ini_err_bkgrnd)
        # ---------------------------------------------------------------------
        tmp_ini_cube_lowpass = 'temporary_initial_cube_lowpass.fits'
        tmp_ini_cube_lowpass = os.path.join(temppath, tmp_ini_cube_lowpass)
        # ---------------------------------------------------------------------
        tmp_ini_err_lowpass = 'temporary_initial_err_lowpass.fits'
        tmp_ini_err_lowpass = os.path.join(temppath, tmp_ini_err_lowpass)
        # ---------------------------------------------------------------------
        median_image_file = 'median.fits'
        median_image_file = os.path.join(fitspath, median_image_file)
        # ---------------------------------------------------------------------
        errfile = os.path.join(temppath, 'errormap.fits')
        # ---------------------------------------------------------------------
        resfile = os.path.join(temppath, 'residual.fits')
        # ---------------------------------------------------------------------
        reconfile = os.path.join(temppath, 'recon.fits')
        # ---------------------------------------------------------------------
        ltbl_file = os.path.join(otherpath, 'linear_recon_coeffs.csv')
        # ---------------------------------------------------------------------
        sed_table = os.path.join(otherpath, 'sed_{objname}_ord{trace_order}.csv')
        # ---------------------------------------------------------------------
        res_no_grey_ord = 'residual_no_grey_ord{trace_order}.fits'
        res_no_grey_ord = os.path.join(fitspath, res_no_grey_ord)
        # ---------------------------------------------------------------------
        res_grey_ord = 'residual_grey_ord{trace_order}.fits'
        res_grey_ord = os.path.join(fitspath, res_grey_ord)
        # ---------------------------------------------------------------------
        spectra_ord = 'spectra_ord{trace_order}.fits'
        spectra_ord = os.path.join(fitspath, spectra_ord)
        # ---------------------------------------------------------------------
        waveord_file = 'wavelength_ord{trace_order}.fits'
        waveord_file = os.path.join(fitspath, waveord_file)
        # ---------------------------------------------------------------------
        tspec_ord = 'tspec_ord{trace_order}.csv'
        tspec_ord = os.path.join(otherpath, tspec_ord)
        # ---------------------------------------------------------------------
        tspec_ord_bin = 'tspec_ord_{trace_order}_bin{res}.csv'
        tspec_ord_bin = os.path.join(otherpath, tspec_ord_bin)
        # ---------------------------------------------------------------------
        eureka_file = 'spectra_ord{trace_order}.h5'
        eureka_file = os.path.join(fitspath, eureka_file)
        # ---------------------------------------------------------------------
        out_spec_lc_file = '{prefix}_slc.fits'
        out_spec_lc_file = os.path.join(outpath, out_spec_lc_file)
        # ---------------------------------------------------------------------
        out_wlc_file = '{prefix}_wlc.fits'
        out_wlc_file = os.path.join(outpath, out_wlc_file)
        # ---------------------------------------------------------------------
        out_tex_file = '{prefix}_sossisse.tex'
        out_tex_file = os.path.join(outpath, out_tex_file)
        # ---------------------------------------------------------------------
        # temp files
        self.set_variable('MEDIAN_IMAGE_FILE', median_image_file)
        self.set_variable('TEMP_AMP_FILE', tmp_amp_file)
        self.set_variable('CLEAN_CUBE_FILE', clean_cube_file)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        self.set_variable('TEMP_CLEAN_NAN', temp_clean_nan)
        self.set_variable('TEMP_CLEAN_NAN_ERR', temp_clean_nan_err)
        self.set_variable('BADPIXEL_CUTOUT_DIR', badpix_cutout_dir)
        self.set_variable('TEMP_INI_CUBE', temp_ini_cube)
        self.set_variable('TEMP_INI_ERR', temp_ini_err)
        self.set_variable('TEMP_INI_DQ', temp_ini_dq)
        self.set_variable('TEMP_FF_CUBE', temp_ff_cube)
        self.set_variable('TEMP_FF_ERR', temp_ff_err)
        self.set_variable('TEMP_FF_DQ', temp_ff_dq)
        self.set_variable('TEMP_CUBE_POST_COSMICS', temp_cube_post_cosmics)
        self.set_variable('TEMP_INI_CUBE_BKGRND', tmp_ini_cube_bkgrnd)
        self.set_variable('TEMP_INI_ERR_BKGRND', tmp_ini_err_bkgrnd)
        self.set_variable('TEMP_INI_CUBE_LOWPASS', tmp_ini_cube_lowpass)
        self.set_variable('TEMP_INI_ERR_LOWPASS', tmp_ini_err_lowpass)
        # ---------------------------------------------------------------------
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
        # final output files
        self.set_variable('OUT_SPEC_LC_FILE', out_spec_lc_file)
        self.set_variable('OUT_WLC_FILE', out_wlc_file)
        self.set_variable('OUT_TEX_FILE', out_tex_file)

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

    def load_cube(self, n_slices: int, image_shape: List[int],
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
        for ifile, filename in enumerate(self.params['GENERAL.FILES']):
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
        # get wlc inputs
        wlc_inputs = self.params['WLC.INPUTS']
        # get definitions of first and last from params['CDS_IDS']
        first = wlc_inputs['CDS_IDS'][0]
        last = wlc_inputs['CDS_IDS'][1]
        # deal with wrong order
        if first > last:
            first, last = last, first
        # deal with first and last being the same
        if first == last:
            emsg = 'WLC.INPUTS.CDS_IDS: First and last frame cannot be the same'
            raise exceptions.SossisseConstantException(emsg)
        # get the CDS readout noise
        cds_ron = wlc_inputs['CDS_RON']
        # deal with no cds_ron given
        if cds_ron is None:
            emsg = ('FILE(s) found to be CDS: WLC.INPUTS.CDS_RON must be '
                    'set in the parameters')
            raise exceptions.SossisseConstantException(emsg)
        # get the difference between the first and last frames
        tmp_data = data[:, last, :, :] - data[:, first, :, :]
        # work out the err and dq values from the cds
        tmp_err = np.sqrt(np.abs(tmp_data) + cds_ron)
        tmp_dq = ~np.isfinite(tmp_data)
        # return the data, err, dq
        return tmp_data, tmp_err, tmp_dq

    # TODO: implement with other functions when first reading the raw files?
    def get_int_times(self) -> np.ndarray:
        """
        Get the integration times (BJD: Barycentric Julian Day)
        :return: np.ndarray, the integration times
        """
        int_times = np.array([])
        for ifile, filename in enumerate(self.params['GENERAL.FILES']):
            # get the raw files
            tmp_data = self.load_data(filename, extname='INT_TIMES')
            # TODO: Known jwst pipeline bug where MJD = BJD
            mjd_times = tmp_data['int_mid_MJD_UTC']
            bjd_times = tmp_data['int_mid_BJD_TDB']
            # conditions to check whether bjd is really bjd (within 1 hour)
            if np.abs((mjd_times[0] - bjd_times[0])) < (1/24):
                emsg = (f'BJD is the same as MJD times.\n'+
                        f'Adding 2400000.5 to the BJD times '
                        f'(seg{ifile+1}[0]: {bjd_times[0]}, {bjd_times[-1]})')
                misc.printc(emsg, 'warning')
                # push to int_times
                int_times = np.append(int_times, bjd_times + 2400000.5)
            else:
                # push to int_times
                int_times = np.append(int_times, bjd_times)
            # make sure tmp data is deleted
            del tmp_data
        return int_times

    def load_data_with_dq(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and correct the data
        :return:
        """
        # set function name
        func_name = f'{__NAME__}.load_data_with_dq()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = self.get_variable('TEMP_INI_CUBE', func_name)
        temp_ini_err = self.get_variable('TEMP_INI_ERR', func_name)
        temp_ini_dq = self.get_variable('TEMP_INI_DQ', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            # check if cube, err and dq files exist
            cond1 = os.path.exists(temp_ini_cube)
            cond2 = os.path.exists(temp_ini_err)
            cond3 = os.path.exists(temp_ini_dq)
            # only if we have all three files do we read them
            if cond1 and cond2 and cond3:
                # read the cube
                misc.printc('Reading temporary file: {0}'.format(temp_ini_cube),
                            'info')
                cube = self.load_data(temp_ini_cube)
                # rea dthe error
                misc.printc('Reading temporary file: {0}'.format(temp_ini_err),
                            'info')
                err = self.load_data(temp_ini_err)
                # read the dq
                misc.printc('Reading temporary file: {0}'.format(temp_ini_dq),
                            'info')
                dq = self.load_data(temp_ini_dq)
                # for future reference in the code, we keep track of data size
                self.set_variable('DATA_X_SIZE', cube.shape[2])
                self.set_variable('DATA_Y_SIZE', cube.shape[1])
                self.set_variable('DATA_N_FRAMES', cube.shape[0])
                # return
                return cube, err, dq
        # ---------------------------------------------------------------------
        # deal with no files
        if len(self.params['GENERAL.FILES']) == 0:
            emsg = 'No files defined in yaml. Please set the FILES parameter.'
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # number of slices in the final cube
        n_slices = 0
        # storage of raw data and shape
        raw_shapes = []
        # handling the files with their varying sizes. We read and count slices
        # TODO: Should be replaced with header keys
        for ifile, filename in enumerate(self.params['GENERAL.FILES']):
            # get the raw files
            tmp_data = self.load_data(filename, extname='SCI')
            # get the shape of the bins
            bin_shape = self.bin_cube(tmp_data, get_shape=True)
            # add to the number of slices
            n_slices += bin_shape[0]
            # store the shape of the raw data
            raw_shapes.append(bin_shape)
            # make sure tmp data is deleted
            del tmp_data
        # ---------------------------------------------------------------------
        # identify cds and get image shape
        # ---------------------------------------------------------------------
        image_shape, flag_cds = self.id_image_shape(raw_shapes)
        # ---------------------------------------------------------------------
        # recalculate tags
        self.update_meta_data()
        # ---------------------------------------------------------------------
        # load and bin the cube
        cube, err, dq = self.load_cube(n_slices, image_shape, flag_cds)
        # ---------------------------------------------------------------------
        # for future reference in the code, we keep track of data size
        self.set_variable('DATA_X_SIZE', cube.shape[2])
        self.set_variable('DATA_Y_SIZE', cube.shape[1])
        self.set_variable('DATA_N_FRAMES', cube.shape[0])
        # ---------------------------------------------------------------------
        return cube, err, dq

    def apply_flat_field(self, cube, err, dq):
        """
        Flag field the cube and error cube
        :return:
        """
        # set function name
        func_name = f'{__NAME__}.apply_flat_field()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # construct temporary file names
        temp_ff_cube = self.get_variable('TEMP_FF_CUBE', func_name)
        temp_ff_err = self.get_variable('TEMP_FF_ERR', func_name)
        temp_ff_dq = self.get_variable('TEMP_FF_DQ', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            # check if cube, err and dq files exist
            cond1 = os.path.exists(temp_ff_cube)
            cond2 = os.path.exists(temp_ff_err)
            cond3 = os.path.exists(temp_ff_dq)
            # only if we have all three files do we read them
            if cond1 and cond2 and cond3:
                # read the cube
                misc.printc('Reading ff file: {0}'.format(temp_ff_cube),
                            'info')
                cube = self.load_data(temp_ff_cube)
                # rea dthe error
                misc.printc('Reading ff file: {0}'.format(temp_ff_err),
                            'info')
                err = self.load_data(temp_ff_err)
                # read the dq
                misc.printc('Reading ff file: {0}'.format(temp_ff_dq),
                            'info')
                dq = self.load_data(temp_ff_dq)
                # update the data sizes (is this required?)
                self.set_variable('DATA_X_SIZE', cube.shape[2])
                self.set_variable('DATA_Y_SIZE', cube.shape[1])
                self.set_variable('DATA_N_FRAMES', cube.shape[0])
                # return
                return cube, err, dq
        # ---------------------------------------------------------------------
        # get flat
        flat, no_flat = self.get_flat(cube.shape[1:])
        # ---------------------------------------------------------------------
        # apply the flat (may be ones)
        if no_flat:
            return cube, err, dq
        # ---------------------------------------------------------------------
        # keep the un-flat-fielded first frame of the cube
        frame0 = np.array(cube[0])
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
            margs = [temp_ff_cube, temp_ff_err]
            misc.printc(msg.format(*margs), 'info')
            # force cubes to be float
            cube = cube.astype(float)
            err = err.astype(float)
            # save the data
            fits.writeto(temp_ff_cube, cube, overwrite=True)
            fits.writeto(temp_ff_err, err, overwrite=True)
            fits.writeto(temp_ff_dq, dq, overwrite=True)
        # ---------------------------------------------------------------------
        plots.plot_flat_field(self, frame0, cube[0])
        # update the data sizes (is this required?)
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
            for f_it, filename in enumerate(self.params['GENERAL.FILES']):
                emsg += f'\n\t{filename}: {image_shapes[f_it]}'
            raise exceptions.SossisseFileException(emsg)
        # ---------------------------------------------------------------------
        # check that all are CDS or all are not CDS
        if len(set(all_cds)) != 1:
            emsg = 'Inconsistent CDS format (Eiher all CDS or not):'
            for f_it, filename in enumerate(self.params['GENERAL.FILES']):
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
        # get the linear model parameters
        lm_params = self.params.get('WLC.LMODEL')
        # don't bin if user doesn't want to bin
        if not lm_params['DATA_BIN_TIME']:
            if get_shape:
                return list(cube.shape)
            else:
                return cube
        # don't bin if the number of frames in each bin is 1
        if lm_params['DATA_BIN_SIZE'] == 1:
            if get_shape:
                return list(cube.shape)
            else:
                return cube
        # get the bin size
        bin_size = lm_params['DATA_BIN_SIZE']
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
        if self.params['GENERAL.FLATFILE'] is None:
            # flat field is a single frame
            return np.ones(image_shape), True
        else:
            # load the flat field
            flat = self.load_data(self.params['GENERAL.FLATFILE'])
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

    def remove_cosmic_rays(self, cube: np.ndarray) -> np.ndarray:
        """
        Remove cosmic rays from the cube (sigma cut)

        :param cube: np.ndarray, the cube to remove cosmic rays from

        :return: np.ndarray, the cube with cosmic rays removed
        """
        # get function name
        func_name = f'{__NAME__}.remove_cosmic_rays()'
        # see if user wants to remove cosmics
        if not self.params['WLC.GENERAL.REMOVE_COSMIC_RAYS']:      
            # print message that we are not removing cosmic rays
            msg = ('WLC.GENERAL.REMOVE_COSMIC_RAYS=False. '
                   'Not removing cosmic rays')
            misc.printc(msg, 'info')
            # return original cube
            return cube

        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # construct temporary file names
        temp_postcosic = self.get_variable('TEMP_CUBE_POST_COSMICS', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            if os.path.exists(temp_postcosic):
                # print that we are reading files
                misc.printc('Reading temporary file: {0}'.format(temp_postcosic),
                            'info')
                # load the data
                cube = self.load_data(temp_postcosic)
                # return
                return cube
        # ---------------------------------------------------------------------
        # print progress
        msg = 'We remove cosmisc rays'
        misc.printc(msg, 'info')
        # first we get a median of the 1-sigma cube
        msg = '\tCalculating median of 1-sigma cube'
        misc.printc(msg, '')
        # We first get the median of the -1 to +1 sigma of the cube
        with warnings.catch_warnings(record=True) as _:
            # we don't care about the warnings
            # we just want to get the percentiles
            # we know that there are some nans
            p16, p84 = np.nanpercentile(cube, [16, 84], axis=0)

        # get the sigma and mean
        sigma = (p84 - p16) / 2
        mean = (p84 + p16) / 2
        # get the sig cut
        sig_cut = self.params['WLC.GENERAL.COSMIC_RAY_SIGMA']
        # storage for heat map
        heat_map = np.zeros(cube.shape[1:], dtype=float)
        # store the first frame from before
        frame_before = np.array(cube[0])
        # now remove the cosmics using a sigma flag
        for iframe in tqdm(range(cube.shape[0])):
            # get the frame
            frame = cube[iframe]
            # calculate the number of sigma away from the mean every pixel is
            nsig = (frame - mean) / sigma
            # cosmic mask
            cmask = nsig > sig_cut
            # add to heat map
            heat_map[cmask] += 1
            # set those above the threshold to nan
            cube[iframe, cmask] = np.nan
        # ---------------------------------------------------------------------
        # plot fractions of good pixels (per pixel) as map
        plots.plot_heatmap(self, heat_map, frame_before,
                           cube[0], 'cosmic rays rate', 'cosmic_rays_corr',
                           'Number of cosmic rays')
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp:
            # print progress
            msg = ('We write intermediate files, they will be read to speed '
                   'things next time\n\ttemp cube: {0}')
            margs = [temp_postcosic]
            misc.printc(msg.format(*margs), 'info')
            # force cubes to be float
            cube = cube.astype(float)
            # save the data
            fits.writeto(temp_postcosic, cube, overwrite=True)
        # ---------------------------------------------------------------------
        # return the cube
        return cube

    def apply_dq(self, cube: np.ndarray, err: np.ndarray, dq: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dq to the cube

        :param cube: np.ndarray, the cube to remove the background from
        :param err: np.ndarray, the error cube
        :param dq: np.ndarray, the data quality cube

        :return: tuple, 1. np.ndarray, the background corrected cube
                        2. np.ndarray, the background corrected error cube
        """
        # get wlc_params
        wlc_params = self.params.get('WLC')
        # ---------------------------------------------------------------------
        # deal with user not wanting to apply dq flags
        if not wlc_params['INPUTS.APPLY_DQ_FLAGS']:
            # print message that we are not patching isolated bad pixels
            msg = 'WLC.INPUTS.APPLY_DQ_FLAGS = False. Not applying DQ flags'
            misc.printc(msg, 'info')
            # return original cube and error
            return cube, err
        # ---------------------------------------------------------------------
        # get valid dq values
        valid_dqs = wlc_params['INPUTS.VALID_DQ']
        if valid_dqs is None:
            emsg = 'WLC.INPUTS.VALID_DQ not set. Please set the valid DQ values'
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # trick to get a mask where True is valid
        cube_mask = np.zeros_like(cube, dtype=bool)
        for valid_dq in valid_dqs:
            # print DQ values
            misc.printc(f'Accepting DQ = {valid_dq}', 'number')
            # get the mask
            cube_mask[dq == valid_dq] = True
        # ---------------------------------------------------------------------
        # mask the values in cube_mask
        cube[~cube_mask] = np.nan
        err[~cube_mask] = np.inf
        # ---------------------------------------------------------------------
        # return the background corrected cube
        return cube, err

    def remove_background(self, cube: np.ndarray, err: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes the background with a 3 DOF model. It's the background image
        times a slope + a DC offset (1 amp, 1 slope, 1 DC)

        Background file is loaded from params['GENERAL.BKGFILE']

        :param cube: np.ndarray, the cube to remove the background from
        :param err: np.ndarray, the error cube

        :return: tuple, 1. np.ndarray, the background corrected cube
                        2. np.ndarray, the background corrected error cube
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.remove_background()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
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
        if not self.params['GENERAL.DO_BACKGROUND']:
            # print progress
            msg = 'We do not clean background. GENERAL.DO_BACKGROUND=False'
            misc.printc(msg, 'warning')
            # return the cube (unchanged)
            return cube, err
        # update the meta data
        self.update_meta_data()
        # store uncorrected frame
        frame0 = np.array(cube[0])
        # ---------------------------------------------------------------------
        # optimal background correction
        # ---------------------------------------------------------------------
        # print progress
        msg = 'Apply the background model correction'
        misc.printc(msg, 'info')
        # get the background file
        background = self.load_data(self.params['GENERAL.BKGFILE'])
        # force the background to floats
        background = background.astype(float)
        # calculate the mean of the cube (along time axis)
        with warnings.catch_warnings(record=True) as _:
            mcube = np.nanmean(cube, axis=0)
        # ---------------------------------------------------------------------
        # get the box size from params
        box = self.params['WLC.INPUTS.BACKGROUND_GLITCH_BOX']
        if box is None:
            emsg = ('WLC.INPUTS.BACKGROUND_GLITCH_BOX not set. '
                    'Please set the BACKGROUND_GLITCH_BOX')
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # get the background shifts
        bgnd_shifts_values = self.params['WLC.INPUTS.BACKGROUND_SHIFTS']
        if bgnd_shifts_values is None:
            emsg = ('WLC.INPUTS.BACKGROUND_SHIFTS not set. '
                    'Please set the BACKGROUND_SHIFTS')
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------
        # find the optimal shift
        rms_min = np.argmin(rms)
        # if the minimum is at the edge, we just say that the offset is 0
        if rms_min == 0 or rms_min == len(bgnd_shifts) - 1:
            optimal_offset = 0.0
            msg = ('\tHere we just take the offset to be 0 '
                   '(Optimal obackgroun offset not found)')
            misc.printc(msg, 'number')
        # otherwise we work out the optimal offset
        else:
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
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------
        # plot the background correction
        plots.plot_background(self, frame0, cube[0])
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
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
        return cube, err


    def low_pass_filter(self, cube, err) -> Tuple[np.ndarray, np.ndarray]:
        # set function name
        func_name = f'{__NAME__}.{self.name}.remove_background()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = self.get_variable('TEMP_INI_CUBE_LOWPASS', func_name)
        temp_ini_err = self.get_variable('TEMP_INI_ERR_LOWPASS', func_name)
        # store the uncorrected first frame
        frame0 = np.array(cube[0])
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
        # ---------------------------------------------------------------------
        # work out the mean of the cube
        with warnings.catch_warnings(record=True) as _:
            mcube = np.nanmean(cube, axis=0)
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

        sum_cube_tile = np.zeros_like(cube[0])
        # loop around frames
        for iframe in tqdm(range(cube.shape[0])):
            with warnings.catch_warnings(record=True) as _:
                med = np.nanmedian(cube[iframe] * mean_mask, axis=0)
                lowp = mp.lowpassfilter(med, 25)
            # subtract the low pass filter from this frame of the cube
            cubetile = np.tile(lowp, mcube.shape[0]).reshape(mcube.shape)
            cube[iframe] -= cubetile
            # store the sum of the corrections for plotting
            sum_cube_tile += cubetile
        # normalize by number of frames
        sum_cube_tile /= cube.shape[0]
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
        plots.plot_lowpass(self, frame0, cube[0], sum_cube_tile)
        # ---------------------------------------------------------------------
        # return the background corrected cube
        return cube, err

    def patch_isolated_bads(self, cube: np.ndarray, err: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Patch isolated bad pixels in the cube

        :param cube: np.ndarray, the cube to patch

        :return: np.ndarray, the patched cube
        """
        # set function name
        func_name = f'{__NAME__}.patch_isolated_bads()'
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # construct temporary file names
        temp_clean_nan = self.get_variable('TEMP_CLEAN_NAN', func_name)
        temp_clean_nan_err = self.get_variable('TEMP_CLEAN_NAN_ERR', func_name)
        # get WLC.GENERAL parameters
        wlc_params = self.params.get('WLC.GENERAL')
        # ---------------------------------------------------------------------
        # deal with no patching isolated bad pixels
        if not wlc_params['PATCH_ISOLATED_BADS']:
            # print message that we are not patching isolated bad pixels
            msg = ('WLC.GENERAL.PATCH_ISOLATED_BADS=False. Not patching '
                   'isolated bad pixels')
            misc.printc(msg, 'info')
            # return the cube (unchanged)
            return cube, err
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
                err = self.load_data(temp_clean_nan_err)
                return cube, err
        # ---------------------------------------------------------------------
        # print progress
        msg = ' Removing isolated NaNs'
        misc.printc(msg, 'info')
        # storage for the heat map
        heat_map = np.zeros_like(cube[0], dtype=float)
        # storage of bad pixels
        bad_pixels_before = dict()
        bad_pixels_after = dict()
        # get the size of the bad pixel stamps
        if wlc_params['PATCH_IBADS_SSIZE'] % 2 == 0:
            emsg = 'WLC.GENERAL.PATCH_IBADS_SSIZE must be odd'
            raise exceptions.SossisseConstantException(emsg)
        else:
            bpixel_ssize = (wlc_params['PATCH_IBADS_SSIZE'] - 1) // 2
        # ---------------------------------------------------------------------
        # store cube[0] before (for plot)
        iframe_before = np.array(cube[0])
        # store the positions of the bad pixels
        badpix_pos = dict()
        # loop around frames in cube
        for it, iframe in tqdm(enumerate(range(cube.shape[0]))):
            # sort out storage for this frame
            bad_pixels_before[it] = []
            bad_pixels_after[it] = []
            # get the cube frame
            cframe = np.array(cube[iframe, :, :])
            ecframe = np.array(err[iframe, :, :])
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
            # add the isolated bad pixels to the heat map
            heat_map += isolated_bad.astype(int)
            # get the pixel positions with isolated bad pixels
            ypix, xpix = np.where(isolated_bad)
            # for each pixel replace the value with the mean of the 4 pixels
            #   around it
            mean_vals = np.mean([cframe[ypix - 1, xpix],
                                 cframe[ypix + 1, xpix],
                                 cframe[ypix, xpix - 1],
                                 cframe[ypix, xpix + 1]], axis=0)
            # do the same for the errors
            mean_err_vals = np.mean([ecframe[ypix - 1, xpix],
                                     ecframe[ypix + 1, xpix],
                                     ecframe[ypix, xpix - 1],
                                     ecframe[ypix, xpix + 1]], axis=0)
            # -----------------------------------------------------------------
            # get the bad pixel stamps (before correction)
            _bad_stamps = self.create_bad_pixel_stamps(ypix, xpix, cframe, 
                                                       bpixel_ssize)
            bad_pixels_before[it] = _bad_stamps
            # -----------------------------------------------------------------
            # update cframe and set make
            cframe[ypix, xpix] = mean_vals
            ecframe[ypix, xpix] = mean_err_vals
            # -----------------------------------------------------------------
            # get the bad pixel stamps (after correction)
            _bad_stamps = self.create_bad_pixel_stamps(ypix, xpix, cframe, 
                                                       bpixel_ssize)
            bad_pixels_after[it] = _bad_stamps
            # -----------------------------------------------------------------
            # mframe[ypix, xpix] = True
            # push back into the cube
            cube[iframe, :, :] = cframe
            err[iframe, :, :] = ecframe
            # save the bad pixel positions
            badpix_pos[iframe] = [ypix, xpix]
        # ---------------------------------------------------------------------
        plots.plot_heatmap(self, heat_map, iframe_before, cube[0],
                           'isolated bad pixels',
                           'isolated_badpixels',
                           'Number across all integrations')
        # ---------------------------------------------------------------------
        # create bad pixel images (one per integration)
        bpixel_cutouts_small, _ = self.create_stamp_images(bad_pixels_before,
                                                           bad_pixels_after,
                                                           badpix_pos,
                                                           num=100)
        csi_out = self.create_stamp_images(bad_pixels_before, bad_pixels_after,
                                           badpix_pos, mode='seperate')
        pixel_cutouts_all, table_cutouts_all = csi_out
        # plot this as a cut out
        plots.plot_pixels(self, bpixel_cutouts_small)
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
            fits.writeto(temp_clean_nan_err, err, overwrite=True)
        # ---------------------------------------------------------------------
        # save the cutouts of all bad pixels
        self.save_cutouts(pixel_cutouts_all, table_cutouts_all)
        # ---------------------------------------------------------------------
        # return the cube
        return cube, err


    def create_bad_pixel_stamps(self, ypix: np.ndarray, xpix: np.ndarray, 
                                array: np.ndarray, size: int):
        """
        Create a list of stamps around the bad pixels
        :param ypix: np.ndarray, the y positions of the bad pixels
        :param xpix: np.ndarray, the x positions of the bad pixels
        :param array: np.ndarray, the array to create stamps from
        :param size: int, the size of the stamp (half size, so 3 means 7x7)

        :return: list of np.ndarray, the stamps around the bad pixels
        """
        # storage for output list of stamps
        stamp_list = []

        # pad the input array with nans so we get edge pixels correctly
        array = np.pad(array, ((size, size), (size, size)), mode='constant',
                       constant_values=np.nan)

        # loop around all positions
        for jt in range(len(ypix)):
            # get the bad pixel stamp size 
            # (need to +size at the end as we padded)
            ystart = (ypix[jt] - size) + size
            yend = (ypix[jt] + size + 1) + size
            xstart = (xpix[jt] - size) + size
            xend = (xpix[jt] + size + 1) + size
            # get a cut out stamp
            stamp = array[ystart:yend, xstart:xend]
            # ignore edge stamps (that are too small)
            if stamp.shape[0] != (size * 2 + 1):
                continue
            if stamp.shape[1] != (size * 2 + 1):
                continue
            # store the bad pixels before
            with warnings.catch_warnings(record=True) as _:
                # ignore warnings about nans
                # normalize the stamp to between 0 and 1
                smin, smax = np.nanmin(stamp), np.nanmax(stamp)
                if smax - smin == 0:
                    # if the stamp is constant, we set it to 0
                    nstamp = np.zeros_like(stamp)
                else:
                    # normalize the stamp
                    nstamp = (stamp - smin) / (smax - smin)
            # append the normalized stamp to the list
            stamp_list.append(nstamp)
        # return the stamp list
        return stamp_list

    def save_cutouts(self, cutouts: Dict[int, Tuple[np.ndarray, np.ndarray]],
                     pos_table: Dict[int, Table]):
        # set function name
        func_name = f'{__NAME__}.save_cutouts()'
        # get the output directory
        outdir = self.get_variable('BADPIXEL_CUTOUT_DIR', func_name)
        # make the directory if it doesn't exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # print progress
        msg = ('Saving cutouts of all bad pixels to {0}.')
        margs = [outdir]
        misc.printc(msg.format(*margs), 'info')
        # ---------------------------------------------------------------------
        # loop around frames
        for iframe in tqdm(cutouts):
            # get the cutout
            cutout = cutouts[iframe]
            # create a fits file for this cutout
            if isinstance(cutout, tuple):
                hdu0 = fits.PrimaryHDU()
                hdu1 = fits.ImageHDU(cutout[0])
                hdu2 = fits.ImageHDU(cutout[1])
                hdu3 = fits.BinTableHDU(pos_table[iframe])
                hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
            else:
                hdu0 = fits.PrimaryHDU()
                hdu1 = fits.ImageHDU(cutout)
                hdu2 = fits.BinTableHDU(pos_table[iframe])
                hdul = fits.HDUList([hdu0, hdu1, hdu2])
            # write to file
            outfilename = f'{self.name}_badpixels_{iframe:03d}.fits'
            outfile = os.path.join(outdir, outfilename)
            hdul.writeto(outfile, overwrite=True)
            hdul.close()
        return

    def create_stamp_images(self, before_dict: Dict[int, List[np.ndarray]], 
                            after_dict: Dict[int, List[np.ndarray]],
                            pos_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                            num: Optional[int] = None,
                            mode: Literal['seperate', 'together'] = 'together'
                            ) -> Tuple[Dict[int, np.ndarray], Table]:
        # set function name
        func_name = f'{__NAME__}.Instrument.create_stamp_images()'
        # calculate the size of the stamps
        ssize = self.params['WLC.GENERAL.PATCH_IBADS_SSIZE']
        # storage of output images
        images = dict()
        # position of the bad pixels
        bad_table = dict()
        # loop around each frame
        for iframe in tqdm(range(len(before_dict))):
            # store positions
            bad_itable = dict(x=[], y=[], row=[], col=[])
            # get the before and after lists
            before_list = before_dict[iframe]
            after_list = after_dict[iframe]
            # -----------------------------------------------------------------
            # get length - and deal with differing lengths (shouldn't happen)
            if len(before_list) != len(after_list):
                emsg = 'Stamp images have inconsistent sizes. Function = {0}'
                eargs = [func_name]
                raise exceptions.SossisseException(emsg.format(*eargs))
            else:
                length = len(before_list)
            # -----------------------------------------------------------------
            # deal with cutting down number of pixels
            if num is not None and num < length:
                # select a random number "num" of pixels in the list
                indices = np.random.choice(len(before_list), num, replace=False)
                before_list = [before_list[i] for i in indices]
                after_list = [after_list[i] for i in indices]
                # update length parameter
                length = len(before_list)

            # -----------------------------------------------------------------
            tile_height = ssize
            # border of NaNs around the final image
            border = 3
            # individual
            if mode == 'seperate':
                tile_width = ssize
            # (before) + border (nan) +(after)
            else:
                tile_width = ssize + 1 + ssize
            # -----------------------------------------------------------------
            # Determine a grid size: make it as square as possible, 
            #    with height ~ 2*width
            # So we want (cols * tile_width) ~ (rows * tile_height), 
            #    and rows*cols >= length
            ideal_cols = int(np.sqrt(length / 2))
            ideal_rows = int(np.ceil(length / ideal_cols))
            # Make sure we have enough rows and columns
            cols, rows = ideal_cols, ideal_rows
            while rows * cols < length:
                cols += 1
            # -----------------------------------------------------------------
            # Prepare the final image size
            height = rows * tile_height + (rows + 1) * border
            width = cols * tile_width + (cols + 1) * border
            # set up two images (only use one if mode == together)
            image1 = np.full((height, width), np.nan)
            image2 = np.full((height, width), np.nan)
            # -----------------------------------------------------------------
            # Fill the image
            for idx in range(length):
                # work out the row and column
                row = idx // cols
                col = idx % cols
                # Calculate the position in the image
                # y0 is the top left corner of the tile
                # x0 is the left side of the tile
                y0 = row * tile_height + (row + 1) * border
                x0 = col * tile_width + (col + 1) * border
                # Extract the before and after arrays
                before = before_list[idx]
                after = after_list[idx]
                # update bad piexl itable
                bad_itable['x'].append(pos_dict[iframe][1][idx])
                bad_itable['y'].append(pos_dict[iframe][0][idx])
                bad_itable['row'].append(row)
                bad_itable['col'].append(col)
                # Insert them into the image
                if mode == 'seperate':
                    image1[y0:y0 + 5, x0:x0 + 5] = before
                    image2[y0:y0 + 5, x0:x0 + 5] = after
                else:
                    image1[y0:y0+5, x0:x0+5] = before
                    image1[y0:y0+5, x0+6:x0+11] = after
            # -----------------------------------------------------------------
            # construct table
            bad_itable = Table(bad_itable)
            # -----------------------------------------------------------------
            # store images
            if mode == 'seperate':
                images[iframe] = (image1, image2)
            else:
                images[iframe] = image1
            # store positions
            bad_table[iframe] = bad_itable


        return images, bad_table

    def optimize_trace_mask(self, log: bool = True):
        """
        Optimize the trace position and save a new pos mask file

        :return: None
        """
        _ = self, log
        raise NotImplementedError('optimize_trace_mask() must be implemented in '
                                  'child Instrument class')

    def get_trace_positions(self, med: np.ndarray = None,
                            cube: np.ndarray = None,  log: bool = True):
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map
        """
        _ = self, med, cube, log
        raise NotImplementedError('get_trace_pos() must be implemented in '
                                  'child Instrument class')

    def get_trace_mask(self, med: np.ndarray = None, cube: np.ndarray = None,
                       log: bool = True, no_plot: bool = True,
                       plot_frames: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the trace mask (True where the trace is, False otherwise)
        This is a combined map of all orders.

        :param med: np.ndarray, median image to plot the trace mask on
                    can be None if cube is supplied
        :param cube: np.ndarray, cube to plot the trace mask on
                     can be None if med is supplied
        :param log: bool, if True print progress
        :param no_plot: bool, if True do not plot the trace mask
        :param images: list of np.ndarray, images to overplot the trace mask on
                       if not given does not plot background image(s)
        :param labels: list of strs, labels for the images
        :return:
        """
        # set function name
        func_name = f'{__NAME__}.get_trace_mask()'
        # print progress
        if log:
            misc.printc('Getting the trace mask', 'info')
        # get x and y size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        # deal with no trace map required
        if self.params['WLC.GENERAL.LINRECON_TRACE_WIDTH'] < 1:
            # set the trace masm to ones
            trace_mask = np.ones((ysize, xsize), dtype=bool)
            # return the trace_mask
            return trace_mask
        # ---------------------------------------------------------------------
        # get the trace map (instrument dependent)
        trace_mask = self.get_trace_positions(med=med, cube=cube, log=log)
        # ---------------------------------------------------------------------
        # deal with wavelength domain cut down
        if self.params['GENERAL.WLC_DOMAIN'] is not None:
            # get the low and high values
            wavelow, wavehigh = self.params['GENERAL.WLC_DOMAIN']
            # print that we are cutting
            if log:
                msg = 'We cut the domain of the WLC to {0:.2f} - {1:.2f} um'
                margs = [wavelow, wavehigh]
                misc.printc(msg.format(*margs), 'number')
            # get the wavegrid
            wavegrid = self.get_wavegrid()
            # mask the trace map
            trace_mask[:, wavegrid < wavelow] = False
            trace_mask[:, wavegrid > wavehigh] = False
            # mask any nan values in the wavegrid
            trace_mask[:, ~np.isfinite(wavegrid)] = False
        # ---------------------------------------------------------------------
        # plot the trace mask
        if not no_plot:
            # get the images to plot
            if plot_frames is not None:
                if cube is None:
                    images = [med]
                    labels = ['Median image']
                else:
                    images = [cube[it] for it in plot_frames]
                    labels = [f'Frame{it}' for it in plot_frames]
            else:
                images, labels = None, None
            # plot the trace mask
            plots.plot_trace_mask(self, trace_mask, images, labels)
        # ---------------------------------------------------------------------
        # return the trace_mask
        return trace_mask

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
        # get wlc generation parameters
        gen_params = self.params.get('GENERAL')
        wlc_gen_params = self.params.get('WLC.GENERAL')
        # get x and y size from cube
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        xoffset = wlc_gen_params['X_TRACE_OFFSET']
        yoffset = wlc_gen_params['Y_TRACE_OFFSET']
        trace_wid_mask = wlc_gen_params['TRACE_WIDTH_MASKING']
        # load the trace position
        tbl_ref = self.load_table(gen_params['POS_FILE'], ext=order_num)
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
        if round_pos:
            posmax = np.array(posmax).astype(float)

        throughput = np.array(spline_throughput(rxpos), dtype=float)

        # deal with out-of-bounds posmax
        valid = (posmax > 0) & (posmax < ysize - 1)
        if np.sum(~valid) > 0:
            posmax[~valid] = np.nan
        # ---------------------------------------------------------------------
        # deal with map2d
        if map2d:
            # make a mask of the image
            posmap = np.zeros([ysize, xsize], dtype=bool)
            # loop around pixels in the x direction
            for ix_pix in range(xsize):
                # deal with nans
                if np.isnan(posmax[ix_pix]):
                    continue
                # get the top and bottom of the trace
                bottom = int(posmax[ix_pix] - trace_wid_mask // 2) + 1
                top = int(posmax[ix_pix] + trace_wid_mask // 2) + 1
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
                     return_xpix: bool = False,
                     source: Optional[str] = None
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the wave grid for the instrument

        :param order_num: int, the order number to use (if source is pos)
        :param return_xpix: bool, if True return xpix as well as wave

        :return: np.ndarray, the wave grid
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_wavegrid()'
        # get wave file definition
        wave_file = self.params['GENERAL.WAVE_FILE']
        # get the wave type
        wave_type = self.params['GENERAL.WAVE_FILE_TYPE']
        # get x size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        # ---------------------------------------------------------------------
        # if order_num is set currently we have to get wavelength solution from
        # the position file (this may change in the future)
        if order_num is not None:
            source = 'pos'
        # if we have a wave file set and a wave type set then we should
        # be loading from wavefile
        elif source is None and wave_file is not None and wave_type is not None:
            source = 'wavefile'
            # deal with base wave type
            if wave_type not in ['ext1d', 'fits', 'hdf5']:
                emsg = f'WAVE_TYPE must be ext1d, fits or hdf5'
                raise exceptions.SossisseConstantException(emsg)
        # if source is still None set it to "pos"
        if source is None:
            source = 'pos'
        # ---------------------------------------------------------------------
        # Deal with ext1d wave file - by default not valid
        if wave_type == 'ext1d' and source == 'wavefile':
            # get the full path to wave file
            wavepath = os.path.join(self.params['PATHS.CALIBPATH'], wave_file)
            # use the default function to load the wave fits file
            xpix, wavevector = io.load_wave_ext1d(str(wavepath))
        # ---------------------------------------------------------------------
        # Deal with fits file wave file
        elif wave_type == 'fits' and source == 'wavefile':
            # get the full path to wave file
            wavepath = os.path.join(self.params['PATHS.CALIBPATH'], wave_file)
            # use the default function to load the wave fits file
            xpix, wavevector = io.load_wave_fits(str(wavepath))
        # ---------------------------------------------------------------------
        # Deal with hdf5 fits file wave file
        elif wave_type == 'hdf5' and source == 'wavefile':
            # get the full path to wave file
            wavepath = os.path.join(self.params['PATHS.CALIBPATH'],
                                    wave_file)
            # use the default function to load the wave fits file
            xpix, wavevector = io.load_wave_hdf5(str(wavepath), xsize)
        # ---------------------------------------------------------------------
        # otherwise we load from pos file
        else:
            # deal with case where we need POS_FILE and it is not given
            if self.params['GENERAL.POS_FILE'] is None:
                emsg = (f'POS_FILE must be defined for {self.name}'
                        f'\n\tfunction = {func_name}')
                raise exceptions.SossisseConstantException(emsg)
            # deal with order num not being set
            elif order_num is None:
                emsg = ('order_num must be defined when using source="pos"'
                        f'\n\tfunction = {func_name}')
                raise exceptions.SossisseConstantException(emsg)
            # otherwise we use POS_FILE
            else:
                # get x trace offset
                xtraceoffset = self.params['WLC.GENERAL.X_TRACE_OFFSET']
                # get the trace position file
                tbl_ref = self.load_table(self.params['GENERAL.POS_FILE'],
                                          ext=order_num)
                xpix, wavevector = io.load_wave_posfile(tbl_ref, xsize,
                                                        xtraceoffset)
        # ---------------------------------------------------------------------
        # return the wave grid
        if return_xpix:
            return xpix, wavevector
        else:
            return wavevector

    def create_median_stack(self, cube: np.ndarray) -> List[Union[np.ndarray]]:
        """
        Clean the 1/f noise from the cube

        :param cube: np.ndarray, the cube to clean

        :return: tuple, 1. the clean cube, 2. the median image,
                 3. the tmp before after clean1f, 4. the transit in vs out
        """
        # define the function name
        func_name = f'{__NAME__}.{self.name}.clean_1f()'
        # ---------------------------------------------------------------------
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        # ---------------------------------------------------------------------
        # save these for later
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        clean_cube_file = self.get_variable('CLEAN_CUBE_FILE', func_name)
        tmp_pcas = self.get_variable('TEMP_PCA_FILE', func_name)
        tmp_amps = self.get_variable('TEMP_AMP_FILE', func_name)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            # make sure all required files exist
            cond = os.path.exists(median_image_file)
            cond &= os.path.exists(clean_cube_file)
            # only look for fit pca file if we are fitting pca
            if self.params['WLC.LMODEL.FIT_PCA']:
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
                # load the tmp file
                misc.printc('\tReading: {0}'.format(tmp_amps), 'info')
                amps = self.load_data(tmp_amps)
                # return these files
                return_list = [clean_cube, median_image, amps]
                return return_list
        # ---------------------------------------------------------------------
        # create a copy of the cube, we will normalize the amplitude of each
        # trace
        cube2 = np.array(cube, dtype=float)
        # first estimate of the trace amplitude
        misc.printc('\tFirst median of cube to create trace esimate', 'info')
        # validate out-of-transit domain
        self.get_baseline_params()
        has_baseline = self.get_variable('HAS_BASELINE', func_name)
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # get flag for median out of transit
        med_baseline = self.params['WLC.GENERAL.MEDIAN_BASELINE']
        # ---------------------------------------------------------------------
        # deal with creating median
        with warnings.catch_warnings(record=True) as _:
            if med_baseline and has_baseline:
                med = np.nanmedian(cube2[baseline_domain], axis=0)
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
            if med_baseline:
                med = np.nanmedian(cube2[baseline_domain], axis=0)
            else:
                med = np.nanmedian(cube2, axis=0)
        # ---------------------------------------------------------------------
        # write files to disk
        # ---------------------------------------------------------------------
        # write the median image
        misc.printc('\tWriting: {0}'.format(median_image_file), 'info')
        fits.writeto(median_image_file, med, overwrite=True)
        # save temporary files
        if allow_temp:
            # write the clean cube
            misc.printc('\tWriting: {0}'.format(clean_cube_file), 'info')
            fits.writeto(clean_cube_file, cube, overwrite=True)
        # ---------------------------------------------------------------------
        return [cube, med, amps]

    def clean_residual_1f(self, cube: np.ndarray, err: np.ndarray,
                          med: np.ndarray, amps: np.ndarray,
                          trace_mask: np.ndarray) -> np.ndarray:
        # define the function name
        func_name = f'{__NAME__}.{self.name}.clean_1f()'
        # ---------------------------------------------------------------------
        # save these for later
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        # ---------------------------------------------------------------------
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        # ---------------------------------------------------------------------
        # Dealing with user turning this off
        if not self.params['WLC.INPUTS.APPLY_1F_CORR']:
            # print message that we are not removing cosmic rays
            msg = ('WLC.INPUTS.APPLY_1F_CORR=False. '
                   'Not applying 1/f correction')
            misc.printc(msg, 'info')
            return cube
        # ---------------------------------------------------------------------
        # print progress
        misc.printc('\nCalculating residuals', 'info')
        # work out the residuals
        residuals = np.zeros_like(cube)
        # loop around frames
        for iframe in tqdm(range(nframes)):
            # mask the nans and apply trace map
            valid = np.isfinite(cube[iframe]) & np.isfinite(med)
            valid &= trace_mask
            # work out the amplitudes in the valid regions
            part1b = np.nansum(cube[iframe][valid] * med[valid])
            part2b = np.nansum(med[valid] ** 2)
            amps[iframe] = part1b / part2b
            # we get the appropriate slice of the error cube
            residuals[iframe] = cube[iframe] - med * amps[iframe]
        # ---------------------------------------------------------------------
        # Subtract of the 1/f noise
        # ---------------------------------------------------------------------
        cube1, scorr = self.subtract_1f(residuals, cube, err, trace_mask)
        # plot the pvalues
        plots.plot_subtract_1f_scorr(self, scorr)
        # plot the first frame before and after 1/f correction
        plots.plot_subtract_1f_comp(self, cube, cube1)
        # ---------------------------------------------------------------------
        # write files to disk
        # ---------------------------------------------------------------------
        if allow_temp:
            # write the median image
            misc.printc('\tWriting: {0}'.format(median_image_file), 'info')
            fits.writeto(median_image_file, med, overwrite=True)
        # ---------------------------------------------------------------------
        # return the cleaned cube, the median image, the median difference
        return cube1

    def process_baseline_ints(self,
                             raw_baseline_ints: Union[None, List[List[int]]]
                             ) -> List[List[int]]:
        """
        Make sure the TRANSIT_INTS parameter are in the correct format
        i.e. None --> []
             lists of len 2 are ints and forced to length 4 (2nd=1st, 3rd=4th)
             lists of len 4 are ints

        :param raw_transit_ints: Union[None, List[List[int]]], the raw transit
                                 integrations
        :return: List[List[int]], the processed transit integrations
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.process_transit_ints()'
        # get the number of frames
        data_n_frames = self.get_variable('DATA_N_FRAMES', func_name)
        # if we don't have transit integrations defined we have no transit
        if raw_baseline_ints is None:
            baseline_ints = [[0, data_n_frames]]
        # otherwise we need a list of lists(length 2 or 4)
        elif isinstance(raw_baseline_ints, list):
            # output transit integrations
            baseline_ints = []
            # loop around raw transit integrations
            for row in range(len(raw_baseline_ints)):
                # if we have a length 4 list
                if len(raw_baseline_ints[row]) == 2:
                    # storage for this specific transit/ecplise
                    baseline_ints_row = []
                    # make sure they are all integers
                    try:
                        for integration in raw_baseline_ints[row]:
                            # test that we have a valid integer
                            valid_int = int(integration)
                            # deal with out-of-bounds value
                            if valid_int < 0:
                                valid_int = 0
                            elif valid_int > data_n_frames:
                                valid_int = data_n_frames
                            # append to the list
                            baseline_ints_row.append(valid_int)
                    except Exception as _:
                        emsg = ('BASELINE_INTS[{0}] must be a list of ints '
                                'between 0 and {1}')
                        emsg = emsg.format(row, data_n_frames)
                        raise exceptions.SossisseConstantException(emsg)
                    # we then add this row to the transit_ints
                    # but sort from smallest to largest
                    baseline_ints_row = np.sort(baseline_ints_row)
                    baseline_ints.append(list(baseline_ints_row))
                else:
                    emsg = ('BASELINE_INTS[{0}] must be a list of ints of '
                            'length 2').format(row)
                    raise exceptions.SossisseConstantException(emsg)
        else:
            emsg = 'BASELINE_INTS must be a list of lists or None'
            raise exceptions.SossisseConstantException(emsg)
        # return the transit integrations in the correct format
        return baseline_ints

    def get_baseline_params(self):
        """
        Get the baseline domain (before, after and full removing any
        rejected domain including transits)

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_baseline_params()'
        # get the oot domain
        has_baseline = self._variables['HAS_BASELINE']
        # deal with value already found (we don't need to run this again)
        if has_baseline is not None:
            return
        # get the number of frames
        data_n_frames = self.get_variable('DATA_N_FRAMES', func_name)
        # get the baseline integrations
        raw_baseline_ints = self.params['WLC.INPUTS.BASELINE_INTS']
        # process the baseline integrations
        baseline_ints = self.process_baseline_ints(raw_baseline_ints)
        # get the rejection domain
        rej_domain = self.params['WLC.INPUTS.REJECT_DOMAIN']
        # ---------------------------------------------------------------------
        # get the valid out of transit domain
        # ---------------------------------------------------------------------
        # at first everything is considered as "out of transit"
        has_baseline = True
        valid_baseline = np.zeros(data_n_frames, dtype=bool)
        # loop around baseline integrations given
        for cframe in baseline_ints:
            # set the baseline frames to True
            valid_baseline[cframe[0]:cframe[1] + 1] = True
            # deal with the rejection of domain
            if rej_domain is not None:
                # get the rejection domain
                for ireject in range(len(rej_domain) // 2):
                    # get the start and end of the domain to reject
                    start = rej_domain[ireject * 2]
                    end = rej_domain[ireject * 2 + 1]
                    # set to False in valid_baseline
                    valid_baseline[start:end] = False
        # ---------------------------------------------------------------------
        # set flag
        self.set_variable('HAS_BASELINE', has_baseline)
        # update variables
        self.set_variable('BASELINE_INTS', baseline_ints)
        self.set_variable('BASELINE_DOMAIN', valid_baseline)

    def add_integration_times(self, ltable: Table):
        """
        Add integration times to the light curve table
        :param ltable: Table, the light curve table
        :return: Table, the updated light curve table
        """
        # get the integration times
        int_times = self.get_int_times()
        # add the integration times to the table
        ltable['bjd'] = int_times
        return ltable

    def subtract_1f(self, residuals: np.ndarray,
                    cube: np.ndarray, err: np.ndarray,
                    trace_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Do the actual subtraction of the 1/f noise, once we have the residuals

        :param residuals: np.ndarray, the residuals
        :param cube: np.ndarray, the cube
        :param err: np.ndarray, the error cube
        :param trace_mask: np.ndarray, the trace map

        :return:
        """
        # define the function name
        func_name = f'{__NAME__}.{self.name}.clean_1f()'
        # print progress
        misc.printc('\tSubtracting 1/f noise', 'info')
        # get the degree for the 1/f polynomial fit
        degree_1f_corr = self.params['WLC.GENERAL.DEGREE_1F_CORR']
        # get the number of frames
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # storage correction for plot
        scorr = np.zeros_like(cube)
        # deal with no poly fit of the 1/f noise
        if degree_1f_corr == 0:
            # get the median noise contribution
            with warnings.catch_warnings(record=True) as _:
                noise_1f = np.nanmedian(residuals, axis=1)
            # subtract this off the cube frame-by-frame
            for iframe in tqdm(range(nframes)):
                # we subtract the 1/f noise off each column
                for col in range(nbxpix):
                    # store the noise model as the pvalues
                    scorr[iframe, :, col] = noise_1f[iframe, col]
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
                    v1 = res[:, col]
                    err1 = err2[:, col]
                    index = np.arange(res.shape[0], dtype=float)
                    # find valid pixels
                    valid = np.isfinite(v1 + err1)
                    valid &= ~trace_mask[:, col]
                    valid &= np.abs(v1 / err1) < 5
                    # try fit the polynomial
                    # noinspection PyBroadException
                    try:
                        pfit = np.polyfit(index[valid], v1[valid],
                                          degree_1f_corr, w=1 / err1[valid])

                        pvalue = np.polyval(pfit, index)
                        # subtract the fit from the cube
                        cube[iframe, :, col] -= pvalue
                        # store the correction
                        scorr[iframe, :, col] = pvalue
                    except Exception as _:
                        # if the fit fails we just set the column to NaN
                        cube[iframe, :, col] = np.nan
                        # store the correction
                        scorr[iframe, :, col] = np.nan
        return cube, scorr

    def fit_pca(self, cube2: np.ndarray, err: np.ndarray,
                med: np.ndarray, trace_mask: np.ndarray
                ) -> Union[None, np.ndarray]:
        """
        Fit the PCA to the trace_mask

        :param cube2: np.ndarray, the cube to fit the PCA to
        :param err: np.ndarray, the error cube
        :param med: np.ndarray, the median image
        :param trace_mask: np.ndarray, the trace map

        :return: None if unable to do the PCA otherwise np.ndarray:
                 the principle components
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.fit_pca()'
        # ---------------------------------------------------------------------
        # get the conditions for allowing and using temporary files
        allow_temp = self.params['GENERAL.ALLOW_TEMPORARY']
        use_temp = self.params['GENERAL.USE_TEMPORARY']
        # update meta data
        self.update_meta_data()
        # make sure we have a pca file
        tmp_pcas = 'temporary_pcas.fits'
        tmp_pcas = os.path.join(self.params['PATHS.TEMP_PATH'], tmp_pcas)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        # ---------------------------------------------------------------------
        # get whether to fit pca and the number of fit components
        flag_fit_pca = self.params['WLC.LMODEL.FIT_PCA']
        n_comp = self.params['WLC.LMODEL.FIT_N_PCA']
        # ---------------------------------------------------------------------
        # if we aren't fitting or we fit no components return None
        if (not flag_fit_pca) or n_comp == 0:
            msg = ('WLC.LMODEL.FIT_PCA={0}  WLC.LMODEL.FIT_N_PCA={1} '
                   'Not fitting with PCA').format(flag_fit_pca, n_comp)
            misc.printc(msg, 'info')
            return None
        # ---------------------------------------------------------------------
        # only look for fit pca file if we are fitting pca
        if allow_temp and use_temp and os.path.exists(tmp_pcas):
            misc.printc('\tReading: {0}'.format(tmp_pcas), 'info')
            return self.load_data(tmp_pcas)

        # get the shape of the data
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # validate out-of-transit domain
        self.get_baseline_params()
        has_baseline = self.get_variable('HAS_BASELINE', func_name)
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # if we don't have oot domain we cannot do the pca analysis
        if not has_baseline:
            wmsg = ('Cannot do PCA analysis without baseline domain.'
                    '\n\tPlease set WLC.INPUTS.CONTACT_FRAMES to use PCA.')
            misc.printc(wmsg, 'warning')
            return None
        # ---------------------------------------------------------------------
        # print progress
        misc.printc('Fitting PCA', 'info')
        # ---------------------------------------------------------------------
        # only fit the pca to the flux in the trace (nan everything else)
        nanmask = np.ones_like(trace_mask, dtype=float)
        nanmask[~trace_mask] = np.nan
        # copy the normalized cube
        cube3ini = cube2[baseline_domain]
        # subtract off the median
        for iframe in tqdm(range(cube3ini.shape[0])):
            cube3ini[iframe] -= med
        # ---------------------------------------------------------------------
        # copy the normalized cube
        cube3 = np.array(cube3ini)
        # get the valid error domain
        err3 = err[baseline_domain]
        # apply the nanmask to the cube3
        for iframe in range(cube3.shape[0]):
            cube3[iframe] *= nanmask
        # reshape the cubes to flatten each frame into a 1D array
        cube3 = cube3.reshape([cube3.shape[0], nbypix * nbxpix])
        err3 = err3.reshape([err3.shape[0], nbypix * nbxpix])
        # find the bad pixels
        badpix = ~(np.isfinite(cube3) & np.isfinite(err3))
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
        # print process
        margs = [n_comp]
        msg = '\tComputing principle components (ncomp={0})...'
        misc.printc(msg.format(*margs), 'info')
        # compute the principle components
        with warnings.catch_warnings(record=True) as _:
            # set up the pca class
            pca_class = EMPCA(n_components=n_comp)
            # process
            misc.printc('\t\tRunning PCA fit, please wait...', 'number')
            start = time.time()
            # fit out data
            pca_fit = pca_class.fit(cube3[:, valid], weights=weights[:, valid])
            # process
            margs = [time.time() - start]
            misc.printc('\t\tFinished PCA fit, took {0:.2f}s'.format(*margs),
                        'number')
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

    def recenter_trace_position(self, trace_mask: np.ndarray,
                                med: np.ndarray) -> np.ndarray:
        # set function name
        func_name = f'{__NAME__}.{self.name}.recenter_trace_position()'

        wlc_gen_params = self.params.get('WLC.GENERAL')
        # deal with not wanting to recenter trace position
        if not wlc_gen_params['RECENTER_TRACE_POSITION']:
            # print message that we are not removing cosmic rays
            msg = ('WLC.GENERAL.RECENTER_TRACE_POSITION=False. '
                   'Not recentering trace position')
            misc.printc(msg, 'info')
            return trace_mask
        # print progress
        msg = 'Recentering trace position'
        misc.printc(msg, 'info')
        # ---------------------------------------------------------------------
        # get some parameters from instrument
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # ---------------------------------------------------------------------
        # print what we are doing
        msg = '\tScan to optimize position of trace mask to maximize flux'
        misc.printc(msg, 'info')
        # save the current width (we will reset it later
        width_current = float(wlc_gen_params['TRACE_WIDTH_MASKING'])
        # force a trace width masking
        wlc_gen_params['TRACE_WIDTH_MASKING'] = 20
        wlc_gen_params.set_source('TRACE_WIDTH_MASKING', func_name)
        # get the trace x and y scales
        trace_x_scale = wlc_gen_params['TRACE_X_SCALE']
        trace_y_scale = wlc_gen_params['TRACE_Y_SCALE']
        # get a range of dys and dxs to scan over for best trace position
        dys = np.arange(-nbypix // trace_y_scale, nbypix // trace_y_scale + 1)
        dys += wlc_gen_params['Y_TRACE_OFFSET']
        dxs = np.arange(-nbypix // trace_x_scale, nbypix // trace_x_scale + 1)
        dxs += wlc_gen_params['X_TRACE_OFFSET']
        sums = np.zeros([len(dxs), len(dys)], dtype=float)
        # storage for best dx and dy
        best_dx = 0
        best_dy = 0
        best_sum = 0
        # re-gen the trace map (without logging) using new x/y trace
        #  offset
        trace_mask = self.get_trace_mask(med=med, log=False)
        # re-get the trace_mask as floats
        tmask = np.array(trace_mask, dtype=float)
        # loop around dxs and dys
        for ix in tqdm(range(len(dxs))):
            for iy in range(len(dys)):
                # # do not process zero fluxes
                # if sums[ix, iy] == 0:
                #     continue
                # update the x and y positions
                wlc_gen_params['X_TRACE_OFFSET'] = dxs[ix]
                wlc_gen_params['Y_TRACE_OFFSET'] = dys[iy]
                # re-gen the trace map (without logging) using new x/y trace
                #  offset
                trace_mask = self.get_trace_mask(med=med, log=False)
                # re-get the trace_mask as floats
                tmask = np.array(trace_mask, dtype=float)
                # get the sum of the median image in the trace
                sums[ix, iy] = np.nansum(tmask * med)
                # # deal with worst values (set to sum)
                # if sums[ix, iy] < 0.5 * best_sum:
                #     sums[ix:ix+5, iy:iy+5] = sums[ix, iy]
                # deal with good values
                if sums[ix, iy] > best_sum:
                    best_sum = sums[ix, iy]
                    best_dx = dxs[ix]
                    best_dy = dys[iy]
        # print the best dx and dy
        misc.printc('Best dx : {} pix'.format(best_dx), 'number')
        misc.printc('Best dy : {} pix'.format(best_dy), 'number')
        # update the trace offsets with the best values found
        wlc_gen_params['X_TRACE_OFFSET'] = best_dx
        wlc_gen_params.set_source('X_TRACE_OFFSET', func_name)
        wlc_gen_params['Y_TRACE_OFFSET'] = best_dy
        wlc_gen_params.set_source('Y_TRACE_OFFSET', func_name)
        # reset the trace width masking to the user defined value
        wlc_gen_params['TRACE_WIDTH_MASKING'] = width_current
        wlc_gen_params.set_source('TRACE_WIDTH_MASKING', func_name)
        # ---------------------------------------------------------------------
        # get the loss in parts per thousand
        loss_ppt = (sums / np.nansum(sums)) * 1e3
        # get the maximum x value
        xmax = np.nanargmax(sums) // sums.shape[1]
        ymax = np.nanargmax(sums) % sums.shape[1]

        # print out of loss
        for iy in range(len(dys)):
            margs = [dys[iy], loss_ppt[xmax, iy]]
            msg = '\toffset dy = {0:3f}, gain = {1:3f} ppt'
            misc.printc(msg.format(*margs), 'number')
        # print the optimum value
        msg = 'We scanned the y position of trace, optimum at dy = {0}'
        margs = [wlc_gen_params['Y_TRACE_OFFSET']]
        misc.printc(msg.format(*margs), 'number')
        # plot the trace flux loss
        plots.plot_trace_flux_loss(self, sums, dxs, dys, xmax, ymax, loss_ppt,
                                   trace_mask, med, best_dx, best_dy)
        # return the updated trace map
        return self.get_trace_mask(med=med)

    def get_fit_params(self, med: np.ndarray,
                       no_plot: bool = False) -> List[np.ndarray]:
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
        dy, dx = np.gradient(med2)
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
        if not no_plot:
            plots.gradient_plot(self, med2, dx, dy, rotxy, ddy)
        # ---------------------------------------------------------------------
        # return these values
        return [dx, dy, rotxy, ddy, med2]

    def get_linear_recon_mask(self, med: np.ndarray, trace_mask: np.ndarray,
                              no_plot: bool = False) -> List[np.ndarray]:
        """
        Get the mask trace positions

        :param med: np.ndarray, the median image
        :param trace_mask: np.ndarray, the trace map

        :return: list, 1. the mask trace positions, 2. the x order 0 positions,
                       3. the y order 0 positions, 4. the x trace positions,
                       5. the y trace positions
        """
        # set up the mask trace (all true to start)
        mask_trace_pos = np.ones_like(med, dtype=int)
        # get the wlc general params
        wlc_gen_params = self.params.get('WLC.GENERAL')
        # ---------------------------------------------------------------------
        if wlc_gen_params['TRACE_WIDTH_MASKING'] != 0:
            mask_trace_pos[~trace_mask] = 0
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
        # return the mask trace positions
        return [mask_trace_pos, x_order0, y_order0, x_trace_pos, y_trace_pos]

    def setup_linear_reconstruction(self, med: np.ndarray, dx: np.ndarray,
                                    dy: np.ndarray, rotxy: np.ndarray,
                                    ddy: np.ndarray, pca: np.ndarray,
                                    ) -> np.ndarray:
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
        # get the linear model params
        lm_params = self.params.get('WLC.LMODEL')
        # ---------------------------------------------------------------------
        # deal with fit dx
        if lm_params['FIT_DX']:
            vector.append(dx.ravel())
            output_names.append('dx')
            output_units.append('mpix')
            output_factor.append(1e3)
        # ---------------------------------------------------------------------
        # deal with fix dy
        if lm_params['FIT_DY']:
            vector.append(dy.ravel())
            output_names.append('dy')
            output_units.append('mpix')
            output_factor.append(1e3)
        # ---------------------------------------------------------------------
        # deal with fit rotation
        if lm_params['FIT_ROTATION']:
            vector.append(rotxy.ravel())
            output_names.append('theta')
            output_units.append('mpix')
            output_factor.append(129600 / (2 * np.pi))
        # ---------------------------------------------------------------------
        # deal with zero point offset fit
        if lm_params['FIT_ZERO_POINT_OFFSET']:
            vector.append(np.ones_like(dx.ravel()))
            output_names.append('zeropoint')
            output_units.append('flux')
            output_factor.append(1.0)
        # ---------------------------------------------------------------------
        # deal with fit second derivative
        if lm_params['FIT_DDY']:
            vector.append(ddy.ravel())
            output_names.append('ddy')
            output_units.append('mpix$^2$')
            output_factor.append(1e6)
        # ---------------------------------------------------------------------
        # deal with fit pca
        if lm_params['FIT_PCA'] and pca is not None:
            n_comp = lm_params['FIT_N_PCA']
            for icomp in range(n_comp):
                vector.append(pca[icomp].ravel())
                output_names.append(f'PCA{icomp + 1}')
                output_units.append('ppm')
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

    def get_linear_coeffs(self, cube: np.ndarray, err: np.ndarray,
                          med: np.ndarray, mask_trace_pos: np.ndarray,
                          lvector: np.ndarray,
                          x_trace_pos: np.ndarray, y_trace_pos: np.ndarray,
                          x_order0: np.ndarray, y_order0: np.ndarray
                          ) -> Tuple[Table, np.ndarray, np.ndarray, np.ndarray]:
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
        zpoint = self.params['WLC.LMODEL.FIT_ZERO_POINT_OFFSET']
        # vectors to keep track of the rotation/amplitudes/dx/dy
        all_recon = np.zeros_like(cube)
        # starting point of the residual cube is the cube 
        #    (will be subtracted off per integration)
        rescube = np.array(cube)
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
        # print progress
        misc.printc('Linear recon of trace model', '')
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
            rescube[iframe] -= recon
            # update the recons (relative to the no model amplitude)
            all_recon[iframe] = recon / amp_model[0]
            # -----------------------------------------------------------------
            # force cube to floats
            tmp_slice = np.array(rescube[iframe], dtype=float)
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
        return output_table, all_recon, valid_arr, rescube

    def normalize_sum_trace(self, loutputs: Table) -> Table:
        """
        Normalize the sum trace by the median of the sum trace

        :param loutputs: dict, the outputs

        :return: dict, the normalized outputs
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.normalize_sum_trace()'
        # validate out-of-transit domain
        self.get_baseline_params()
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # get the normalization factor
        with warnings.catch_warnings(record=True) as _:
            norm_factor = np.nanmedian(loutputs[baseline_domain]['sum_trace'])
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
        self.get_baseline_params()
        has_baseline = self.get_variable('HAS_BASELINE', func_name)
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # if we don't have oot domain we cannot do the normalization
        if not has_baseline:
            wmsg = ('Cannot do per pixel baseline correction without '
                    'baseline domain.'
                    '\n\tPlease set WLC.INPUTS.BASELINE_INTS to do per pixel'
                    ' baseline correction.')
            misc.printc(wmsg, 'warning')
            # return loutputs without normalization
            return cube
        # ---------------------------------------------------------------------
        # get the polynomial degree for the transit baseline
        poly_order = self.params['WLC.GENERAL.TRANSIT_BASELINE_POLYORD']
        # ---------------------------------------------------------------------
        # print progress
        msg = 'Correcting the per pixel baseline'
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
                sample_oot = sample[baseline_domain]
                # get the indices of the out of transit domain
                frames_oot = frames[baseline_domain]
                # find any nans in the oot sample
                finite_mask = finite_mask_slice[:, iy][baseline_domain]
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
        # return the updated cube
        return cube

    def rms_baselines(self) -> List[str]:
            return ['naive_sigma', 'linear_sigma', 'lowpass_sigma',
                    'quadratic_sigma']

    def get_rms_baseline(self, vector: Union[np.ndarray, None] = None,
                         method: str = 'linear_sigma') -> float:
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
        if method not in self.rms_baselines():
            emsg = (f'Unknown method: {method}, available methods are: '
                    f'{",".join(self.rms_baselines())}')
            raise exceptions.SossisseException(emsg)
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

    def get_effective_wavelength(self, med: np.ndarray) -> Tuple[float, float]:
        """
        Get the effective wavelength factors (photon and energy weights)

        :return: tuple, 1. the photon weighted mean, 2. the energy weighted mean
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_effective_wavelength()'
        # get the trace_mask
        trace_mask = self.get_trace_mask(med=med)
        # get the wave grid
        wavegrid = self.get_wavegrid(order_num=1)
        # ---------------------------------------------------------------------
        # find the domain that has spectra
        with warnings.catch_warnings(record=True) as _:
            sp_domain = np.nanmean(trace_mask * med, axis=0)
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
        if self.params['GENERAL.WLC_DOMAIN'] is not None:
            msg = 'Domain:\t {0:.3f} -- {1:.3f} um'
            margs = self.params['WLO_DOMAIN']
            misc.printc(msg.format(*margs), 'number')
        else:
            msg = ('Full domain included, parameter GENERAL.WLC_DOMAIN '
                   'not defined')
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

    def save_wlc_results(self, rescube: np.ndarray, err: np.ndarray, 
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
        io.save_fits(resfile, datalist=[rescube], datatypes=['image'],
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
    def get_trace_orders(self) -> List[int]:
        """
        Get the trace orders
        If None this is set to [0]
        :return:
        """
        trace_orders = self.params['GENERAL.TRACE_ORDERS']
        # deal with no trace orders set (or none needed)
        if trace_orders is None:
            return [0]
        else:
            return trace_orders


    def load_input_spec_data(self, trace_order: int = 0) -> List[np.ndarray]:
        """
        Load input data for this trace order for the spectral extraction

        :param trace_order: int, trace order to load

        returns:
        - med: np.ndarray
        - dx: np.ndarray
        - dy: np.ndarray
        - rotxy: np.ndarray
        - ddy: np.ndarray
        - med_clean: np.ndarray
        - residual: np.ndarray
        - err: np.ndarray
        - recon: np.ndarray
        - ltable: np.ndarray
        - posmax: np.ndarray
        - throughput: np.ndarray
        - wavegrid: np.ndarray

        :return: list of numpy arrays: loaded input data (see above)
        """
        # set the function name
        func_name = f'{__NAME__}.spectral_extraction'
        # load the median image
        med_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        med = io.load_fits(med_file)
        # get clean median trace for spectrum
        dx, dy, rotxy, ddy, med_clean = self.get_fit_params(med, no_plot=True)
        # load the residuals
        res_file = self.get_variable('WLC_RES_FILE', func_name)
        residual = io.load_fits(res_file)
        # load the error file
        err_file = self.get_variable('WLC_ERR_FILE', func_name)
        err = io.load_fits(err_file)
        # load the residuals
        recon_file = self.get_variable('WLC_RECON_FILE', func_name)
        recon = io.load_fits(recon_file)
        # load the linear fit table
        ltable_file = self.get_variable('WLC_LTBL_FILE', func_name)
        ltable = io.load_table(ltable_file)
        # for future reference in the code, we keep track of data size
        self.set_variable('DATA_X_SIZE', med.shape[1], func_name)
        self.set_variable('DATA_Y_SIZE', med.shape[0], func_name)
        # get the trace position
        posmax, throughput = self.get_trace_pos(order_num=trace_order)
        # get wave grid
        wavegrid = self.get_wavegrid(order_num=trace_order)
        # return all input data
        return [med, dx, dy, rotxy, ddy, med_clean, residual, err,
                recon, ltable, posmax, throughput, wavegrid]



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
        # get the trace width extraction
        trace_width = self.params['WLC.GENERAL.LINRECON_TRACE_WIDTH']
        # ---------------------------------------------------------------------
        # construct the sed
        sp_sed = np.zeros(med.shape[1])
        # get a ribbon on the trace that extends over the input width
        for ix in range(med.shape[1]):
            # get width
            width = trace_width // 2
            # deal with posmax out-of-bounds
            if np.isnan(posmax[ix]):
                continue
            # get start and end positions
            ystart = int(posmax[ix] - width)
            yend = int(posmax[ix] + width)
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
        # load the trace_mask
        trace_mask = self.get_trace_mask(med=med)
        # ---------------------------------------------------------------------
        # the model starts as the recon
        model = np.array(recon)
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
        needs to include a propagation of errors. The error of pixels along
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
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
        # get the trace width extraction
        trace_width = self.params['WLC.GENERAL.LINRECON_TRACE_WIDTH']
        # ---------------------------------------------------------------------
        # print progress
        misc.printc('Finding the ratio of residuals to the trace', '')
        time.sleep(0.1)
        # placeholder for the cube spectra
        spec = np.full([nbframes, nbxpix], np.nan)
        spec_err = np.full([nbframes, nbxpix], np.nan)
        # get width
        width = trace_width // 2
        # only keep valid posmax (nans were placed where trace was out of
        #    bounds)
        bmask = np.isfinite(posmax)
        bmask &= posmax > width
        bmask &= posmax < nbypix - width
        posmax_valid = posmax[bmask]
        # get the positions in the spatial dimensions of the trace
        posmax_x2d = []
        posmax_y2d = []
        for iwidth in range(-width, width+1):
            # calculate the upper bound
            yvalues = posmax_valid + iwidth
            # push into the y and x indices (for slicing)
            posmax_y2d.append(yvalues)
            posmax_x2d.append(np.arange(len(yvalues)))
        # convert to 2D numpy array of integers
        posmax_y2d = np.array(posmax_y2d).astype(int)
        posmax_x2d = np.array(posmax_x2d).astype(int)
        # get a cut down (for every frame) around the trace
        #   these are straightened to the nearest integer due to the slicing
        #   by posmax_y2d and posmax_x2d (between -width and +width around the
        #   central trace position
        model_cut = np.array(model[:, posmax_y2d, posmax_x2d])
        residual_cut = np.array(residual[:, posmax_y2d, posmax_x2d])
        err_cut = np.array(err[:, posmax_y2d, posmax_x2d])
        nans = ~np.isfinite(residual_cut+err_cut)
        residual_cut[nans] = 0
        err_cut[nans] = np.inf
        # get the size in the spatial dimension of the spectrum
        ysize = posmax_y2d.shape[0]
        # loop through observations and spectral bins
        for iframe in tqdm(range(nbframes)):
            # get the model, residual and uncertainy for this frame
            tmp_model = np.array(model_cut[iframe])
            tmp_residual = np.array(residual_cut[iframe])
            tmp_err = np.array(err_cut[iframe])
            # zero the arrays
            mean1 = np.zeros(tmp_model.shape[1])
            sig1 = np.zeros(tmp_model.shape[1])
            # calculate the ratio and respective ratio
            ratio1 = tmp_residual / tmp_model
            noise_ratio1 = tmp_err / tmp_model
            # probability that any given pixel is consistent with uncertainties
            p_valid = np.ones_like(tmp_model)
            # place holder for the change between steps in likelihood
            max_dp_valid = 1
            # counter for how many iterations we have done
            counter = 0
            # iterate until we get a dp below 1e-4
            while max_dp_valid > 1e-4 and counter < 10:
                with warnings.catch_warnings(record=True) as _:
                    # calculate the weight
                    weight = p_valid / noise_ratio1 ** 2
                    # get the weighted mean per column
                    denom = np.nansum(weight, axis=0)
                    mean1 = np.nansum(ratio1 * weight, axis=0) / denom
                    # convert mean1 into a 2D array of xsize x ysize
                    model2 = np.tile(mean1, (ysize, 1)) * tmp_model
                    nsig = (tmp_residual - model2) / tmp_err
                    p1 = ne.evaluate('exp(-0.5*nsig**2)')
                    p2 = 1e-4
                    p_valid_prev = p_valid.copy()
                    p_valid = ne.evaluate('p1/(p1+p2)')
                    dp_valid = np.abs(p_valid - p_valid_prev)
                    max_dp_valid = np.max(dp_valid)
                    sigsum = np.sum(p_valid / noise_ratio1 ** 2, axis=0)
                    sig1 = np.sqrt(1 / sigsum)
            # deal with no convergence
            if counter >= 10:
                emsg = 'ratio_residual_to_trace did not converge'
                raise exceptions.SossisseException(emsg)
            # update the cube
            spec[iframe, bmask] = mean1
            spec_err[iframe, bmask] = sig1
        # ---------------------------------------------------------------------
        # return the spectrum and corresponding error
        return spec, spec_err

    def remove_trend_spec(self, spec: np.ndarray) -> np.ndarray:
        """
        Remove the out-of-transit trend from the spectrum and the linear fit
        table

        :param spec: np.ndarray, the spectrum

        :return: np.ndarray, the updated spectrum
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.remove_trend_spec()'
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # get the polynomial degree for trace baseline
        polydeg = self.params['WLC.GENERAL.TRACE_BASELINE_POLYORD']
        # get the out-of-transit domain
        self.get_baseline_params()
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # loop around
        for ix in range(nbxpix):
            # get the slide without the transit
            v1 = spec[baseline_domain, ix]
            # find valid pixels
            valid = np.isfinite(v1)
            # if we don't have enough good pixels skip removing trend for this
            #  row
            if np.sum(valid) < 2:
                continue
            # get the valid xpix for this row
            index = np.arange(nbframes)[baseline_domain]
            # fit the trend
            tfit = np.polyfit(index[valid], v1[valid], polydeg)
            # remove the trend and update the spectrum
            spec[:, ix] -= np.polyval(tfit, np.arange(nbframes))
        # ---------------------------------------------------------------------
        # return the updated spec and ltable
        return spec

    def remove_trend_phot(self, spec: np.ndarray, ltable: Table) -> Table:
        """
        Remove the out-of-transit trend from the spectrum and the linear fit
        table

        :param spec: np.ndarray, the spectrum
        :param ltable: Table, the linear fit table
        :return:
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.remove_trend_phot()'
        # get the out-of-transit domain
        self.get_baseline_params()
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # ---------------------------------------------------------------------
        # do the same for the photometric time series
        # ---------------------------------------------------------------------
        # to the same as we did on the spectrum on the photometric time series
        v1 = np.array(ltable['amplitude'])
        # get an index array for v1
        index = np.arange(spec.shape[0])
        # fit the out-of-transit trend
        tfit = np.polyfit(index[baseline_domain], v1[baseline_domain], 1)
        # remove this off the amplitudes
        ltable['amplitude'] /= np.polyval(tfit, index)
        # ---------------------------------------------------------------------
        # return the updated spec and ltable
        return ltable

    def save_spe_results(self, storage: Dict[str, Any], trace_order: int):
        """
        Save the results to disk
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.save_results()'
        # deal with not saving results --> return
        if not self.params['GENERAL.SAVE_RESULTS']:
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
        spec2 = storage['spec2']
        wavegrid_2d = storage['wavegrid_2d']
        int_times = storage['ltable']['bjd']
        # spec_in = storage['spec_in']
        # spec_err_in = storage['spec_err_in']
        # transit_depth = storage['transit_depth']
        # wave_bin = storage['wave_bin']
        # flux_bin = storage['flux_bin']
        # flux_bin_err = storage['flux_bin_err']
        # ---------------------------------------------------------------------
        # Save the SED table
        # ---------------------------------------------------------------------
        # get objname
        objname = self.params['INPUTS.OBJECTNAME']
        # construct sed table name
        sed_table_name = self.get_variable('SPE_SED_TBL', func_name)
        sed_table_name = sed_table_name.format(objname=objname,
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
                     datanames=['SPECTRUM', 'SPECTRUM_ERROR'],
                     meta=meta_data)
        # ---------------------------------------------------------------------
        # Save the residual grey order file (from amplitudes
        # ---------------------------------------------------------------------
        # get file name
        res_grey_ord = self.get_variable('RES_GREY_ORD', func_name)
        res_grey_ord = res_grey_ord.format(trace_order=trace_order)
        # write the residual grey order tile
        io.save_fits(res_grey_ord, datalist=[spec2, spec_err],
                     datatypes=['image', 'image'],
                     datanames=['SPECTRUM', 'SPECTRUM_ERROR'],
                     meta=meta_data)
        # ---------------------------------------------------------------------
        # Save the spectra for this trace order to file
        # ---------------------------------------------------------------------
        # get file name
        spec_file = self.get_variable('SPECTRA_ORD', func_name)
        spec_file = spec_file.format(trace_order=trace_order)
        # get the data list
        datalist = [wavegrid_2d, spec2, spec_err, int_times]
        # get the data type list
        datatypes = ['image', 'image', 'image', 'image']
        # get the data names
        datanames = ['WAVELENGTH', 'RELFLUX', 'RELFLUX_ERROR', 'BJD']
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
        # # get file name
        # transit_file = self.get_variable('TSPEC_ORD', func_name)
        # transit_file = transit_file.format(trace_order=trace_order)

        # # make transit table
        # t_table = Table()
        # t_table['wavelength'] = wavegrid
        # t_table['flux'] = (spec_in + transit_depth) * 1e6
        # t_table['flux_err'] = spec_err_in * 1e6
        # # save table
        # io.save_table(transit_file, t_table, fmt='csv')

        # # ---------------------------------------------------------------------
        # # Save the transit spectrum for this trace order (binned) to file
        # # ---------------------------------------------------------------------
        # # get resolution_bin
        # res_bin = self.params['SPEC_EXT.RESOLUTION_BIN']
        # # get file name
        # transit_bin_file = self.get_variable('TSPEC_ORD_BIN', func_name)
        # transit_bin_file = transit_bin_file.format(trace_order=trace_order,
        #                                            res=res_bin)
        # # make transit table
        # tb_table = Table()
        # tb_table['wavelength'] = wave_bin
        # tb_table['flux'] = (flux_bin + transit_depth) * 1e6
        # tb_table['flux_err'] = flux_bin_err * 1e6
        # # save table
        # io.save_table(transit_bin_file, tb_table, fmt='csv')

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
            flux = np.array(storage[trace_order]['spec2'])
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
            time_arr = self.get_int_times()
            # -----------------------------------------------------------------
            # get the filename
            filename = self.get_variable('EUREKA_FILE', func_name)
            filename = filename.format(trace_order=trace_order)
            # -----------------------------------------------------------------
            # save eureka format file
            io.save_eureka(filename, flux, flux_err, wavegrid, time_arr)


    def save_final_outputs(self, storage: Dict[int, Dict[str, Any]]):
        """
        Save the final outputs to disk

        Currently these are:
        - OUT_SPEC_LC_FILE: spectral light curve fits file
        - OUT_WLC_FILE: white light curve fits file
        - OUT_TEX_FILE: latex summary file (.tex file)

        :param storage: dict, storage dictionary
        :return: None
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.save_final_outputs()'
        # print progress
        msg = 'Saving final outputs'
        misc.printc(msg, 'info')
        # get output directory
        output_dir = self.params['PATHS.OUT_PATH']
        # check output directory
        if not os.path.exists(output_dir):
            msg = f'Output directory {output_dir} does not exist'
            raise exceptions.SossisseException(msg)
        elif 'ipykernel' in sys.modules:
            msg = (f'For final outputs please run from the command line with '
                   f'your yaml file.')
            misc.printc(msg, 'warning')
            return
        # ---------------------------------------------------------------------
        # update the meta data
        self.update_meta_data()
        # get the meta data
        meta_data = self.get_variable('META', func_name)
        # ---------------------------------------------------------------------
        # generate the param snapshot table
        psnapshot = self.params.snapshot_table(include_zero_counts=True)
        # Producing spectroscopy light curves
        self.out_spec_lc_file(storage, meta_data, psnapshot)
        # Producing white light curve
        self.out_wlc_file(storage, meta_data, psnapshot)
        # Producing output.tex
        self.out_tex_file()

    def out_spec_lc_file(self, storage: Dict[int, Dict[str, Any]],
                         meta_data: Dict[str, Any],
                         psnapshot: Table):
        """
        Produce the spectral light curve fits file with the 2D spectra for
        each trace order and the param snapshot table

        :param storage: dict, storage dictionary
        :param meta_data: dict, meta data dictionary
        :param psnapshot: Table, param snapshot table

        :return: None, saves fits file to OUT_SPEC_LC_FILE
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.out_spec_lc_file()'
        # get the trace orders from the storage keys
        trace_orders = list(storage.keys())
        # storage for fits data writing
        datalist = []
        datatypes = []
        datanames = []
        # loop around trace ordesr
        for trace_order in trace_orders:
            # get the wavelength from storage for this trace order
            wavegrid = storage[trace_order]['wavegrid_2d']
            # get the 2D spectrum from storage for this trace order
            spec2 = storage[trace_order]['spec2']
            # get the 2D spectrum error from storage for this trace order
            spec_err = storage[trace_order]['spec_err']
            # get the integration times from storage for each integration
            int_times = storage[trace_order]['ltable']['bjd']
            # push into data list
            datalist.extend([wavegrid, spec2, spec_err, int_times])
            datatypes.extend(['image', 'image', 'image', 'image'])
            datanames.extend([f'WAVELENGTH_{trace_order}',
                              f'RELFLUX_{trace_order}',
                              f'RELFLUX_ERROR_{trace_order}',
                              f'BJD_{trace_order}'])
        # add the param table
        datalist.append(psnapshot)
        datatypes.append('table')
        datanames.append('PARAMS')
        # construct the fits filename
        filename = self.get_variable('OUT_SPEC_LC_FILE', func_name)
        filename = filename.format(prefix=self.params['GENERAL']['PREFIX'])
        # save the fits file
        io.save_fits(filename, datalist=datalist, datatypes=datatypes,
                     datanames=datanames, meta=meta_data)

    def out_wlc_file(self, storage: Dict[int, Dict[str, Any]],
                     meta_data: Dict[str, Any],
                     psnapshot: Table):
        """
        Produce the white light curve fits file with the linear fit table
        and the param snapshot table

        :param storage: dict, storage dictionary
        :param meta_data: dict, meta data dictionary
        :param psnapshot: Table, param snapshot table

        :return: None, saves fits file to OUT_WLC_FILE
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.out_wlc_file()'
        # get the trace orders from the storage keys
        trace_orders = list(storage.keys())
        # storage for fits data writing
        datalist = []
        datatypes = []
        datanames = []
        # get the linear fit table from storage for this trace order
        #   note the linear fit table is the same for all orders used
        ltable = storage[trace_orders[0]]['ltable']
        # get columns names to keep
        rtable = Table()
        rtable['Time'] = ltable['bjd']
        rtable['RelFlux'] = ltable['amplitude']
        rtable['RelFlux_Err'] = ltable['amplitude_error']
        # push into data list
        datalist.append(rtable)
        datatypes.append('table')
        datanames.append(f'WLC_TABLE')
        # add the param table
        datalist.append(psnapshot)
        datatypes.append('table')
        datanames.append('PARAMS')
        # construct the fits filename
        filename = self.get_variable('OUT_WLC_FILE', func_name)
        filename = filename.format(prefix=self.params['GENERAL']['PREFIX'])
        # Add the pogos modes
        meta_data['POGOS_EM'] = ('SOSSISSE', 'POGOS extraction module')
        meta_data['POGOS_V'] = ('N/A', 'POGOS Version')
        meta_data['POGOS_EV'] = (base.__version__, 'POGOS extraction version')
        # save the fits file
        io.save_fits(filename, datalist=datalist, datatypes=datatypes,
                     datanames=datanames, meta=meta_data)

    def out_tex_file(self):
        """
        Use ParamDicts build in latex table functionality to produce
        a latex table of the parameters used in this run

        :return: None, saves a tex table to OUT_TEX_FILE
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.out_tex_file()'
        # construct the fits filename
        filename = self.get_variable('OUT_TEX_FILE', func_name)
        filename = filename.format(prefix=self.params['GENERAL']['PREFIX'])
        # get tex file snapshot table
        tsnapshot = self.params.out_tex_file()
        # write to latex table
        io.save_table(filename, tsnapshot, fmt='ascii.latex',
                      latexdict=LATEX_DICT)

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
