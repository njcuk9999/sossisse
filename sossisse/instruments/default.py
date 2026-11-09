#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import copy
import hashlib
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing import cast

import numexpr as ne
import numpy as np
from numpy.typing import NDArray
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
from sossisse.core import constants
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.core import misc
from sossisse.general.datacube import DataCube
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
        # jump processing context.  Empty suffix/slice is the normal path.
        # This suffix is appended to temporary and output filenames per chunk.
        self._jump_suffix = ''
        # Detector children append this suffix to every generated filename.
        self._trace_suffix = ''
        # This slice defines the active global integration range for a chunk.
        self._jump_slice = None
        # This string is written to FITS headers so products record jump use.
        self._jump_ints_header = 'No jumps'
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
        output_meta_func = f'{self.name}.set_linear_output_metadata()'
        self.vsources['OUTPUT_NAMES'] = output_meta_func
        self.vsources['OUTPUT_UNITS'] = output_meta_func
        self.vsources['OUTPUT_FACTOR'] = output_meta_func
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
        do_background = self.params.rget('GENERAL.DO_BACKGROUND', required=True)
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
        if do_background:
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
        # Record which jumps were used in every FITS product header.
        meta_data['JUMPINTS'] = (self._jump_ints_header,
                                 'SOSSISSE jump integrations')
        # Cache the metadata for the save routines.
        self.set_variable('META', meta_data)

    def set_jump_context(self, start: Optional[int] = None,
                         stop: Optional[int] = None, suffix: str = '',
                         jump_ints: Optional[List[int]] = None):
        """
        Set the current jump-processing context for filenames and time slicing.

        :param start: int, first integration to include
        :param stop: int, first integration after this chunk
        :param suffix: str, suffix to add to output filenames
        :param jump_ints: list, all user-supplied jump integrations
        :return: None
        """
        # set the filename suffix
        # Empty suffix means normal no-jump product names.
        self._jump_suffix = suffix
        # set the time slice
        # A missing start or stop disables time slicing.
        if start is None or stop is None:
            self._jump_slice = None
        else:
            # Store stop as exclusive, matching Python slicing rules.
            self._jump_slice = (int(start), int(stop))
        # update the header representation if supplied
        # jump_ints is None when only the suffix/slice is being changed.
        if jump_ints is not None:
            # Keep the no-jump header human-readable.
            if len(jump_ints) == 0:
                self._jump_ints_header = 'No jumps'
            else:
                # Store jump integrations in a compact FITS-header string.
                self._jump_ints_header = ','.join(map(str, jump_ints))
        # cached baseline/meta values depend on the active chunk
        # Clear these so they are recomputed for the new chunk context.
        for key in ['HAS_BASELINE', 'BASELINE_INTS', 'BASELINE_DOMAIN', 'META']:
            self._variables[key] = None

    def add_jump_suffix(self, filename: str) -> str:
        """
        Add the active jump suffix to a filename.

        :param filename: str, input filename
        :return: str, filename with active suffix before the extension
        """
        # Do not change filenames when no jump chunk is active.
        if self._jump_suffix == '':
            return filename
        # Split the extension so residual.fits becomes residual_jump000.fits.
        root, ext = os.path.splitext(filename)
        # Rebuild the filename with the active suffix inserted before ext.
        return root + self._jump_suffix + ext

    def add_trace_suffix(self, filename: str) -> str:
        """Append this instrument context's trace label before the extension.

        :param filename: str, generated filename or path
        :return: str, filename with the active trace suffix
        """
        # Parent instruments and non-multi-trace instruments keep old names.
        if self._trace_suffix == '':
            return filename
        # Keep the extension last so FITS and plot tools still recognize it.
        root, ext = os.path.splitext(filename)
        # Add the detector/order label to distinguish shared-directory files.
        return root + self._trace_suffix + ext

    def add_product_suffix(self, filename: str) -> str:
        """Append trace and jump labels to a generated product filename.

        :param filename: str, generated filename or path
        :return: str, filename with active trace and jump suffixes
        """
        # Add the trace first so jump chunks follow the detector/order label.
        trace_filename = self.add_trace_suffix(filename)
        # Preserve the established jump suffix behavior after trace labeling.
        return self.add_jump_suffix(trace_filename)

    def slice_jump_data(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Slice time-dependent arrays to the active jump context.

        :param arrays: np.ndarray, arrays to slice along axis 0
        :return: tuple, sliced arrays
        """
        # In the normal path, leave every array unchanged.
        if self._jump_slice is None:
            return arrays
        # Unpack the active global integration range.
        start, stop = self._jump_slice
        # Slice every supplied array on axis 0 to keep time products aligned.
        return tuple(array[start:stop] for array in arrays)

    def get_input_n_frames(self) -> int:
        """
        Count the number of binned integrations in the input data.

        :return: int, number of frames
        """
        input_files = self.params.rget('GENERAL.FILES', required=True)
        bin_size = self.get_data_bin_size()
        try:
            return DataCube.input_n_integrations(input_files, bin_size)
        except ValueError as error:
            raise exceptions.SossisseFileException(str(error)) from error

    def get_data_bin_size(self) -> int:
        """Return the active temporal bin size, or one when disabled."""
        lm_params = self.params.get('WLC.LMODEL')
        bin_time = lm_params.rget('DATA_BIN_TIME', required=True)
        if not bin_time:
            return 1
        return lm_params.rget('DATA_BIN_SIZE', required=True)

    def get_datacube_options(self) -> dict:
        """Return instrument-specific options for raw DataCube reads."""
        return dict(from_cds=self.from_cds,
                    conversion_signature=self.get_cds_signature)

    def get_cds_signature(self):
        """Identify reuse settings only when raw inputs require CDS."""
        input_params = self.params['WLC.INPUTS']
        cds_ids = input_params['CDS_IDS']
        cds_ron = input_params['CDS_RON']
        return [cds_ids, cds_ron]

    def localize_frame_domains(self, raw_domains: Union[None, List],
                               param_name: str) -> List[List[int]]:
        """
        Convert global frame domains into active-chunk frame domains.

        :param raw_domains: list, input frame domains
        :param param_name: str, parameter name for errors
        :return: list, chunk-local frame domains
        """
        # Missing domains mean there is nothing to localize.
        if raw_domains is None:
            return []
        # Domains must arrive as a list so we can parse start/end pairs.
        if not isinstance(raw_domains, list):
            emsg = f'{param_name} must be a list of frame domains'
            raise exceptions.SossisseConstantException(emsg)
        # Empty lists are allowed and simply have no overlap.
        if len(raw_domains) == 0:
            return []
        # Preferred yaml shape is already a list of [start, end] rows.
        if isinstance(raw_domains[0], list):
            domains = raw_domains
        else:
            # Flat lists must have an even number of start/end entries.
            if len(raw_domains) % 2 != 0:
                emsg = f'{param_name} must contain start/end pairs'
                raise exceptions.SossisseConstantException(emsg)
            # Build the normalized list-of-lists representation.
            domains = []
            # Walk over each start/end pair in the flat list.
            for it in range(len(raw_domains) // 2):
                # Store this pair as [start, end].
                domains.append([raw_domains[2 * it], raw_domains[2 * it + 1]])
        # Store domains after clipping them to the active chunk.
        localized = []
        # Validate and localize each input domain independently.
        for row, domain in enumerate(domains):
            # Each domain must be exactly a start and an end value.
            if len(domain) != 2:
                emsg = f'{param_name}[{row}] must be a list of length 2'
                raise exceptions.SossisseConstantException(emsg)
            # Convert yaml values to integers before comparing or slicing.
            try:
                start = int(domain[0])
                end = int(domain[1])
            # Raise a parameter error if either value is not integer-like.
            except Exception as _:
                emsg = f'{param_name}[{row}] must contain integers'
                raise exceptions.SossisseConstantException(emsg)
            # Accept reversed ranges by sorting the endpoints.
            if start > end:
                start, end = end, start
            # When chunking, convert global frame numbers into chunk-local ones.
            if self._jump_slice is not None:
                # Get the active chunk range in global frame coordinates.
                chunk_start, chunk_stop = self._jump_slice
                # Clip the domain start to the first frame in this chunk.
                start = max(start, chunk_start)
                # Clip the domain end to the last included frame in this chunk.
                end = min(end, chunk_stop - 1)
                # Skip domains that do not intersect the active chunk.
                if start > end:
                    continue
                # Translate from global frame numbers to local chunk numbers.
                start -= chunk_start
                # Translate the domain end by the same chunk offset.
                end -= chunk_start
            # Keep this localized domain for baseline or reject-domain logic.
            localized.append([start, end])
        # Return domains in the shape expected by process_baseline_ints.
        return localized

    def define_filenames(self):
        # set function name
        func_name = f'{__NAME__}.define_filenames()'
        # get file paths
        temppath = self.params.rget('PATHS.TEMP_PATH', required=True, func=func_name)
        otherpath = self.params.rget('PATHS.OTHER_PATH', required=True, func=func_name)
        outpath = self.params.rget('PATHS.OUT_PATH', required=True, func=func_name)
        # update meta data
        self.update_meta_data()
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
        median_image_file = os.path.join(temppath, median_image_file)
        # ---------------------------------------------------------------------
        errfile = os.path.join(temppath, 'errormap.fits')
        # ---------------------------------------------------------------------
        resfile = os.path.join(temppath, 'residual.fits')
        # ---------------------------------------------------------------------
        reconfile = os.path.join(temppath, 'recon.fits')
        # ---------------------------------------------------------------------
        ltbl_file = os.path.join(otherpath, 'linear_recon_coeffs.csv')
        # ---------------------------------------------------------------------
        sed_table = os.path.join(otherpath, 'sed_{objname}_ord{trace}.csv')
        # ---------------------------------------------------------------------
        res_no_grey_ord = 'residual_no_grey_ord{trace}.fits'
        res_no_grey_ord = os.path.join(temppath, res_no_grey_ord)
        # ---------------------------------------------------------------------
        res_grey_ord = 'residual_grey_ord{trace}.fits'
        res_grey_ord = os.path.join(temppath, res_grey_ord)
        # ---------------------------------------------------------------------
        spectra_ord = 'spectra_ord{trace}.fits'
        spectra_ord = os.path.join(temppath, spectra_ord)
        # ---------------------------------------------------------------------
        waveord_file = 'wavelength_ord{trace}.fits'
        waveord_file = os.path.join(temppath, waveord_file)
        # ---------------------------------------------------------------------
        tspec_ord = 'tspec_ord{trace}.csv'
        tspec_ord = os.path.join(otherpath, tspec_ord)
        # ---------------------------------------------------------------------
        tspec_ord_bin = 'tspec_ord_{trace}_bin{res}.csv'
        tspec_ord_bin = os.path.join(otherpath, tspec_ord_bin)
        # ---------------------------------------------------------------------
        eureka_file = 'spectra_ord{trace}.h5'
        eureka_file = os.path.join(temppath, eureka_file)
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
        # add active jump suffixes to all products produced in this context
        tmp_amp_file = self.add_product_suffix(tmp_amp_file)
        clean_cube_file = self.add_product_suffix(clean_cube_file)
        tmp_pcas = self.add_product_suffix(tmp_pcas)
        temp_clean_nan = self.add_product_suffix(temp_clean_nan)
        temp_clean_nan_err = self.add_product_suffix(temp_clean_nan_err)
        temp_ini_cube = self.add_product_suffix(temp_ini_cube)
        temp_ini_err = self.add_product_suffix(temp_ini_err)
        temp_ini_dq = self.add_product_suffix(temp_ini_dq)
        temp_ff_cube = self.add_product_suffix(temp_ff_cube)
        temp_ff_err = self.add_product_suffix(temp_ff_err)
        temp_ff_dq = self.add_product_suffix(temp_ff_dq)
        temp_cube_post_cosmics = self.add_product_suffix(
            temp_cube_post_cosmics)
        tmp_ini_cube_bkgrnd = self.add_product_suffix(tmp_ini_cube_bkgrnd)
        tmp_ini_err_bkgrnd = self.add_product_suffix(tmp_ini_err_bkgrnd)
        tmp_ini_cube_lowpass = self.add_product_suffix(tmp_ini_cube_lowpass)
        tmp_ini_err_lowpass = self.add_product_suffix(tmp_ini_err_lowpass)
        median_image_file = self.add_product_suffix(median_image_file)
        errfile = self.add_product_suffix(errfile)
        resfile = self.add_product_suffix(resfile)
        reconfile = self.add_product_suffix(reconfile)
        ltbl_file = self.add_product_suffix(ltbl_file)
        sed_table = self.add_product_suffix(sed_table)
        res_no_grey_ord = self.add_product_suffix(res_no_grey_ord)
        res_grey_ord = self.add_product_suffix(res_grey_ord)
        spectra_ord = self.add_product_suffix(spectra_ord)
        waveord_file = self.add_product_suffix(waveord_file)
        tspec_ord = self.add_product_suffix(tspec_ord)
        tspec_ord_bin = self.add_product_suffix(tspec_ord_bin)
        eureka_file = self.add_product_suffix(eureka_file)
        out_spec_lc_file = self.add_product_suffix(out_spec_lc_file)
        out_wlc_file = self.add_product_suffix(out_wlc_file)
        out_tex_file = self.add_product_suffix(out_tex_file)
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
    def load_data(self, filename: str, ext: int = None, extname: str = None,
                  force_pixel_grid: bool = False):
        """
        Load the data from a file

        :param filename: str, the filename to load
        :param ext: int, the extension number to load
        :param extname: str, the extension name to load
        :param force_pixel_grid: bool, ignored for the default instrument

        :return: data, the loaded data
        """
        _ = self
        _ = force_pixel_grid
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
                  flag_cds: bool, force_pixel_grid: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO(DataCube): Legacy full-array loader; load_data_with_dq now uses
        # normalized products. Retire this API after its callers are migrated.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        _ = force_pixel_grid
        input_files = self.params.rget('GENERAL.FILES', required=True)
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
        for ifile, filename in enumerate(input_files):
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
            loaded_extensions = [1]
            if not flag_cds:
                loaded_extensions.extend([2, 3])
            io.log_fits_load(filename, ext=loaded_extensions)
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
        input_files = self.params.rget('GENERAL.FILES', required=True)
        int_times = np.array([])
        for ifile, filename in enumerate(input_files):
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
        # slice to the active jump chunk if requested
        if self._jump_slice is not None:
            int_times = self.slice_jump_data(int_times)[0]
        return int_times

    def load_data_with_dq(self) -> DataCube:
        """Normalize raw segments into bounded SCI/ERR/DQ working products.

        Old combined primary-image caches are deliberately ignored. Only
        complete normalized products matching source/settings may be reused.

        :return: DataCube manager with header-sorted acquisition metadata
        """
        func_name = f'{__NAME__}.load_data_with_dq()'
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        max_integrations = self.params.rget('GENERAL.MAX_INTEGRATIONS',
                                            required=True, func=func_name)
        if (isinstance(max_integrations, bool)
                or not isinstance(max_integrations, (int, np.integer))
                or max_integrations < 1):
            emsg = 'GENERAL.MAX_INTEGRATIONS must be a positive integer'
            raise exceptions.SossisseConstantException(emsg)
        max_integrations = int(max_integrations)
        # Validate before DataCube construction or any raw pixel reads.
        memory_value = self.params.rget('GENERAL.MAX_MATH_MEMORY_GB',
                        required=True, func=func_name)
        max_math_memory_gb = constants.MathMemoryConst.memory(memory_value)
        temp_ini_cube = self.get_variable('TEMP_INI_CUBE', func_name)
        input_files = self.params.rget('GENERAL.FILES', required=True,
                                       func=func_name)
        if len(input_files) == 0:
            emsg = 'No files defined in yaml. Please set the FILES parameter.'
            raise exceptions.SossisseConstantException(emsg)
        try:
            datacube = DataCube(max_integrations=max_integrations,
                               max_math_memory_gb=max_math_memory_gb)
            normalize_kwargs = self.get_datacube_options()
            normalize_kwargs['bin_size'] = self.get_data_bin_size()
            normalize_kwargs['integration_slice'] = self._jump_slice
            normalize_kwargs['reuse'] = use_temp
            datacube.normalize(input_files, temp_ini_cube, **normalize_kwargs)
        except ValueError as error:
            raise exceptions.SossisseFileException(str(error)) from error
        self.params.set('GENERAL.FILES', list(datacube.filenames),
                        source=func_name)
        self.set_variable('FLAG_CDS', datacube.flag_cds)
        self.update_meta_data()
        self.set_variable('DATA_X_SIZE', datacube.shape[2])
        self.set_variable('DATA_Y_SIZE', datacube.shape[1])
        self.set_variable('DATA_N_FRAMES', datacube.shape[0])
        return datacube

    def apply_flat_field(self, datacube: DataCube) -> DataCube:
        """Persist flat-corrected SCI/ERR blocks, retaining DQ unchanged.

        Use a bounded first pass for the observation-wide nonzero error
        minimum, then replace zero errors and write in a second pass. Commit
        the next filenames only after every expected output validates.

        :param datacube: manager of the current aligned SCI/ERR/DQ FITS blocks
        :return: the same manager, advanced to the successful flat stage
        """
        # ---------------------------------------------------------------------
        # Resolve the calibration and identify the expected next products.
        # ---------------------------------------------------------------------
        func_name = f'{__NAME__}.apply_flat_field()'
        # Working products are always written. This flag only allows reuse of
        # a complete, validated flat stage from an earlier run.
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        # The flat is one 2D detector image shared by every integration.
        # get_flat also reports whether flat correction should be skipped.
        flat, no_flat = self.get_flat(list(datacube.shape[1:]))
        if no_flat:
            # Keep the existing input filenames active when no flat applies.
            return datacube
        # Floating-point arithmetic preserves NaNs in invalid flat pixels.
        flat = np.asarray(flat, dtype=float)
        # Check alignment before reading blocks or creating any output files.
        if flat.shape != datacube.shape[1:]:
            raise ValueError('Flat shape does not match DataCube')
        # Include the actual calibration values in the stage identity so a
        # changed flat cannot silently reuse previously corrected products.
        signature = hashlib.sha256(flat.tobytes()).hexdigest()
        # Plan filenames/provenance without changing current_infiles.
        products = datacube.plan_stage('flat', signature)
        # Retain just the first uncorrected science frame for the comparison
        # plot; the full observation is never needed for plotting this step.
        frame0 = datacube.read_integrations(0, 1)['FLUX'][0].copy()
        if use_temp and datacube.products_valid(products):
            # Reuse requires every planned output to match its expected layout
            # and provenance, not merely for some output filenames to exist.
            plots.plot_flat_field(self, frame0, frame0 / flat)
            # Publish the already validated flat products as the next inputs.
            datacube.commit_products(products)
            return datacube

        # ---------------------------------------------------------------------
        # First pass: find one error replacement value for the whole selection.
        # ---------------------------------------------------------------------
        misc.printc('Applying flat field to data', 'info')
        # Starting with NaN lets the first finite candidate establish the min.
        minimum = np.nan
        # Distinguish an entirely zero error dataset from NaN-containing data.
        has_nonzero = False
        for _, arrays in datacube.iter_integrations():
            # Apply the flat temporarily to this block's ERR array
            # so the minimum uses corrected uncertainties, not raw ones.
            corrected = arrays['ERR'] / flat
            # Exclude exact zeros: these are the values we will replace later.
            nonzero = corrected[corrected != 0]
            if nonzero.size:
                has_nonzero = True
                # Preserve the original NaN-tolerant behavior without warnings
                # from a block whose nonzero candidates are all NaN.
                with warnings.catch_warnings(record=True):
                    # Combine block minima without collecting errors in RAM.
                    minimum = np.fmin(minimum, np.nanmin(nonzero))
        if not has_nonzero:
            # Do not write a stage when there is no replacement candidate.
            raise ValueError('No nonzero errors for flat zero replacement')

        # ---------------------------------------------------------------------
        # Second pass: correct each block and persist its SCI/ERR/DQ product.
        # ---------------------------------------------------------------------
        # Read the original inputs again now that the global minimum is known.
        # Each planned output corresponds to one existing integration block.
        for product, (_, arrays) in zip(products,
                                         datacube.iter_integrations()):
            # These are independent block arrays, not the manager's storage.
            flux = arrays['FLUX']
            err = arrays['ERR']
            # Apply the same 2D flat to science and uncertainty frame by frame.
            # DQ flags are neither divided nor otherwise modified in this step.
            for iframe in range(flux.shape[0]):
                flux[iframe] /= flat
                err[iframe] /= flat
            # Use the SAME observation-wide minimum in every output block,
            # preserving the previous full-cube zero-error replacement rule.
            err[err == 0] = minimum
            # Write one bounded SCI/ERR/DQ FITS product atomically. Keep
            # current_infiles unchanged so later blocks still read old inputs.
            datacube.write_product(product, arrays)

        # ---------------------------------------------------------------------
        # Finish diagnostics, then advance the manager to the completed stage.
        # ---------------------------------------------------------------------
        plots.plot_flat_field(self, frame0, frame0 / flat)
        # Validate the entire output set before publishing it. A failed stage
        # leaves current_infiles pointing at the previous successful products.
        datacube.commit_products(products)
        # Callers keep the same manager, now reading the flat-corrected files.
        return datacube

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
        inconsistent_shapes = len(set(image_shapes)) != 1
        inconsistent_cds = len(set(all_cds)) != 1
        if inconsistent_shapes or inconsistent_cds:
            input_files = self.params.rget('GENERAL.FILES', required=True)
        if inconsistent_shapes:
            emsg = 'Inconsistent image shapes:'
            for f_it, filename in enumerate(input_files):
                emsg += f'\n\t{filename}: {image_shapes[f_it]}'
            raise exceptions.SossisseFileException(emsg)
        # ---------------------------------------------------------------------
        # check that all are CDS or all are not CDS
        if inconsistent_cds:
            emsg = 'Inconsistent CDS format (Eiher all CDS or not):'
            for f_it, filename in enumerate(input_files):
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
        lm_params = self.params.get('WLC.LMODEL')
        bin_time = lm_params.rget('DATA_BIN_TIME', required=True,
                      func=func_name)
        # don't bin if user doesn't want to bin
        if not bin_time:
            if get_shape:
                return list(cube.shape)
            else:
                return cube
        # don't bin if the number of frames in each bin is 1
        bin_size = lm_params.rget('DATA_BIN_SIZE', required=True,
                      func=func_name)
        if bin_size == 1:
            if get_shape:
                return list(cube.shape)
            else:
                return cube
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

    def remove_cosmic_rays(self, datacube: DataCube) -> DataCube:
        """Mask cosmic rays with an observation-wide percentile sigma cut.

        The per-pixel 16th/84th percentiles of all active integrations use
        exact DataCube bounded math (never partial-block statistics). Each
        frame then sets SCI to NaN where (frame - mid) / half-width exceeds
        WLC.GENERAL.COSMIC_RAY_SIGMA, exactly as the original full-cube loop.
        ERR and DQ are unchanged. Reuse skips writes but recomputes the heat
        map so the diagnostic plot is always produced.

        :param datacube: active normalized SCI/ERR/DQ stage manager
        :return: same manager; advanced only after all writes and the plot
                 succeed, or unchanged when WLC.GENERAL.REMOVE_COSMIC_RAYS is
                 false (no file I/O or further setting reads)
        :raises MemoryError: if the exact percentiles exceed the math budget
        :raises ValueError: if products or alignment fail validation
        :raises OSError: if a working product cannot be read or written
        :raises Exception: propagate plot errors without committing
        """
        # get function name
        func_name = f'{__NAME__}.remove_cosmic_rays()'
        # Read only the enable switch until the stage is actually requested.
        remove_cosmics = self.params.rget('WLC.GENERAL.REMOVE_COSMIC_RAYS',
                                          required=True, func=func_name)
        # see if user wants to remove cosmics
        if not remove_cosmics:
            # print message that we are not removing cosmic rays
            msg = ('WLC.GENERAL.REMOVE_COSMIC_RAYS=False. '
                   'Not removing cosmic rays')
            misc.printc(msg, 'info')
            # Keep the original working products active.
            return datacube
        # Working products are mandatory; USE_TEMPORARY controls reuse only.
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        sig_cut = self.params.rget('WLC.GENERAL.COSMIC_RAY_SIGMA',
                                   required=True, func=func_name)
        # The threshold changes masked pixels, so it identifies products.
        signature = dict(sigma=float(sig_cut), percentiles=[16, 84],
                         algorithm='percentile-sigma-cut-v1')
        products = datacube.plan_stage('cosmics', signature)
        reuse = use_temp and datacube.products_valid(products)
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
            # Tiles hold every integration of a pixel: exact percentiles.
            p16, p84 = datacube.nanpercentile([16, 84], 'FLUX', axis=0)
        # get the sigma and mean
        sigma = (p84 - p16) / 2
        mean = (p84 + p16) / 2
        # storage for heat map
        heat_map = np.zeros(datacube.shape[1:], dtype=float)
        # store the first frame from before and after (for plot)
        frame_before = datacube.read_integrations(0, 1)['FLUX'][0].copy()
        frame_after = frame_before.copy()
        # Only one block and detector-sized maps reside in memory.
        for product, (integration, arrays) in zip(
                products, datacube.iter_integrations()):
            flux = arrays['FLUX']
            # now remove the cosmics using a sigma flag
            for local_index, frame in enumerate(flux):
                # calculate the number of sigma away from the mean every pixel
                nsig = (frame - mean) / sigma
                # cosmic mask
                cmask = nsig > sig_cut
                # add to heat map
                heat_map[cmask] += 1
                # set those above the threshold to nan
                frame[cmask] = np.nan
                # keep the first masked frame for the heat map plot
                if integration.start + local_index == 0:
                    frame_after = frame.copy()
            if not reuse:
                # Never mutate ERR/DQ or publish a partially written stage.
                datacube.write_product(product, arrays)
            del flux, arrays
        if reuse:
            # Plot the validated cached output, not merely its reconstruction.
            with fits.open(products[0].filename, memmap=False) as hdul:
                science_hdu = cast(fits.ImageHDU, hdul['SCI'])
                frame_after = science_hdu.section[0].astype(float)
        # ---------------------------------------------------------------------
        # plot fractions of good pixels (per pixel) as map
        plots.plot_heatmap(self, heat_map, frame_before,
                           frame_after, 'cosmic rays rate', 'cosmic_rays_corr',
                           'Number of cosmic rays')
        # Validate all products again before changing active manager filenames.
        datacube.commit_products(products)
        return datacube

    def apply_dq(self, datacube: DataCube) -> DataCube:
        """Apply one conservative spatial DQ whitelist to all active frames.

        VALID_DQ maps accepted EXACT values (not bits) to maximum occurrence
        fractions. A legacy list uses DQ_THRESHOLD for each accepted value.
        Limits are inclusive: 0 forbids occurrence, 1 permits every frame.
        Fractions use the entire active DataCube (the current jump chunk),
        never an individual block or an unselected part of the observation.
        Any unlisted occurrence, including NaN/inf, or any exceeded limit
        rejects that spatial pixel in EVERY frame, even otherwise valid ones.

        :param datacube: bounded, aligned SCI/ERR/DQ working-product manager
        :return: same manager; rejected SCI becomes NaN and ERR becomes inf,
                 DQ is unchanged. Disabled processing performs no file I/O
                 or further setting reads. Enabled processing writes bounded
                 dq-stage FITS products and publishes only a complete stage.
                 USE_TEMPORARY permits reuse only of an entirely valid set.
        :raises SossisseConstantException: for invalid whitelist or fractions
        :raises ValueError: for invalid or incomplete output products
        :raises OSError: for read/write failures; active inputs stay unchanged
        """
        # Read only the enable switch until this stage is actually requested.
        func_name = f'{__NAME__}.apply_dq()'
        input_params = self.params.get('WLC.INPUTS')
        apply_dq_flags = input_params.rget('APPLY_DQ_FLAGS', required=True,
                                           func=func_name)
        if not apply_dq_flags:
            msg = 'WLC.INPUTS.APPLY_DQ_FLAGS = False. Not applying DQ flags'
            misc.printc(msg, 'info')
            return datacube

        # Validate before reading images; ParamDict callers may bypass Const.
        valid_dq = input_params.rget('VALID_DQ', required=True, func=func_name)
        valid_dq = constants.ValidDQConst.configuration(valid_dq)
        if isinstance(valid_dq, list):
            # Mappings override the fallback, so only lists need this setting.
            threshold = input_params.rget('DQ_THRESHOLD', required=True,
                                          func=func_name)
            threshold = constants.DQThresholdConst.fraction(threshold)
            valid_dq = dict.fromkeys(valid_dq, threshold)
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        # Canonical ordering makes equivalent YAML mappings share products.
        # The rule version prevents reuse of older per-integration semantics.
        signature = dict(rule='exact-whitelist-shared-inclusive-v1',
                         limits=sorted(valid_dq.items()))
        products = datacube.plan_stage('dq', signature)
        if use_temp and datacube.products_valid(products):
            # A complete hit needs no science/DQ block reads or recounting.
            datacube.commit_products(products)
            return datacube

        # ---------------------------------------------------------------------
        # First pass: count each accepted exact category across the selection.
        # ---------------------------------------------------------------------
        # Retain N int64 detector maps, never a cube-wide Boolean mask.
        dq_count: dict[int, NDArray[np.int64]] = dict()
        for value in valid_dq:
            dq_count[value] = np.zeros(datacube.shape[1:], dtype=np.int64)
        for _, arrays in datacube.iter_integrations():
            # Equality excludes NaN/inf and does not interpret combined bits.
            quality = arrays['DQ']
            for value, count in dq_count.items():
                count += np.count_nonzero(quality == value, axis=0)
            # Do not keep the last image block alive during the second pass.
            del quality, arrays

        # One shared 2D mask applies to ALL integrations, not just bad samples.
        cube_mask = np.ones(datacube.shape[1:], dtype=bool)
        for value, count in dq_count.items():
            # Inclusive comparison preserves exact boundary acceptance.
            cube_mask &= count <= valid_dq[value] * len(datacube)
        # Each sample matches at most one exact category. Sum counts in place
        # after testing limits, avoiding another detector-sized count map.
        counts = iter(dq_count.values())
        accepted_count = next(counts)
        for count in counts:
            accepted_count += count
        # Any unmatched sample rejects the entire pixel, including NaN/inf.
        cube_mask &= accepted_count == len(datacube)

        # ---------------------------------------------------------------------
        # Second pass: mask science/errors and persist unchanged DQ per block.
        # ---------------------------------------------------------------------
        for product, (_, arrays) in zip(products,
                                         datacube.iter_integrations()):
            flux = arrays['FLUX']
            err = arrays['ERR']
            flux[:, ~cube_mask] = np.nan
            err[:, ~cube_mask] = np.inf
            # Atomic block writes do not publish partial stage filenames.
            datacube.write_product(product, arrays)
            del flux, err, arrays
        # Failed counting/writes/validation leave the original stage active.
        datacube.commit_products(products)
        return datacube

    def remove_background(self, datacube: DataCube) -> DataCube:
        """Fit one observation-wide background and persist corrected blocks.

        Preserve the shifted calibration, amplitude and DC-offset fit using
        a per-pixel NaN-aware mean of the active integrations. Only bounded
        blocks and detector-sized maps reside in memory. ERR and DQ are
        unchanged. Disabled correction reads only the enable parameter.

        Enabled correction updates instrument metadata before loading the
        GENERAL.BKGFILE calibration. USE_TEMPORARY permits reuse only when
        calibration contents, fit settings and all parent products match.
        Plotting and full output validation must succeed before publishing
        new filenames; failures leave the manager's current inputs unchanged.

        :param datacube: manager of aligned float64 FLUX/ERR/DQ FITS blocks
        :return: the same manager, advanced only after a successful stage,
                 or unchanged when GENERAL.DO_BACKGROUND is false
        :raises SossisseConstantException: if fit box or shifts are unset
        :raises ValueError: for mismatched calibration shape, invalid fit
                            inputs or invalid output products
        :raises OSError: if calibration or product I/O fails
        :raises Exception: propagated metadata, fitting or plotting failure
        """
        # set function name
        func_name = f'{__NAME__}.remove_background()'
        # Skip before any calibration, cache or science access.
        do_background = self.params.rget('GENERAL.DO_BACKGROUND',
                                         required=True, func=func_name)
        # deal with not doing background correction
        if not do_background:
            # print progress
            msg = 'We do not clean background. GENERAL.DO_BACKGROUND=False'
            misc.printc(msg, 'warning')
            # Keep the original working products active.
            return datacube
        # update the meta data
        self.update_meta_data()
        # Gather enabled-stage settings after the metadata update.
        general_params = self.params.get('GENERAL')
        input_params = self.params.get('WLC.INPUTS')
        use_temp = general_params.rget('USE_TEMPORARY', required=True,
                                       func=func_name)
        background_file = general_params.rget('BKGFILE', required=True,
                                              func=func_name)
        # ---------------------------------------------------------------------
        # optimal background correction
        # ---------------------------------------------------------------------
        # print progress
        msg = 'Apply the background model correction'
        misc.printc(msg, 'info')
        # get the background file
        background = self.load_data(background_file)
        # force the background to floats
        background = background.astype(float)
        # Reject misalignment before fitting or writing working products.
        if background.shape != datacube.shape[1:]:
            raise ValueError('Background shape does not match DataCube')
        # ---------------------------------------------------------------------
        # get the box size from params
        box = input_params.rget('BACKGROUND_GLITCH_BOX', func=func_name)
        if box is None:
            emsg = ('WLC.INPUTS.BACKGROUND_GLITCH_BOX not set. '
                    'Please set the BACKGROUND_GLITCH_BOX')
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # get the background shifts
        bgnd_shifts_values = input_params.rget('BACKGROUND_SHIFTS',
                              func=func_name)
        if bgnd_shifts_values is None:
            emsg = ('WLC.INPUTS.BACKGROUND_SHIFTS not set. '
                    'Please set the BACKGROUND_SHIFTS')
            raise exceptions.SossisseConstantException(emsg)
        # ---------------------------------------------------------------------
        # Identity includes actual calibration values and effective fit inputs.
        bgnd_shifts = np.arange(*bgnd_shifts_values)
        signature = dict()
        signature['calibration'] = hashlib.sha256(
            background.tobytes()).hexdigest()
        signature['box'] = np.asarray(box).tolist()
        signature['shifts'] = bgnd_shifts.tolist()
        products = datacube.plan_stage('background', signature)
        # Retain only the first original frame for the comparison plot.
        frame0 = datacube.read_integrations(0, 1)['FLUX'][0].copy()
        if use_temp and datacube.products_valid(products):
            # Read one corrected frame, without counting or fitting science.
            with fits.open(products[0].filename, memmap=True) as hdul:
                science_hdu = cast(fits.ImageHDU, hdul['SCI'])
                corrected0 = science_hdu.section[0].astype(float)
            plots.plot_background(self, frame0, corrected0)
            datacube.commit_products(products)
            return datacube

        # Accumulate in integration order with one count for each non-NaN
        # sample. Infinities count, exactly as in the original np.nanmean.
        mean_sum = np.zeros(datacube.shape[1:], dtype=np.float64)
        mean_count = np.zeros(datacube.shape[1:], dtype=np.int64)
        with np.errstate(invalid='ignore', divide='ignore'):
            for _, arrays in datacube.iter_integrations():
                for frame in arrays['FLUX']:
                    valid = ~np.isnan(frame)
                    np.add(mean_sum, frame, out=mean_sum, where=valid)
                    mean_count += valid
                # Do not keep the previous block during the next FITS read.
                del frame, arrays
            mcube = mean_sum / mean_count
        # Only the mean is needed by the original background fit below.
        del mean_sum, mean_count, valid
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
                rms_fit = np.polyfit(bgnd_shifts[bstart:bend],
                                     rms[bstart:bend], 2)
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
        misc.printc('Writing background-corrected working products', 'info')
        # Subtract the same fitted detector image from every integration.
        # ERR and DQ retain their original float64 payloads without changes.
        for product, (_, arrays) in zip(products,
                                        datacube.iter_integrations()):
            for frame in arrays['FLUX']:
                frame -= background
            # Writes remain unpublished until all diagnostics succeed.
            datacube.write_product(product, arrays)
            del frame, arrays
        # ---------------------------------------------------------------------
        # plot the background correction
        plots.plot_background(self, frame0, frame0 - background)
        # Validate the complete output set before advancing current inputs.
        datacube.commit_products(products)
        return datacube


    def low_pass_filter(self, datacube: DataCube) -> DataCube:
        """Persist the legacy column low-pass correction in bounded products.

        Use the exact observation-wide nanmean for the below-column-median
        mask, then subtract each frame's median-filtered width-25 profile.
        ERR and DQ are unchanged. The fixed scientific settings and complete
        parent stage identify reusable outputs; the math memory budget does
        not affect identity. Reuse avoids writes but recomputes diagnostic
        corrections so the plot retains contributions from every integration,
        including columns whose original samples were all NaN.

        :param datacube: active normalized SCI/ERR/DQ stage, including any
                         jump-local selection or earlier processing stages
        :return: same manager, advanced only after writes and plot succeed
        :raises MemoryError: if the exact mean cannot fit the math budget
        :raises ValueError: if products or alignment fail validation
        :raises OSError: if a working product cannot be read or written
        :raises Exception: propagate filtering/plot errors without committing
        """
        func_name = f'{__NAME__}.low_pass_filter()'
        # Working products are mandatory; USE_TEMPORARY controls reuse only.
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        # No low-pass enable/calibration settings exist in the legacy stage.
        percentile = 50
        filter_width = 25
        signature = dict(percentile=percentile, filter_width=filter_width,
                         algorithm='column-median-lowpass-v1')
        products = datacube.plan_stage('lowpass', signature)
        reuse = use_temp and datacube.products_valid(products)
        # The mean tiles only output axes; no partial-mean reweighting occurs.
        with warnings.catch_warnings(record=True):
            mcube = datacube.nanmean('FLUX', axis=0)
        frame0 = datacube.read_integrations(0, 1)['FLUX'][0].copy()
        misc.printc('\tLow pass filtering the data', 'info')
        # Preserve the strict below-median rule independently in each column.
        mean_mask = np.zeros_like(mcube, dtype=float)
        for column in tqdm(range(mcube.shape[1])):
            mean_cut = np.nanpercentile(mcube[:, column], percentile)
            mean_mask[:, column] = mcube[:, column] < mean_cut
        mean_mask[mean_mask != 1] = np.nan
        # Only detector-sized diagnostic images survive between blocks.
        sum_cube_tile = np.zeros_like(mcube)
        corrected0 = frame0.copy()
        for product, (integration, arrays) in zip(
                products, datacube.iter_integrations()):
            for frame_index, frame in enumerate(arrays['FLUX']):
                with warnings.catch_warnings(record=True):
                    med = np.nanmedian(frame * mean_mask, axis=0)
                    lowp = mp.lowpassfilter(med, filter_width)
                # Broadcasting is identical to tiling the profile over rows.
                frame -= lowp
                # Weight integrations equally, never average block means.
                sum_cube_tile += lowp
                if integration.start == 0 and frame_index == 0:
                    corrected0 = frame.copy()
            if not reuse:
                # Never mutate ERR/DQ or publish a partially written stage.
                datacube.write_product(product, arrays)
            del frame, arrays
        sum_cube_tile /= len(datacube)
        if reuse:
            # Plot the validated cached output, not merely its reconstruction.
            with fits.open(products[0].filename, memmap=False) as hdul:
                science_hdu = cast(fits.ImageHDU, hdul['SCI'])
                corrected0 = science_hdu.section[0].astype(float)
        plots.plot_lowpass(self, frame0, corrected0, sum_cube_tile)
        # Validate all products again before changing active manager filenames.
        datacube.commit_products(products)
        return datacube

    def patch_isolated_bads(self, datacube: DataCube) -> DataCube:
        """Patch isolated NaNs from their four neighbours in bounded blocks.

        A non-finite pixel whose four edge neighbours are all finite becomes
        the plain mean of those neighbours in SCI; ERR becomes the plain mean
        of the same four ERR values (infinite neighbours propagate). DQ is
        unchanged. The heat map, before/after stamps, cutouts and plots use
        global integration indices of the active DataCube, accumulated across
        blocks exactly as the original full-cube loop. Reuse skips writes but
        recomputes diagnostics so plots and cutouts are always produced.

        :param datacube: active normalized SCI/ERR/DQ stage manager
        :return: same manager; advanced only after all writes, plots and
                 cutouts succeed, or unchanged when
                 WLC.GENERAL.PATCH_ISOLATED_BADS is false (no file I/O)
        :raises SossisseConstantException: if PATCH_IBADS_SSIZE is even
        :raises ValueError: if products or alignment fail validation
        :raises OSError: if a working product cannot be read or written
        :raises Exception: propagate stamp/plot/cutout errors uncommitted
        """
        # set function name
        func_name = f'{__NAME__}.patch_isolated_bads()'
        # Read only the enable switch until the stage is actually requested.
        wlc_params = self.params.get('WLC.GENERAL')
        patch_isolated_bads = wlc_params.rget('PATCH_ISOLATED_BADS',
                                              required=True, func=func_name)
        # deal with no patching isolated bad pixels
        if not patch_isolated_bads:
            # print message that we are not patching isolated bad pixels
            msg = ('WLC.GENERAL.PATCH_ISOLATED_BADS=False. Not patching '
                   'isolated bad pixels')
            misc.printc(msg, 'info')
            # Keep the original working products active.
            return datacube
        # Working products are mandatory; USE_TEMPORARY controls reuse only.
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        bad_pixel_stamp_size = wlc_params.rget('PATCH_IBADS_SSIZE',
                                               required=True, func=func_name)
        # Validate the stamp size before reading or writing any block.
        if bad_pixel_stamp_size % 2 == 0:
            emsg = 'WLC.GENERAL.PATCH_IBADS_SSIZE must be odd'
            raise exceptions.SossisseConstantException(emsg)
        # get the half size of the bad pixel stamps
        bpixel_ssize = (bad_pixel_stamp_size - 1) // 2
        # The stamp size only affects diagnostics, never product values.
        signature = dict(neighbours=4, fill='mean',
                         algorithm='isolated-four-neighbour-mean-v1')
        products = datacube.plan_stage('isolated', signature)
        reuse = use_temp and datacube.products_valid(products)
        # print progress
        msg = ' Removing isolated NaNs'
        misc.printc(msg, 'info')
        # storage for the heat map
        heat_map = np.zeros(datacube.shape[1:], dtype=float)
        # storage of bad pixel stamps, keyed by global integration index
        bad_pixels_before = dict()
        bad_pixels_after = dict()
        # store the positions of the bad pixels
        badpix_pos = dict()
        # make a kernel surrounding the pixel (3x3)
        kernel = np.zeros([3, 3], dtype=float)
        # set the four edge neighbours and the pixel itself to 1
        kernel[1, :] = 1
        kernel[:, 1] = 1
        # store the first frame before and after patching (for plot)
        iframe_before = datacube.read_integrations(0, 1)['FLUX'][0].copy()
        iframe_after = iframe_before.copy()
        # Only one block and detector-sized diagnostics reside in memory.
        for product, (integration, arrays) in zip(
                products, datacube.iter_integrations()):
            flux = arrays['FLUX']
            err = arrays['ERR']
            for local_index in range(flux.shape[0]):
                # Diagnostics retain the full-cube frame numbering.
                iframe = integration.start + local_index
                # get the cube frame
                cframe = np.array(flux[local_index])
                ecframe = np.array(err[local_index])
                # get a mask of the nan values
                mframe = np.isfinite(cframe)
                # find bad pixels that are isolated
                n_bad = convolve2d(np.array(mframe, dtype=float), kernel,
                                   mode='same')
                isolated_bad = (n_bad == 4) & ~mframe
                # add the isolated bad pixels to the heat map
                heat_map += isolated_bad.astype(int)
                # get the pixel positions with isolated bad pixels
                ypix, xpix = np.where(isolated_bad)
                # replace each value with the mean of its 4 neighbours
                mean_vals = np.mean([cframe[ypix - 1, xpix],
                                     cframe[ypix + 1, xpix],
                                     cframe[ypix, xpix - 1],
                                     cframe[ypix, xpix + 1]], axis=0)
                # do the same (plain mean) for the errors
                mean_err_vals = np.mean([ecframe[ypix - 1, xpix],
                                         ecframe[ypix + 1, xpix],
                                         ecframe[ypix, xpix - 1],
                                         ecframe[ypix, xpix + 1]], axis=0)
                # get the bad pixel stamps (before correction)
                bad_pixels_before[iframe] = self.create_bad_pixel_stamps(
                    ypix, xpix, cframe, bpixel_ssize)
                # update the science and error frames
                cframe[ypix, xpix] = mean_vals
                ecframe[ypix, xpix] = mean_err_vals
                # get the bad pixel stamps (after correction)
                bad_pixels_after[iframe] = self.create_bad_pixel_stamps(
                    ypix, xpix, cframe, bpixel_ssize)
                # push back into the block
                flux[local_index] = cframe
                err[local_index] = ecframe
                # save the bad pixel positions
                badpix_pos[iframe] = [ypix, xpix]
                # keep the first patched frame for the heat map plot
                if iframe == 0:
                    iframe_after = cframe.copy()
            if not reuse:
                # Writes remain unpublished until all diagnostics succeed.
                datacube.write_product(product, arrays)
            del flux, err, arrays
        if reuse:
            # Plot the validated cached output, not merely its reconstruction.
            with fits.open(products[0].filename, memmap=False) as hdul:
                science_hdu = cast(fits.ImageHDU, hdul['SCI'])
                iframe_after = science_hdu.section[0].astype(float)
        # ---------------------------------------------------------------------
        plots.plot_heatmap(self, heat_map, iframe_before, iframe_after,
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
        # plot the first frame that has isolated bad pixels (if any)
        if bpixel_cutouts_small:
            first_frame = min(bpixel_cutouts_small)
            plots.plot_pixels(self, bpixel_cutouts_small,
                              frame_num=first_frame)
        # save the cutouts of all bad pixels
        self.save_cutouts(pixel_cutouts_all, table_cutouts_all)
        # Validate all products again before changing active manager filenames.
        datacube.commit_products(products)
        return datacube


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
            outfilename = self.add_product_suffix(outfilename)
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
        ssize = self.params.rget('WLC.GENERAL.PATCH_IBADS_SSIZE', required=True,
                     func=func_name)
        # storage of output images
        images = dict()
        # position of the bad pixels
        bad_table = dict()
        # loop around each frame
        for iframe in tqdm(before_dict):
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
            # frames without isolated bad pixels have no stamps to show
            if length == 0:
                continue
            # index of each stamp in this frame's position arrays
            indices = np.arange(length)
            # -----------------------------------------------------------------
            # deal with cutting down number of pixels
            if num is not None and num < length:
                # select a random number "num" of pixels in the list
                indices = np.random.choice(length, num, replace=False)
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
            ideal_cols = max(1, int(np.sqrt(length / 2)))
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
                bad_itable['x'].append(pos_dict[iframe][1][indices[idx]])
                bad_itable['y'].append(pos_dict[iframe][0][indices[idx]])
                bad_itable['row'].append(row)
                bad_itable['col'].append(col)
                # Insert them into the image
                if mode == 'seperate':
                    image1[y0:y0 + ssize, x0:x0 + ssize] = before
                    image2[y0:y0 + ssize, x0:x0 + ssize] = after
                else:
                    after_x0 = x0 + ssize + 1
                    image1[y0:y0 + ssize, x0:x0 + ssize] = before
                    image1[y0:y0 + ssize, after_x0:after_x0 + ssize] = after
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

    def get_trace_positions(self, med: Optional[np.ndarray] = None,
                            cube: Optional[DataCube] = None,
                            log: bool = True) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :param med: optional 2D median image used by instrument overrides
        :param cube: optional active DataCube used by instrument overrides
                     for exact bounded statistics; never materialized
        :param log: bool, if True print progress
        :return: np.ndarray, the trace position map
        :raises NotImplementedError: always; child instruments must override
        """
        _ = self, med, cube, log
        raise NotImplementedError('get_trace_pos() must be implemented in '
                                  'child Instrument class')

    def get_trace_mask(self, med: Optional[np.ndarray] = None,
                       cube: Optional[DataCube] = None,
                       log: bool = True, no_plot: bool = True,
                       plot_frames: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the trace mask (True where the trace is, False otherwise)
        This is a combined map of all orders.

        :param med: np.ndarray, median image to plot the trace mask on
                    can be None if cube is supplied
        :param cube: active DataCube to plot the trace mask on; only the
                     plotted integrations are read, one at a time
                     can be None if med is supplied
        :param log: bool, if True print progress
        :param no_plot: bool, if True do not plot the trace mask
        :param plot_frames: optional integration indices (negative values
                            count from the last integration) to plot under
                            the mask; the median image is plotted instead
                            when cube is None
        :return: np.ndarray, Boolean trace mask shaped (y, x)
        :raises IndexError: if a plotted frame is outside the DataCube
        :raises OSError: if a plotted integration cannot be read
        """
        # set function name
        func_name = f'{__NAME__}.get_trace_mask()'
        trace_width = self.params.rget('WLC.GENERAL.LINRECON_TRACE_WIDTH',
                           required=True, func=func_name)
        # print progress
        if log:
            misc.printc('Getting the trace mask', 'info')
        # get x and y size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        # deal with no trace map required
        if trace_width < 1:
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
            images: Optional[List[np.ndarray]]
            if plot_frames is not None:
                if cube is None:
                    # As before, a med-only call plots the median image.
                    images = [cast(np.ndarray, med)]
                    labels = ['Median image']
                else:
                    nframes = len(cube)
                    images = []
                    for it in plot_frames:
                        # Negative indices count back as in cube indexing.
                        index = int(it) + nframes if it < 0 else int(it)
                        if not 0 <= index < nframes:
                            emsg = 'Plot frame {0} outside {1} integrations'
                            eargs = [it, nframes]
                            raise IndexError(emsg.format(*eargs))
                        # Read only this integration, never the full cube.
                        frame_arrays = cube.read_integrations(index,
                                                              index + 1)
                        images.append(frame_arrays['FLUX'][0])
                    # Labels keep the requested (possibly negative) index.
                    labels = [f'Frame{it}' for it in plot_frames]
            else:
                images, labels = None, None
            # plot the trace mask
            plots.plot_trace_mask(self, trace_mask, images, labels)
        # ---------------------------------------------------------------------
        # return the trace_mask
        return trace_mask

    def get_trace_pos(self, map2d: bool = False,
                      order_num: Optional[int] = None, round_pos: bool = True,
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the trace position

        :param map2d: bool, if True return a 2D map of the trace
        :param order_num: int, the order number to use
        :param round_pos: bool, if True round the positions to integers

        :return:
        """
        # set function name
        func_name = f'{__NAME__}.get_trace_pos()'
        # get wlc generation parameters
        gen_params = self.params.get('GENERAL')
        wlc_gen_params = self.params.get('WLC.GENERAL')
        xoffset = wlc_gen_params.rget('X_TRACE_OFFSET', required=True,
                                          func=func_name)
        yoffset = wlc_gen_params.rget('Y_TRACE_OFFSET', required=True,
                                          func=func_name)
        trace_wid_mask = wlc_gen_params.rget('TRACE_WIDTH_MASKING',
                             required=True, func=func_name)
        # get x and y size from cube
        ysize = self.get_variable('DATA_Y_SIZE', func_name)
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        # check that that pos file exists
        if not os.path.exists(gen_params['POS_FILE']):
            emsg = ('The trace position file {0} does not exist. '
                    'Please check the config file and set POS_FILE')
            eargs = [gen_params['POS_FILE']]
            raise exceptions.SossisseConstantException(emsg.format(*eargs))
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

    def get_wavegrid(self, order_num: Optional[int] = None,
                     return_xpix: bool = False,
                     source: Optional[str] = None
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the wave grid for the instrument

        Here we assume this is not order dependent
           Insturments with order dependent wavelength solutions should
           override this function

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

        # ---------------------------------------------------------------------
        # Here we assume this is not order dependent
        #   Instruments with order dependent wavelength solutions should
        #   override this function
        # ---------------------------------------------------------------------
        # These are only used in order dependent cases
        _ = order_num, source

        # if we have a wave file set and a wave type set then we should
        # be loading from wavefile
        if wave_file is not None and wave_type is not None:
            source = 'wavefile'
            # deal with base wave type
            if wave_type not in ['ext1d', 'fits', 'hdf5']:
                emsg = f'WAVE_TYPE must be ext1d, fits or hdf5'
                raise exceptions.SossisseConstantException(emsg)
            calibration_path = self.params.rget('PATHS.CALIBPATH', required=True,
                                                func=func_name)
            wavepath = os.path.join(calibration_path, wave_file)
        else:
            emsg = ('WAVE_FILE and WAVE_TYPE must be set in the config '
                    'file for {0}')
            eargs = [self.name]
            raise exceptions.SossisseConstantException(emsg.format(*eargs))

        # ---------------------------------------------------------------------
        # Deal with ext1d wave file - by default not valid
        if wave_type == 'ext1d' and source == 'wavefile':
            # use the default function to load the wave fits file
            xpix, wavevector = io.load_wave_ext1d(str(wavepath))
        # ---------------------------------------------------------------------
        # Deal with fits file wave file
        elif wave_type == 'fits' and source == 'wavefile':
            # use the default function to load the wave fits file
            xpix, wavevector = io.load_wave_fits(str(wavepath))
        # ---------------------------------------------------------------------
        # Deal with hdf5 fits file wave file
        elif wave_type == 'hdf5' and source == 'wavefile':
            # get x size from cube
            try:
                xsize = self.get_variable('DATA_X_SIZE', func_name)
            except exceptions.SossisseException:
                emsg = ('DATA_X_SIZE must be set in the config file for '
                        '{0} when using hdf5 wave file')
                eargs = [self.name]
                raise exceptions.SossisseConstantException(emsg.format(*eargs))
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
                # get x size from cube
                try:
                    xsize = self.get_variable('DATA_X_SIZE', func_name)
                except exceptions.SossisseException:
                    emsg = ('DATA_X_SIZE must be set in the config file for '
                            '{0} when using pos file')
                    eargs = [self.name]
                    raise exceptions.SossisseConstantException(emsg.format(*eargs))
                # get x trace offset
                x_trace_offset = self.params.rget('WLC.GENERAL.X_TRACE_OFFSET',
                                                  required=True, func=func_name)
                # get the trace position file
                tbl_ref = self.load_table(self.params['GENERAL.POS_FILE'],
                                          ext=None)
                xpix, wavevector = io.load_wave_posfile(tbl_ref, xsize,
                                                        x_trace_offset)
        # ---------------------------------------------------------------------
        # return the wave grid
        if return_xpix:
            return xpix, wavevector
        else:
            return wavevector

    def create_median_stack(self, datacube: DataCube
                            ) -> List[Union[DataCube, np.ndarray]]:
        """Build the amplitude-normalized median image with bounded math.

        The first exact median (baseline frames when MEDIAN_BASELINE and
        HAS_BASELINE, otherwise all frames) gives per-frame amplitudes, the
        dot product of each frame with the median over pixels finite in
        both. Frames divided by their amplitude (ERR and DQ unchanged) are
        written as 'mednorm' scratch products, reopened in a separate
        DataCube and reduced by a second exact median (baseline frames when
        MEDIAN_BASELINE). The scratch stage is never committed to datacube,
        whose active unnormalized products are returned unchanged. Only the
        median image is written to MEDIAN_IMAGE_FILE; the legacy full-cube
        CLEAN_CUBE_FILE is no longer written.

        :param datacube: active normalized SCI/ERR/DQ stage manager
        :return: list of 1. the same unchanged datacube, 2. the median image
                 shaped (y, x), 3. the per-integration amplitudes; with
                 GENERAL.USE_TEMPORARY, valid MEDIAN_IMAGE_FILE and
                 TEMP_AMP_FILE (and TEMP_PCA_FILE when WLC.LMODEL.FIT_PCA)
                 are reused without reading the cube; invalid files are
                 recomputed
        :raises MemoryError: if an exact median exceeds the math budget
        :raises ValueError: if products or alignment fail validation
        :raises OSError: if a product or the median image cannot be written
        """
        # define the function name
        func_name = f'{__NAME__}.{self.name}.create_median_stack()'
        # ---------------------------------------------------------------------
        # Read the sole control for reusing working products
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        # The active stage defines the number of integrations.
        nframes = len(datacube)
        # ---------------------------------------------------------------------
        # save these for later
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        tmp_pcas = self.get_variable('TEMP_PCA_FILE', func_name)
        tmp_amps = self.get_variable('TEMP_AMP_FILE', func_name)
        # ---------------------------------------------------------------------
        # Reuse existing working files only when requested
        if use_temp:
            fit_pca = self.params.rget('WLC.LMODEL.FIT_PCA', required=True,
                                       func=func_name)
            # make sure all required files exist
            cond = os.path.exists(median_image_file)
            cond &= os.path.exists(tmp_amps)
            # only look for fit pca file if we are fitting pca
            if fit_pca:
                cond &= os.path.exists(tmp_pcas)
            # if all conditions are satisfied we load the files
            if cond:
                # print that we are loading temporary files to speed things up
                msg = 'We read temporary files to speed things up.'
                misc.printc(msg, 'info')
                try:
                    # load the median image
                    msg = '\tReading: {0}'.format(median_image_file)
                    misc.printc(msg, 'info')
                    median_image = self.load_data(median_image_file)
                    # load the tmp file
                    misc.printc('\tReading: {0}'.format(tmp_amps), 'info')
                    amps = self.load_data(tmp_amps)
                except exceptions.SossisseFileException:
                    median_image, amps = None, None
                # Only cached shapes matching the active stage are valid.
                valid_median = (median_image is not None and np.shape(
                    median_image) == datacube.shape[1:])
                valid_amps = (amps is not None
                              and np.shape(amps) == (nframes,))
                if valid_median and valid_amps:
                    # return these files with the unchanged active stage
                    return_list = [datacube, median_image, amps]
                    return return_list
                # Invalid caches are misses; recompute everything below.
                msg = '\tTemporary median/amplitude files invalid: recompute'
                misc.printc(msg, 'warning')
        # ---------------------------------------------------------------------
        # first estimate of the trace amplitude
        misc.printc('\tFirst median of cube to create trace esimate', 'info')
        # validate out-of-transit domain
        self.get_baseline_params()
        has_baseline = self.get_variable('HAS_BASELINE', func_name)
        baseline_domain = self.get_variable('BASELINE_DOMAIN', func_name)
        # get flag for median out of transit
        med_baseline = self.params.rget('WLC.GENERAL.MEDIAN_BASELINE',
                                        required=True, func=func_name)
        # The baseline mask or index vector is passed unchanged as selection.
        if med_baseline and has_baseline:
            first_selection = baseline_domain
        else:
            first_selection = None
        # ---------------------------------------------------------------------
        # deal with creating median
        with warnings.catch_warnings(record=True) as _:
            # Tiles hold every selected integration of a pixel: exact median.
            med = datacube.nanmedian('FLUX', axis=0,
                                     selection=first_selection)
        # ---------------------------------------------------------------------
        # Amplitudes depend only on the first median's frame selection.
        if first_selection is None:
            first_frames = None
        else:
            first_frames = np.arange(nframes)[first_selection].tolist()
        signature = dict(baseline_frames=first_frames,
                         algorithm='median-amplitude-normalization-v1')
        products = datacube.plan_stage('mednorm', signature)
        reuse = use_temp and datacube.products_valid(products)
        # ---------------------------------------------------------------------
        # do a dot product of each trace to the median and adjust amplitude
        #   so that they all match
        amps = np.zeros(nframes)
        # Only one block and detector-sized maps reside in memory.
        blocks = zip(products, datacube.iter_integrations())
        for product, (integration, arrays) in tqdm(blocks,
                                                   total=len(products)):
            for local_index, frame in enumerate(arrays['FLUX']):
                # Amplitudes use global integration indices.
                iframe = integration.start + local_index
                # only keep finite values in all frames
                valid = np.isfinite(frame) & np.isfinite(med)
                # work out the amps
                part1a = np.nansum(frame[valid] * med[valid])
                part2a = np.nansum(med[valid] ** 2)
                amps[iframe] = part1a / part2a
                # normalize the data in the cube by these amplitues
                frame /= amps[iframe]
            if not reuse:
                # Scratch products only; ERR and DQ are not normalized.
                datacube.write_product(product, arrays)
            del arrays
        # ---------------------------------------------------------------------
        # median of the normalized cube
        misc.printc('\tSecond median of cube with proper normalization', 'info')
        # A separate manager keeps the active unnormalized stage unchanged.
        norm_paths = [product.filename for product in products]
        norm_kwargs: Dict[str, Any] = dict(
            max_integrations=datacube.max_integrations,
            max_math_memory_gb=datacube.max_math_memory_gb)
        normcube = DataCube(norm_paths, **norm_kwargs)
        # The second median ignores HAS_BASELINE, as before.
        second_selection = baseline_domain if med_baseline else None
        # normalize cube
        with warnings.catch_warnings(record=True) as _:
            # Exact median of the normalized frames, tiled spatially.
            med = normcube.nanmedian('FLUX', axis=0,
                                     selection=second_selection)
        # ---------------------------------------------------------------------
        # write files to disk
        # ---------------------------------------------------------------------
        # write the median image
        misc.printc('\tWriting: {0}'.format(median_image_file), 'info')
        fits.writeto(median_image_file, med, overwrite=True)
        # ---------------------------------------------------------------------
        return [datacube, med, amps]

    def clean_residual_1f(self, datacube: DataCube, med: np.ndarray,
                          amps: np.ndarray,
                          trace_mask: np.ndarray) -> DataCube:
        """Subtract the residual 1/f model in bounded SCI/ERR/DQ blocks.

        Per-frame amplitudes are recomputed from the trace and mutate ``amps``
        in place. SCI is corrected; ERR and DQ remain unchanged. The stage is
        published only after products, median image and diagnostic plots all
        succeed. Temporary reuse skips product writes but still recomputes
        amplitudes and diagnostics from the active input products.

        :param datacube: active normalized SCI/ERR/DQ product manager
        :param med: detector median image shaped (y, x)
        :param amps: mutable per-integration amplitude vector
        :param trace_mask: Boolean map marking trace pixels
        :return: same manager, advanced to the ``subtract_1f`` products unless
                 WLC.INPUTS.APPLY_1F_CORR is false, when it is unchanged
        :raises ValueError: if planned product blocks fail validation
        :raises OSError: if a product or median image cannot be written/read
        :raises Exception: propagate subtraction and plot errors uncommitted
        """
        # Keep the function identity stable for parameter diagnostics.
        func_name = f'{__NAME__}.{self.name}.clean_residual_1f()'
        # Read the feature switch before touching DataCube products or paths.
        apply_1f_corr = self.params.rget('WLC.INPUTS.APPLY_1F_CORR',
                                         required=True, func=func_name)
        if not apply_1f_corr:
            # A disabled stage leaves files and amplitudes untouched.
            msg = ('WLC.INPUTS.APPLY_1F_CORR=False. '
                   'Not applying 1/f correction')
            misc.printc(msg, 'info')
            return datacube
        # Read reuse and numerical settings only for an active correction.
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        degree_1f_corr = self.params.rget(
            'WLC.GENERAL.DEGREE_1F_CORR', required=True, func=func_name)
        median_image_file = self.get_variable('MEDIAN_IMAGE_FILE', func_name)
        # Hash exact median/mask bytes so changed diagnostics invalidate reuse.
        median_hash = hashlib.sha256(
            np.ascontiguousarray(med).tobytes()).hexdigest()
        trace_hash = hashlib.sha256(
            np.ascontiguousarray(trace_mask).tobytes()).hexdigest()
        signature = dict(median_shape=med.shape, median_sha256=median_hash,
                         trace_shape=trace_mask.shape,
                         trace_sha256=trace_hash,
                         degree=degree_1f_corr,
                         algorithm='residual-1f-subtraction-v1')
        # Parents and current selection are included by DataCube planning.
        products = datacube.plan_stage('subtract_1f', signature)
        reuse = use_temp and datacube.products_valid(products)
        misc.printc('\nCalculating residuals', 'info')
        # Retain only 2-D diagnostics, never the full correction cube.
        frame_before = None
        frame_after = None
        scorr_first = None
        scorr_last = None
        scorr_sum = np.zeros(med.shape, dtype=float)
        scorr_count = np.zeros(med.shape, dtype=np.int64)
        # Residuals and corrections stay within one active product block.
        blocks = zip(products, datacube.iter_integrations())
        for product, (integration, arrays) in tqdm(blocks,
                                                   total=len(products)):
            flux = arrays['FLUX']
            residuals = np.zeros_like(flux)
            # Preserve the old amplitude formula in global integration order.
            for local_index, frame in enumerate(flux):
                iframe = integration.start + local_index
                valid = np.isfinite(frame) & np.isfinite(med)
                valid &= trace_mask
                numerator = np.nansum(frame[valid] * med[valid])
                denominator = np.nansum(med[valid] ** 2)
                amps[iframe] = numerator / denominator
                residuals[local_index] = frame - med * amps[iframe]
                if iframe == 0:
                    # Capture the actual pre-correction first integration.
                    frame_before = frame.copy()
            # The helper mutates only SCI and returns this block's scorr map.
            corrected, scorr = self.subtract_1f(residuals, flux,
                                                arrays['ERR'], trace_mask)
            # Keep endpoint images and NaN-aware mean sufficient statistics.
            for local_index in range(scorr.shape[0]):
                iframe = integration.start + local_index
                if iframe == 0:
                    scorr_first = scorr[local_index].copy()
                    frame_after = corrected[local_index].copy()
                scorr_last = scorr[local_index].copy()
            valid_scorr = ~np.isnan(scorr)
            scorr_sum += np.sum(np.where(valid_scorr, scorr, 0), axis=0)
            scorr_count += np.sum(valid_scorr, axis=0)
            if not reuse:
                # ERR/DQ are passed through unchanged in the product payload.
                datacube.write_product(product, arrays)
            del flux, residuals, corrected, scorr, arrays
        # Divide only where at least one non-NaN correction was observed.
        mean_scorr = np.full(med.shape, np.nan, dtype=float)
        np.divide(scorr_sum, scorr_count, out=mean_scorr,
                  where=scorr_count > 0)
        assert frame_before is not None and frame_after is not None
        assert scorr_first is not None and scorr_last is not None
        # Preserve the median-image side effect in both fresh and reused runs.
        misc.printc('\tWriting: {0}'.format(median_image_file), 'info')
        fits.writeto(median_image_file, med, overwrite=True)
        # Plot only the first, last and mean correction maps plus first frames.
        plots.plot_subtract_1f_scorr(self, scorr_first, scorr_last, mean_scorr)
        plots.plot_subtract_1f_comp(self, frame_before, frame_after)
        # Publish only after every product, image and plot has succeeded.
        datacube.commit_products(products)
        return datacube

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
            raw_baseline_ints = self.localize_frame_domains(raw_baseline_ints,
                                                            'BASELINE_INTS')
            if len(raw_baseline_ints) == 0:
                emsg = 'BASELINE_INTS has no overlap with this jump chunk'
                raise exceptions.SossisseConstantException(emsg)
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
        rej_domain = self.localize_frame_domains(rej_domain, 'REJECT_DOMAIN')
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
            if len(rej_domain) > 0:
                # get the rejection domain
                for ireject in range(len(rej_domain)):
                    # get the start and end of the domain to reject
                    start = rej_domain[ireject][0]
                    end = rej_domain[ireject][1]
                    # set to False in valid_baseline
                    valid_baseline[start:end + 1] = False
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

    def subtract_1f(self, residuals: np.ndarray, cube: np.ndarray,
                    err: np.ndarray, trace_mask: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subtract the 1/f correction from one bounded integration block.

        :param residuals: residual SCI block shaped (n, y, x)
        :param cube: SCI block modified in place
        :param err: unchanged ERR block shaped like ``cube``
        :param trace_mask: Boolean trace map shaped (y, x)
        :return: mutated SCI block and its correction block (n, y, x)
        :raises ValueError: if NumPy cannot perform the requested reductions
        """
        # Keep parameter errors tied to the public correction stage.
        func_name = f'{__NAME__}.{self.name}.clean_residual_1f()'
        degree_1f_corr = self.params.rget(
            'WLC.GENERAL.DEGREE_1F_CORR', required=True, func=func_name)
        # Product shapes, not instrument globals, define this block's extent.
        nframes, _, xsize = cube.shape
        scorr = np.zeros_like(cube)
        misc.printc('\tSubtracting 1/f noise', 'info')
        if degree_1f_corr == 0:
            # Compute the original per-frame column median correction.
            with warnings.catch_warnings(record=True) as _:
                noise_1f = np.nanmedian(residuals, axis=1)
            for iframe in tqdm(range(nframes)):
                for column in range(xsize):
                    scorr[iframe, :, column] = noise_1f[iframe, column]
                    cube[iframe, :, column] -= noise_1f[iframe, column]
        else:
            # Preserve the weighted polynomial fit and its sparse-column skip.
            for iframe in tqdm(range(nframes)):
                frame_residual = residuals[iframe]
                frame_error = np.array(err[iframe])
                for column in range(xsize):
                    column_residual = frame_residual[:, column]
                    finite_count = np.sum(np.isfinite(column_residual))
                    if finite_count < degree_1f_corr + 3:
                        continue
                    column_error = frame_error[:, column]
                    index = np.arange(frame_residual.shape[0], dtype=float)
                    valid = np.isfinite(column_residual + column_error)
                    valid &= ~trace_mask[:, column]
                    valid &= np.abs(column_residual / column_error) < 5
                    try:
                        fit = np.polyfit(index[valid], column_residual[valid],
                                         degree_1f_corr,
                                         w=1 / column_error[valid])
                        correction = np.polyval(fit, index)
                        cube[iframe, :, column] -= correction
                        scorr[iframe, :, column] = correction
                    except Exception:
                        cube[iframe, :, column] = np.nan
                        scorr[iframe, :, column] = np.nan
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
        temp_path = self.params.rget('PATHS.TEMP_PATH', required=True,
                         func=func_name)
        flag_fit_pca = self.params.rget('WLC.LMODEL.FIT_PCA', required=True,
                        func=func_name)
        n_comp = self.params.rget('WLC.LMODEL.FIT_N_PCA', required=True,
                      func=func_name)
        # ---------------------------------------------------------------------
        # Read the sole control for reusing working products
        use_temp = self.params.rget('GENERAL.USE_TEMPORARY', required=True,
                                    func=func_name)
        # update meta data
        self.update_meta_data()
        # make sure we have a pca file
        tmp_pcas = 'temporary_pcas.fits'
        tmp_pcas = os.path.join(temp_path, tmp_pcas)
        self.set_variable('TEMP_PCA_FILE', tmp_pcas)
        # ---------------------------------------------------------------------
        # if we aren't fitting or we fit no components return None
        if (not flag_fit_pca) or n_comp == 0:
            msg = ('WLC.LMODEL.FIT_PCA={0}  WLC.LMODEL.FIT_N_PCA={1} '
                   'Not fitting with PCA').format(flag_fit_pca, n_comp)
            misc.printc(msg, 'info')
            return None
        # ---------------------------------------------------------------------
        # only look for fit pca file if we are fitting pca
        if use_temp and os.path.exists(tmp_pcas):
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
        # Always persist working products
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
        recenter_trace = wlc_gen_params.rget('RECENTER_TRACE_POSITION',
                            required=True, func=func_name)
        # deal with not wanting to recenter trace position
        if not recenter_trace:
            # print message that we are not removing cosmic rays
            msg = ('WLC.GENERAL.RECENTER_TRACE_POSITION=False. '
                   'Not recentering trace position')
            misc.printc(msg, 'info')
            return trace_mask
        trace_width = wlc_gen_params.rget('TRACE_WIDTH_MASKING', required=True,
                                          func=func_name)
        width_current = float(trace_width)
        trace_x_scale = wlc_gen_params.rget('TRACE_X_SCALE', required=True,
                                            func=func_name)
        trace_y_scale = wlc_gen_params.rget('TRACE_Y_SCALE', required=True,
                                            func=func_name)
        y_trace_offset = wlc_gen_params.rget('Y_TRACE_OFFSET', required=True,
                                             func=func_name)
        x_trace_offset = wlc_gen_params.rget('X_TRACE_OFFSET', required=True,
                                             func=func_name)
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
        # force a trace width masking
        wlc_gen_params['TRACE_WIDTH_MASKING'] = 20
        wlc_gen_params.set_source('TRACE_WIDTH_MASKING', func_name)
        # get a range of dys and dxs to scan over for best trace position
        dys = np.arange(-nbypix // trace_y_scale, nbypix // trace_y_scale + 1)
        dys += y_trace_offset
        dxs = np.arange(-nbypix // trace_x_scale, nbypix // trace_x_scale + 1)
        dxs += x_trace_offset
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
        wlc_gen_params = self.params.get('WLC.GENERAL')
        trace_width_masking = wlc_gen_params.rget('TRACE_WIDTH_MASKING',
                             required=True)
        # set up the mask trace (all true to start)
        mask_trace_pos = np.ones_like(med, dtype=int)
        # ---------------------------------------------------------------------
        if trace_width_masking != 0:
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

    def set_linear_output_metadata(self, include_pca: bool = True):
        """
        Set output column names, units, and plotting scale factors.

        This state is normally created while constructing the linear model,
        but must also be restored when temporary products skip that work.

        :param include_pca: bool, include configured PCA output columns
        :return: None
        """
        lm_params = self.params.get('WLC.LMODEL')
        fit_dx = lm_params.rget('FIT_DX', required=True)
        fit_dy = lm_params.rget('FIT_DY', required=True)
        fit_rotation = lm_params.rget('FIT_ROTATION', required=True)
        fit_zero_point_offset = lm_params.rget('FIT_ZERO_POINT_OFFSET',
                               required=True)
        fit_ddy = lm_params.rget('FIT_DDY', required=True)
        fit_pca = lm_params.rget('FIT_PCA', required=True)
        # Start with the amplitude term, which is always fitted.
        output_names: List[str] = ['amplitude']
        output_units: List[str] = ['flux']
        output_factor: List[float] = [1.0]
        # Add horizontal trace motion metadata when that term is enabled.
        if fit_dx:
            output_names.append('dx')
            output_units.append('mpix')
            output_factor.append(1e3)
        # Add vertical trace motion metadata when that term is enabled.
        if fit_dy:
            output_names.append('dy')
            output_units.append('mpix')
            output_factor.append(1e3)
        # Add trace rotation metadata when that term is enabled.
        if fit_rotation:
            output_names.append('theta')
            output_units.append('mpix')
            output_factor.append(129600 / (2 * np.pi))
        # Add the constant zero-point metadata when that term is enabled.
        if fit_zero_point_offset:
            output_names.append('zeropoint')
            output_units.append('flux')
            output_factor.append(1.0)
        # Add second-derivative metadata when that term is enabled.
        if fit_ddy:
            output_names.append('ddy')
            output_units.append('mpix$^2$')
            output_factor.append(1e6)
        # Add one metadata entry for every configured PCA component.
        if fit_pca and include_pca:
            n_comp = lm_params.rget('FIT_N_PCA', required=True)
            for icomp in range(n_comp):
                output_names.append(f'PCA{icomp + 1}')
                output_units.append('ppm')
                output_factor.append(1.0)
        # Cache all three lists for coefficient handling and stability plots.
        self.set_variable('OUTPUT_NAMES', output_names)
        self.set_variable('OUTPUT_UNITS', output_units)
        self.set_variable('OUTPUT_FACTOR', output_factor)

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
        lm_params = self.params.get('WLC.LMODEL')
        fit_dx = lm_params.rget('FIT_DX', required=True)
        fit_dy = lm_params.rget('FIT_DY', required=True)
        fit_rotation = lm_params.rget('FIT_ROTATION', required=True)
        fit_zero_point_offset = lm_params.rget('FIT_ZERO_POINT_OFFSET',
                               required=True)
        fit_ddy = lm_params.rget('FIT_DDY', required=True)
        fit_pca = lm_params.rget('FIT_PCA', required=True)
        # vector is the median ravelled
        vector = [med.ravel()]
        # ---------------------------------------------------------------------
        # deal with fit dx
        if fit_dx:
            vector.append(dx.ravel())
        # ---------------------------------------------------------------------
        # deal with fix dy
        if fit_dy:
            vector.append(dy.ravel())
        # ---------------------------------------------------------------------
        # deal with fit rotation
        if fit_rotation:
            vector.append(rotxy.ravel())
        # ---------------------------------------------------------------------
        # deal with zero point offset fit
        if fit_zero_point_offset:
            vector.append(np.ones_like(dx.ravel()))
        # ---------------------------------------------------------------------
        # deal with fit second derivative
        if fit_ddy:
            vector.append(ddy.ravel())
        # ---------------------------------------------------------------------
        # deal with fit pca
        if fit_pca and pca is not None:
            n_comp = lm_params.rget('FIT_N_PCA', required=True)
            for icomp in range(n_comp):
                vector.append(pca[icomp].ravel())
        # ---------------------------------------------------------------------
        # convert vector to numpy array
        vector = np.array(vector)
        # ---------------------------------------------------------------------
        # Set metadata independently so temporary cache hits can restore it.
        self.set_linear_output_metadata(include_pca=pca is not None)
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
        zpoint = self.params.rget('WLC.LMODEL.FIT_ZERO_POINT_OFFSET',
                                  required=True, func=func_name)
        # get parameters
        nframes = self.get_variable('DATA_N_FRAMES', func_name)
        output_names = self.get_variable('OUTPUT_NAMES', func_name)
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
        poly_order = self.params.rget('WLC.GENERAL.TRANSIT_BASELINE_POLYORD',
                                      required=True, func=func_name)
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
    def get_traces(self) -> List[Union[int, None]]:
        """
        Return validated trace identities without coercing their values.

        SOSS traces label diffraction orders. NIRSpec overrides this method
        for detector labels (1=NRS1, 2=NRS2); other instruments may use None.

        :return: list, configured positive integers or [None] if unnumbered
        :raises SossisseConstantException: if a non-null selection is invalid
        """
        # A null selection denotes one unnumbered extraction, not no traces.
        traces = self.params['GENERAL.TRACES']
        if traces is None:
            return [None]
        # Keep type and nonempty checks ahead of iteration and uniqueness.
        invalid_selection = not isinstance(traces, list) or not traces
        if not invalid_selection:
            invalid_selection = any(type(trace) is not int or trace < 1
                                    for trace in traces)
        if not invalid_selection:
            invalid_selection = len(set(traces)) != len(traces)
        if invalid_selection:
            emsg = ('GENERAL.TRACES must be a nonempty list of unique '
                    'positive integers or null')
            raise exceptions.SossisseConstantException(emsg)
        # Preserve the configured list and its ordering for valid selections.
        return traces


    def load_input_spec_data(self, trace: Optional[int] = None
                             ) -> List[np.ndarray]:
        """
        Load input data for this trace for the spectral extraction

        :param trace: int, trace to load

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
        posmax, throughput = self.get_trace_pos(order_num=trace)
        # get wave grid
        wavegrid = self.get_wavegrid(order_num=trace)
        # return all input data
        return [med, dx, dy, rotxy, ddy, med_clean, residual, err,
                recon, ltable, posmax, throughput, wavegrid]



    def create_sed(self, med: np.ndarray, residual: np.ndarray,
                   wavegrid: np.ndarray, posmax: np.ndarray,
                   throughput: np.ndarray, med_clean: np.ndarray,
                   trace: int) -> np.ndarray:
        """
        Sum the clean median inside the configured trace ribbon and plot it.

        :param med: array, median image with shape (n_rows, n_pixels)
        :param residual: array, residual cube defining frame and spatial sizes
        :param wavegrid: array, wavelength values at each spectral pixel
        :param posmax: array, trace row per pixel; NaN skips that column
        :param throughput: array, throughput used only for the SED plot
        :param med_clean: array, cleaned median image summed within the ribbon
        :param trace: int or None, SOSS order, NIRSpec detector, or unnumbered
        :return: array, SED of shape (n_pixels,); updates stored dimensions
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.create_sed()'
        trace_width = self.params.rget('WLC.GENERAL.LINRECON_TRACE_WIDTH',
                                       required=True, func=func_name)
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
        plots.plot_sed(self, wavegrid, sp_sed / throughput, trace)
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
        trace_width = self.params.rget('WLC.GENERAL.LINRECON_TRACE_WIDTH',
                                       required=True, func=func_name)
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        nbypix = self.get_variable('DATA_Y_SIZE', func_name)
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
        polydeg = self.params.rget('WLC.GENERAL.TRACE_BASELINE_POLYORD',
                                   required=True, func=func_name)
        # get the number of frames
        nbframes = self.get_variable('DATA_N_FRAMES', func_name)
        # get the number of x and y pixels
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
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

    def save_spe_results(self, storage: Dict[str, Any],
                         trace: Optional[int] = None):
        """
        Save the results to disk
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.save_results()'
        save_results = self.params.rget('GENERAL.SAVE_RESULTS', required=True,
                                        func=func_name)
        # deal with not saving results --> return
        if not save_results:
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

        # deal with no trace
        if trace is None:
            trace = ''
        # ---------------------------------------------------------------------
        # Save the SED table
        # ---------------------------------------------------------------------
        # get objname
        objname = self.params['INPUTS.OBJECTNAME']
        # construct sed table name
        sed_table_name = self.get_variable('SPE_SED_TBL', func_name)
        sed_table_name = sed_table_name.format(objname=objname,
                                               trace=trace)
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
        res_no_grey_ord = res_no_grey_ord.format(trace=trace)
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
        res_grey_ord = res_grey_ord.format(trace=trace)
        # write the residual grey order tile
        io.save_fits(res_grey_ord, datalist=[spec2, spec_err],
                     datatypes=['image', 'image'],
                     datanames=['SPECTRUM', 'SPECTRUM_ERROR'],
                     meta=meta_data)
        # ---------------------------------------------------------------------
        # Save the spectra for this trace to file
        # ---------------------------------------------------------------------
        # get file name
        spec_file = self.get_variable('SPECTRA_ORD', func_name)
        spec_file = spec_file.format(trace=trace)
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
        # Save the wavelength for this trace to file
        # ---------------------------------------------------------------------
        # get file name
        wave_file = self.get_variable('WAVE_ORD', func_name)
        wave_file = wave_file.format(trace=trace)
        # write the wavelength file
        io.save_fits(wave_file, datalist=[wavegrid], datatypes=['image'],
                     datanames=['WAVELENGTH'], meta=meta_data)

        # ---------------------------------------------------------------------
        # Save the transit spectrum for this trace to file
        # ---------------------------------------------------------------------
        # # get file name
        # transit_file = self.get_variable('TSPEC_ORD', func_name)
        # transit_file = transit_file.format(trace=trace)

        # # make transit table
        # t_table = Table()
        # t_table['wavelength'] = wavegrid
        # t_table['flux'] = (spec_in + transit_depth) * 1e6
        # t_table['flux_err'] = spec_err_in * 1e6
        # # save table
        # io.save_table(transit_file, t_table, fmt='csv')

        # # ---------------------------------------------------------------------
        # # Save the transit spectrum for this trace (binned) to file
        # # ---------------------------------------------------------------------
        # # get resolution_bin
        # res_bin = self.params['SPEC_EXT.RESOLUTION_BIN']
        # # get file name
        # transit_bin_file = self.get_variable('TSPEC_ORD_BIN', func_name)
        # transit_bin_file = transit_bin_file.format(trace=trace,
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
        # get the traces from the
        traces = list(storage.keys())
        # print progress
        msg = 'Converting to Eureka format'
        misc.printc(msg, 'info')
        # loop around traces
        for trace in traces:
            # ----------------------------------------------------------------
            # print progress
            msg = f'Processing order {trace}'
            misc.printc(msg, 'info')
            # get variables from storage
            flux = np.array(storage[trace]['spec2'])
            flux_err = np.array(storage[trace]['spec_err'])
            wavegrid = np.array(storage[trace]['wavegrid'])
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
            # set filename (Dealing with trace or lack of orders)
            if trace is None:
                filename = filename.format(trace='')
            else:
                filename = filename.format(trace=trace)
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
        output_dir = self.params.rget('PATHS.OUT_PATH', required=True,
                                      func=func_name)
        # print progress
        msg = 'Saving final outputs'
        misc.printc(msg, 'info')
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
        each trace and the param snapshot table

        :param storage: dict, storage dictionary
        :param meta_data: dict, meta data dictionary
        :param psnapshot: Table, param snapshot table

        :return: None, saves fits file to OUT_SPEC_LC_FILE
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.out_spec_lc_file()'
        # get the traces from the storage keys
        traces = list(storage.keys())
        # storage for fits data writing
        datalist = []
        datatypes = []
        datanames = []
        # loop around trace ordesr
        for trace in traces:
            # get the wavelength from storage for this trace
            wavegrid = storage[trace]['wavegrid_2d']
            # get the 2D spectrum from storage for this trace
            spec2 = storage[trace]['spec2']
            # get the 2D spectrum error from storage for this trace
            spec_err = storage[trace]['spec_err']
            # get the integration times from storage for each integration
            int_times = storage[trace]['ltable']['bjd']
            # push into data list
            datalist.extend([wavegrid, spec2, spec_err, int_times])
            datatypes.extend(['image', 'image', 'image', 'image'])
            # Deal with not having traces
            if trace is None:
                datanames.extend(['WAVELENGTH', 'RELFLUX', 'RELFLUX_ERROR',
                                  'BJD'])
            else:
                datanames.extend([f'WAVELENGTH_{trace}',
                                  f'RELFLUX_{trace}',
                                  f'RELFLUX_ERROR_{trace}',
                                  f'BJD_{trace}'])
        # add the param table
        datalist.append(psnapshot)
        datatypes.append('table')
        datanames.append('PARAMS')
        # construct the fits filename
        filename = self.get_variable('OUT_SPEC_LC_FILE', func_name)
        filename = filename.format(prefix=self.params['GENERAL.PREFIX'])
        # save the fits file
        io.save_fits(filename, datalist=datalist, datatypes=datatypes,
                     datanames=datanames, meta=meta_data)

    def out_wlc_file(self, storage: Dict[int, Dict[str, Any]],
                     meta_data: Dict[str, Any],
                     psnapshot: Table):
        """
        Produce the white light curve fits file with the linear fit table
        and the param snapshot table

        NIRSpec extensions retain independent detector light curves and
        timestamps. Numbered SOSS extensions retain the shared linear
        reconstruction amplitude, identified by WLCMODEL=SHARED.

        :param storage: dict, storage dictionary
        :param meta_data: dict, meta data dictionary
        :param psnapshot: Table, param snapshot table

        :return: None, saves fits file to OUT_WLC_FILE
        """
        # set the function name
        func_name = f'{__NAME__}.{self.name}.out_wlc_file()'
        # get the traces from the storage keys
        traces = list(storage.keys())
        # storage for fits data writing
        datalist = []
        datatypes = []
        datanames = []
        # Preserve each trace's own timing rather than reusing the first table.
        for trace in traces:
            # SOSS entries can share amplitudes; detector entries are distinct.
            ltable = storage[trace]['ltable']
            rtable = Table()
            # Keep each integration's timestamp aligned with its amplitude.
            rtable['Time'] = ltable['bjd']
            rtable['RelFlux'] = ltable['amplitude']
            rtable['RelFlux_Err'] = ltable['amplitude_error']
            # One table extension is written for each stored trace identity.
            datalist.append(rtable)
            datatypes.append('table')
            table_name = 'WLC_TABLE' if trace is None else f'WLC_TABLE_{trace}'
            datanames.append(table_name)
        # SOSS orders are not independently extracted detector light curves.
        if self.name.startswith('JWST.NIRISS.SOSS'):
            meta_data['WLCMODEL'] = (
                'SHARED', 'Traces share linear reconstruction amplitude')
        # add the param table
        datalist.append(psnapshot)
        datatypes.append('table')
        datanames.append('PARAMS')
        # construct the fits filename
        filename = self.get_variable('OUT_WLC_FILE', func_name)
        filename = filename.format(prefix=self.params['GENERAL.PREFIX'])
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
        filename = filename.format(prefix=self.params['GENERAL.PREFIX'])
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
