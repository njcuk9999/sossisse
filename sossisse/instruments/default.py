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
from typing import Any, Dict, Tuple, Union

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc

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
        # set the parameters from input
        self.params = copy.deepcopy(params)
        # set all sources to Unknown
        self.sources = {key: 'Unknown' for key in self.params}
        # set up the instrument
        self.param_override()
        # variables to keep in memory and pass around with class
        self.variables = dict()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        pass

    def load_data(self, filename):
        """
        Load the data from a file

        :param filename: str, the filename to load
        :return: data, the loaded data
        """
        try:
            data = fits.getdata(filename)
        except Exception as e:
            emsg = 'Error loading data from file: {0}\n\t{1}: {2}'
            eargs = [filename, type(e), str(e)]
            raise exceptions.SossisseFileException(emsg.format(*eargs))

    def load_data_with_dq(self) -> Tuple[np.ndarray, np.ndarray]:

        # get the conditions for allowing and using temporary files
        allow_temp = self.params['ALLOW_TEMPORARY']
        use_temp = self.params['USE_TEMPORARY']
        # construct temporary file names
        temp_ini_cube = 'temporary_initial_cube.fits'
        temp_ini_cube = os.path.join(self.params['TEMP_PATH'], temp_ini_cube)
        temp_ini_err = 'temporary_initial_err.fits'
        temp_ini_err = os.path.join(self.params['TEMP_PATH'], temp_ini_err)
        # ---------------------------------------------------------------------
        # if we are allowed temporary files and are using them then load them
        if allow_temp and use_temp:
            if os.path.exists(temp_ini_cube) and os.path.exists(temp_ini_err):
                # load the data
                cube = self.load_data(temp_ini_cube)
                err = self.load_data(temp_ini_err)
                # for future reference in the code, we keep track of data size
                self.variables['DATA_X_SIZE'] = cube.shape[2]
                self.variables['DATA_Y_SIZE'] = cube.shape[1]
                self.variables['DATA_N_FRAMES'] = cube.shape[0]
                # return
                return cube, err
        # ---------------------------------------------------------------------
        # number of slices in the final cube
        n_slices = 0
        # storage of raw data
        raw_data = []
        # handling the files with their varying sizes. We read and count slices
        for ifile, filename in enumerate(self.params['FILES']):
            # get the raw files
            raw_data.append(self.load_data(filename))
            # get the shape of the bins
            bin_shape = self.bin_cube(raw_data[ifile], get_shape=True)
            # add to the number of slices
            n_slices += bin_shape[0]
        # ---------------------------------------------------------------------
        # store the shape of a single frame
        raw_shape = raw_data[0].shape
        # deal with cds
        if len(raw_shape) == 4:
            raw_data, raw_shape =
        # ---------------------------------------------------------------------
        # get flat
        flat = self.get_flat(cube_shape=raw_shape)
        # ---------------------------------------------------------------------
        # create the containers for the cube of science data,
        # the error cube, and the DQ cube
        cube = np.zeros([n_slices, raw_shape[1], raw_shape[2]])
        err = np.zeros([n_slices, raw_shape[1], raw_shape[2]])
        dq = np.zeros([n_slices, raw_shape[1], raw_shape[2]])


        pass

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
            raise exceptions.SossisseException(emsg.format(*eargs))
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
        return new_cube

    def get_flat(self, cube_shape: Tuple[int, int, int]):
        """
        Get the flat field

        :return:
        """
        if self.params['FLATFILE'] is None:
            # flat field is a single frame
            return np.ones((cube_shape[1], cube_shape[2]))
        else:
            # load the flat field
            flat = self.load_data(self.params['FLATFILE'])
            # check the shape of the flat field
            if flat.shape[1:] != cube_shape[1:]:
                emsg = 'Flat field shape does not match data frame shape'
                raise exceptions.SossisseException(emsg)
            # some sanity checks in flat
            flat[flat == 0] = np.nan
            flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
            flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
            # return the flat field
            return flat




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
