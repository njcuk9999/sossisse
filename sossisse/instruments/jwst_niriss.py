#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
from typing import Tuple

import numpy as np
from astropy.io import fits

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.jwst_niriss'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
class JWST_NIRISS(default.Instrument):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRISS, self).__init__(params)
        # set up the instrument
        self.param_override()


class JWST_NIRISS_SOSS(JWST_NIRISS):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRISS_SOSS, self).__init__(params)
        # set name
        self.name = 'JWST.NIRISS.SOSS'
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        # run the super version first, then override after
        super().param_override()

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

            # for SOSS we cut down the flat field to the correct size
            # if it is a full frame flat
            # Question: Is this only for substrip256?
            if flat.shape == (2048, 2048):
                # cut down the flat field
                flat = flat[-256:]
            # check the shape of the flat field
            if flat.shape[1:] != cube_shape[1:]:
                emsg = 'Flat field shape does not match data frame shape'
                raise exceptions.SossisseInstException(emsg, self.name)
            # some sanity checks in flat
            flat[flat == 0] = np.nan
            flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
            flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
            # return the flat field
            return flat

    def get_trace_positions(self, log: bool = True) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map
        """
        # only get order 1 if the a wavelength domain is set
        if self.params['WLC_DOMAIN'] is not None:
            # get the trace positions from the white light curve
            tracemap, _ = self.get_trace_pos(map2d=True, order_num=1)
        else:
            tracemap1, _ = self.get_trace_pos(map2d=True, order_num=1)
            tracemap2, _ = self.get_trace_pos(map2d=True, order_num=2)
            # combine the two trace maps
            tracemap = tracemap1 | tracemap2
        # return the trace positions
        return tracemap


class JWST_NIRISS_FGS(JWST_NIRISS_SOSS):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRISS_FGS, self).__init__(params)
        # set name
        self.name = 'JWST.NIRISS.FGS'
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        # run the super version first, then override after
        super().param_override()

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
                    tmp_err = np.sqrt(np.abs(tmp_data))
                    tmp_dq = np.zeros_like(tmp_data)
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
