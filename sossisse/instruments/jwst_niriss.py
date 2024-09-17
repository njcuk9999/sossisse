#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
from typing import List, Tuple

import numpy as np
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.ndimage import binary_dilation
from skimage import measure
from tqdm import tqdm

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import math as mp
from sossisse.instruments import default
from sossisse.general import plots

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

    def get_flat(self, image_shape: Tuple[int, int, int]
                 ) -> Tuple[np.ndarray, bool]:
        """
        Get the flat field
        
        :param image_shape: tuple, the shape of the image
        
        :return: tuple, 1. the flat field, 2. a boolean indicating if the flat
                 field is all ones
        """
        if self.params['FLATFILE'] is None:
            # flat field is a single frame
            return np.ones(image_shape), False
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
            if tuple(flat.shape) != tuple(image_shape):
                emsg = 'Flat field shape does not match data frame shape'
                raise exceptions.SossisseInstException(emsg, self.name)
            # some sanity checks in flat
            flat[flat == 0] = np.nan
            flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
            flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
            # return the flat field
            return flat, True

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
        # mask trace pos is not used - we get it from the tracemap
        _ = mask_trace_pos
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_mask_order0()'
        # get in vs out filename
        in_vs_out_file = self.get_variable('TEMP_TRANSIT_IN_VS_OUT', func_name)
        # get the size of the data
        nbxpix = self.get_variable('DATA_X_SIZE', func_name)
        # get the diff file
        diff = self.load_data(in_vs_out_file)
        # get the trace position for order zero
        posmax, throughput = self.get_trace_pos(order_num=1, round_pos=False)
        # remove the mean from the posmax
        posmax -= np.nanmean(posmax)
        # copy the diff array
        diff2 = np.array(diff)
        # ---------------------------------------------------------------------
        # xpixel values
        ypix = np.arange(diff.shape[0])
        # loop around the array and spline the diff - posmax onto diff2
        for ix in range(diff.shape[1]):
            # get the valid values
            valid = np.isfinite(diff[:, ix])
            # if there are less than 50% good pixels skip
            if np.mean(valid) < 0.5:
                continue
            # get the spline
            spline = ius(ypix[valid] - posmax[ix], diff[:, ix][valid],
                         k=3, ext=1)
            # update the diff2
            diff2[:, ix] = spline(ypix - posmax[ix])
        # ---------------------------------------------------------------------
        # apply a low pass filter to the diff2
        for iy in range(diff.shape[0]):
            diff2[iy] = mp.lowpassfilter(diff2[iy])
        # ---------------------------------------------------------------------
        # spline the values again
        for ix in range(diff.shape[1]):
            # get the spline
            spline = ius(ypix + posmax[ix], diff2[:, ix], k=3, ext=1)
            # update the diff2
            diff2[:, ix] = spline(ypix)
        # ---------------------------------------------------------------------
        # remove the diff2 from diff
        diff -= diff2
        # ---------------------------------------------------------------------
        # take of the median of each row
        for ix in range(nbxpix):
            diff[:, ix] -= np.nanmedian(diff[:, ix])
        # ---------------------------------------------------------------------
        # work out the sigma away from median
        nsig = np.array(diff)
        # loop around each row and remove the low pass
        for iy in tqdm(range(diff.shape[0])):
            nsig[iy] /= mp.lowpassfilter(np.abs(diff[iy]))
        # ---------------------------------------------------------------------
        # we look for a consistent set of >1 sigma pixels
        sig_mask = nsig > 1

        sig_mask2 = np.zeros_like(sig_mask)
        # apply a clustering algorithm to the mask
        all_labels = measure.label(sig_mask, connectivity=2)
        # get a list of unique labels
        unique_labels = np.unique(all_labels)
        # loop around cluster (labels) and find clusters with at least
        #   100 points
        for ulabel in tqdm(unique_labels, leave=False):
            # find all points that are in this cluster (label)
            good = all_labels == ulabel
            # filter out if the first element is False
            #  (quicker than calculating the sum)
            if not sig_mask[good][0]:
                continue
            # if clusters that have more than 100 points
            if np.sum(good) > 100:
                sig_mask2[good] = True
        # ---------------------------------------------------------------------
        # get a circular mask for a bianry dilation
        circle = np.sqrt(np.nansum((np.indices([7, 7]) - 3.0) ** 2, axis=0))
        # apply the circular mask to a binary dilation of the sig_mask2
        sig_mask = binary_dilation(sig_mask2, circle < 3.5)
        # ---------------------------------------------------------------------
        # binary dilate the mask with a box
        box = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        bdilate = binary_dilation(sig_mask, structure=box)
        # ---------------------------------------------------------------------
        # find the positions where the mask is True (with a box of 3 pixels)
        ypos, xpos = np.where(bdilate)
        # ---------------------------------------------------------------------
        # plot this relation
        plots.mask_order0_plot(self, diff, sig_mask)
        # ---------------------------------------------------------------------
        # return the mask trace positions, x positions and y positions
        return sig_mask, xpos, ypos



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
