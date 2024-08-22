#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
import os
from typing import Union

import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from sossisse.core import base
from sossisse.instruments import default
from sossisse.core import exceptions


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.jwst_nirspec'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
class JWST_NIRSPEC_PRISM(default.Instrument):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_PRISM, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.PRISM'
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        # run the super version first, then override after
        super().param_override()
        # we do not mask order zero for NIRSPEC PRISM
        self.params['MASK_ORDER_ZERO'] = False
        # for NIRSPEC PRISM we shouldnot have recenter trace position
        self.params['RECENTER_TRACE_POSITION'] = False

    def get_trace_positions(self) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map 
        """
        # deal with no trace pos file for prism
        if not os.path.exists(self.params['POS_FILE']):
            wavegrid = self.get_wavegrid(source='params')
        
        # get the trace positions from the white light curve
        tracemap, _ = self.get_trace_pos(map2d=True, order_num=1)
        # return the trace positions
        return tracemap

    def get_wavegrid(self, source: str ='pos',
                     order_num: Union[int, None] = None):
        """
        Get the wave grid for the instrument

        :param source: str, the source of the wave grid
        :return: np.ndarray, the wave grid
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_wavegrid()'
        # get x size from cube
        xsize = self.get_variable('DATA_X_SIZE', func_name)
        # deal with wave grid coming from params
        if source == 'params' and self.params['WAVE_FILE'] is not None:
            wavepath = os.path.join(self.params['CALIBPATH'],
                                    self.params['WAVE_FILE'])
            hf = h5py.File(wavepath, 'r')
            xpix, wave = hf['x'], hf['wave_1d']
            # set up a wave vector across the x direction
            wavevector = np.full(xsize, np.nan)
            # push values into wave vector at correct position
            wavevector[np.array(xpix, dtype=int)] = wave
        # deal with case where we need WAVE_FILE and it is not given
        elif source == 'params' and self.params['WAVE_FILE'] is None:
            emsg = f'WAVE_FILE must be defined for {func_name}'
            raise exceptions.SossisseInstException(emsg, self.name)
        # deal with case where we need POS_FILE and it is not given
        elif self.params['POS_FILE'] is None:
            emsg = f'POS_FILE must be defined for {func_name}'
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
