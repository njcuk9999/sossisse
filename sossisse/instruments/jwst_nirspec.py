#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
import os
from typing import Tuple, Union

import h5py
import numpy as np
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from aperocore import math as mp

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.instruments import default

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
        self.params['WLC.GENERAL.MASK_ORDER_ZERO'] = False
        # for NIRSPEC PRISM we shouldn't have recenter trace position
        self.params['WLC.GENERAL.RECENTER_TRACE_POSITION'] = False

    def get_trace_positions(self, log: bool = True) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map 
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_trace_positions()'

        gen_params = self.params.get('GENERAL')
        wlc_gen_params = self.params.get('WLC.GENERAL')
        # must have pos file defined in file
        if gen_params['POS_FILE'] is None:
            emsg = f'POS_FILE must be defined for {func_name}'
            raise exceptions.SossisseConstantException(emsg)
        # deal with no trace pos file for prism
        if not os.path.exists(gen_params['POS_FILE']):
            # log that we don't have a POS_FILE and are creating one
            if log:
                msg = 'No POS_FILE defined for mode={0} - creating trace map'
                margs = [self.name]
                misc.printc(msg.format(*margs), msg_type='warning')
            # get trace offset in x and y
            xoffset = wlc_gen_params['X_TRACE_OFFSET']
            yoffset = wlc_gen_params['Y_TRACE_OFFSET']
            # get the wave grid from the parameters
            xpix, wavegrid = self.get_wavegrid(source='params',
                                               return_xpix=True)
            # get the median of the cleand data (patch isolated bads)
            clean_cube = self.get_variable('TEMP_CLEAN_NAN', func_name)
            med = np.nanmedian(clean_cube, axis=0)
            # get the indices of the median image
            pixy, pixx = np.indices(med.shape)
            # calculate the sum and running sum of the median image
            s1 = np.nansum(pixy * med, axis=0)
            s2 = np.nansum(med, axis=0)
            # get the indices along the x direction
            index = np.arange(med.shape[1])
            # make a mask for values which will be considered as part of
            #    the trace
            is_flux = s2 > np.nanpercentile(s2, 95) / 5
            # fit the indices robustly (with a quadratic
            tfit, tmask = mp.robust_polyfit(index[is_flux],
                                            s1[is_flux] / s2[is_flux],
                                            degree=2, nsigcut=4)
            # push this into a trace map table
            tracetable = Table()
            tracetable['X'] = index
            tracetable['Y'] = np.polyval(tfit + yoffset, index + xoffset)
            tracetable['WAVELENGTH'] = np.full(len(index), np.nan)
            # push in the wave sol
            valid_wave = np.array(xpix, dtype=int)
            tracetable[valid_wave] = wavegrid
            # print that we are writing pos file
            if log:
                msg = 'Writing POS_FILE={0}'
                margs = [self.params['GENERAL.POS_FILE']]
                misc.printc(msg.format(*margs), msg_type='info')
            # write trace file
            tracetable.write(self.params['GENERAL.POS_FILE'], overwrite=True)
        # ---------------------------------------------------------------------
        # get the trace positions from the white light curve
        tracemap, _ = self.get_trace_pos(map2d=True, order_num=1)
        # return the trace positions
        return tracemap

    def get_wavegrid(self, source: str ='pos',
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
        # get wave file definition
        wave_file = self.params['GENERAL.WAVE_FILE']
        # get pos file definition
        pos_file = self.params['GENERAL.POS_FILE']
        # deal with wave grid coming from params
        if source == 'params' and wave_file is not None:
            wavepath = os.path.join(self.params['PATHS.CALIBPATH'],
                                    wave_file)
            hf = h5py.File(wavepath, 'r')
            xpix, wave = hf['x'], hf['wave_1d']
            # set up a wave vector across the x direction
            wavevector = np.full(xsize, np.nan)
            # push values into wave vector at correct position
            wavevector[np.array(xpix, dtype=int)] = wave
        # deal with case where we need WAVE_FILE and it is not given
        elif source == 'params' and wave_file is None:
            emsg = f'WAVE_FILE must be defined for {func_name}'
            raise exceptions.SossisseInstException(emsg, self.name)
        # deal with case where we need POS_FILE and it is not given
        elif pos_file is None:
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
            tbl_ref = self.load_table(pos_file, ext=order_num)
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
