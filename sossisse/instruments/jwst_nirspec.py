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
class JWST_NIRSPEC(default.Instrument):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC'
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
            xpix, wavegrid = self.get_wavegrid(return_xpix=True)
            # get the median of the cleand data (patch isolated bads)
            clean_cube = self.get_variable('TEMP_CLEAN_NAN', func_name)
            # load the data in the clean cube
            clean_cube_fits = self.load_data(clean_cube)
            # take the median of the clean cube
            med = np.nanmedian(clean_cube_fits, axis=0)
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
            tracetable['WAVELENGTH'][valid_wave] = wavegrid[valid_wave]
            # print that we are writing pos file
            if log:
                msg = 'Writing POS_FILE={0}'
                margs = [gen_params['POS_FILE']]
                misc.printc(msg.format(*margs), msg_type='info')
            # write trace file
            tracetable.write(gen_params['POS_FILE'], overwrite=True)
        # ---------------------------------------------------------------------
        # get the trace positions from the white light curve
        tracemap, _ = self.get_trace_pos(map2d=True, order_num=1)
        # return the trace positions
        return tracemap


class JWST_NIRSPEC_PRISM(JWST_NIRSPEC):
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


class JWST_NIRSPEC_GRATING(JWST_NIRSPEC):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_GRATING, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.GRATING'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G395(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G395, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.G395'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G235(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G235, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.G235'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G140(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G140, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.G140'
        # set up the instrument
        self.param_override()



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
