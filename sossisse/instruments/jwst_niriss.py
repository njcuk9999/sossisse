#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
from typing import Any, Dict, Tuple, Union

import numpy as np

from sossisse.core import base
from sossisse.instruments import default
from sossisse.core import exceptions
from sossisse.core import misc

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
class JWST_SOSS(default.Instrument):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_SOSS, self).__init__(params)
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        pass

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
            if flat.shape == (2024, 2048):
                # cut down the flat field
                flat = flat[-256:]
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
