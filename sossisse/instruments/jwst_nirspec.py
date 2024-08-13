#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
from sossisse.core import base
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
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        pass


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
