#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import copy
from typing import Any, Dict

from sossisse.core import base

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
