#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2025-03-25 at 09:57

@author: cook
"""
from sossisse.core import base

# =============================================================================
# Define storage
# =============================================================================
# fill this with the instrument modes
DEMOS = dict()
for instrument in base.INSTRUMENTS:
    DEMOS[instrument] = dict()

# =============================================================================
# Define usable demos
# =============================================================================
# # JWST.NIRISS.SOSS Trappist-1 demo
# DEMOS['JWST.NIRISS.SOSS']['TRAPPIST 1b'] = niriss_soss_trappist1b
#
# # JWST.NIRSPEC.BOTS.PRISM WASP 39b
# DEMOS['JWST.NIRSPEC.BOTS.PRISM']['WASP 39b'] = nirspec_prism_wasp39b
#
# # JWST.NIRSPEC.BOTS.G395 WASP 39b
# DEMOS['JWST.NIRSPEC.BOTS.G395']['WASP 39b'] = nirspec_g395_wasp39b


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
