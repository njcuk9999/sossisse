#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.base.py

base variables here

Created on 2022-09-20

@author: cook

Rules: no sosssise imports
"""
from pathlib import Path

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.base'

__PATH__ = Path(__file__).parent.parent
with open(__PATH__.parent.joinpath('version.txt'), 'r') as vfile:
    vtext = vfile.readlines()

__version__ = vtext[0].strip()
__date__ = vtext[1].strip()
__authors__ = 'Etienne Artigau, Neil James Cook, Loic Albert'

# Define basic types (non nested)
BASIC_TYPES = (int, float, bool, str)

# Define console width
CONSOLE_WIDTH = 120

# =============================================================================
# Define functions
# =============================================================================

# =============================================================================
# End of code
# =============================================================================