#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on {DATE}

@author: cook

Rules: only import from sossisse.base
"""
from sossisse.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.exceptions'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__

# =============================================================================
# Define functions
# =============================================================================
class SossisseException(Exception):
    """
    Basic Sossisse Exception
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'SossisseException: {0}'.format(self.message)


class SossisseConstantException(SossisseException):
    """
    Exception from a constant
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'SossisseConstantException: {0}'.format(self.message)


class SossisseFileException(SossisseException):
    """
    Exception from a constant
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'SossisseConstantException: {0}'.format(self.message)


class SossisseIOException(SossisseException):
    """
    Exception from a constant
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'SossisseIOException: {0}'.format(self.message)


class SossisseInstException(SossisseException):
    """
    Exception from a constant
    """
    def __init__(self, message, classname: str):
        super().__init__(message)
        self.classname = classname
        self.message = message

    def __str__(self) -> str:
        return 'SossisseInstException[{0}]: {1}'.format(self.classname,
                                                        self.message)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
