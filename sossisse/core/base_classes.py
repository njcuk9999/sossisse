#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:15

@author: cook

Rules: only import from sossisse.base and sossisse.exceptions
"""
from typing import Any, List, Type, Union

from sossisse.core import base
from sossisse.core import exceptions

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.base_classes'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# -----------------------------------------------------------------------------
SossisseConstantException = exceptions.SossisseConstantException


# =============================================================================
# Define functions
# =============================================================================
class Const:
    """
    Define a constant for use in loading SOSSICE variables
    """

    def __init__(self, name: str, value: Any = None, dtype: Type = None,
                 dtypei: Type = None, required: bool = False,
                 minimum: Union[int, float] = None,
                 maximum: Union[int, float] = None,
                 options: List[Any] = None, length: int = None,
                 comment: str = None, active: bool = False,
                 modes: str = None):
        """
        Construct the constant class

        :param name: str, the name of the constant
        :param value: Any, the value of the constant
        :param dtype: Type, the type of the constant
        :param dtypei: Type, the type of the constant element (for list or dict)
        :param required: bool, whether the constant is required
        :param minimum: int/float, the minimum value for the constant
        :param maximum: int/float, the maximum value for the constant
        :param options: list, the options for the constant
        :param length: int, the length of the constant (for list only)
        :param comment: str, a comment for the constant (for yaml creation)
                        if not given, constant is never in parameter file
        :param active: bool, whether the constant is active (for yaml creation)
                       if False not included in yaml file
        :param modes: str or None, if set only included in yamls for this mode
        """
        self.name: str = name
        self.value: Any = value
        self.dtype: Type = dtype
        self.dtypei: Type = dtypei
        self.required: bool = required
        self.minimum: Union[int, float] = minimum
        self.maximum: Union[int, float] = maximum
        self.options: List[Any] = options
        self.length: int = length
        self.comment: str = comment
        self.active: bool = active
        self.modes: str = modes

    def verify(self, name: str = None, value: Any = None, dtype: Type = None,
               dtypei: Type = None, source: str = None) -> bool:
        """
        Verify the constant is valid

        :param name: str, the name of the constant
        :param value: Any, the value of the constant
        :param dtype: Type, the type of the constant
        :param dtypei: Type, the type of the constant element (for list or dict)
        :param source: str, the source of the constant

        :raises: exceptions.SossisseConstantException, if constant is invalid
        :return: bool, True if constant is valid
        """
        # deal with no name
        if name is None:
            name = self.name
        # deal with no dtype set
        if dtype is None:
            dtype = self.dtype
        # deal with still having no dtype --> skip check
        if dtype is None:
            return True
        # ---------------------------------------------------------------------
        # deal with no value (get default value)
        if value is None:
            value = self.value
        # ---------------------------------------------------------------------
        # deal with required check
        # ---------------------------------------------------------------------
        # check if value is required make sure its not None
        if self.required and value is None:
            emsg = 'Constant {0}: Value={1} is required'
            eargs = [name, value]
            # deal with having a source
            if source is not None:
                emsg += ' (from {2})'
                eargs.append(source)
            # raise the exception
            raise SossisseConstantException(emsg.format(*eargs))
        elif value is None:
            self.value = value
            return True
        # ---------------------------------------------------------------------
        # deal with value as a list
        # ---------------------------------------------------------------------
        if isinstance(value, (list, dict)) and self.dtype in [list, dict]:
            # we iterate differently for lists and dicts
            if isinstance(value, dict):
                items = list(value.keys())
            else:
                items = list(range(len(value)))
            # loop around items and check them
            for item in items:
                # update the name and value (for sub-test)
                _name = f'{name}[{item}]'
                _value = value[item]
                # only test if dtypei is a basic type
                if dtypei in base.BASIC_TYPES:
                    # create a fake constant to test element
                    _const = Const(_name, _value, dtypei, dtypei=None,
                                   required=self.required, minimum=self.minimum,
                                   maximum=self.maximum, options=self.options)
                    # verify this constant
                    _const.verify()
            # deal with length check on lists
            if isinstance(value, list) and self.length is not None:
                if len(value) != self.length:
                    emsg = ('Constant {0}: Value={1} has length {2} not '
                            'length {3}')
                    eargs = [name, value, len(value), self.length]
                    # deal with having a source
                    if source is not None:
                        emsg += ' (from {4})'
                        eargs.append(source)
                    # raise the exception
                    raise SossisseConstantException(emsg.format(*eargs))
            # if we get here we have validated our constant
            self.value = value
        # ---------------------------------------------------------------------
        # deal with int/float/bool/str
        # ---------------------------------------------------------------------
        elif dtype in [int, float, bool, str]:
            # -----------------------------------------------------------------
            # check data type
            # -----------------------------------------------------------------
            # try to cast value by dtype
            try:
                self.value = dtype(value)
            except TypeError:
                emsg = 'Constant {0}: Value={1} is not type {2}'
                eargs = [name, value, dtype]
                # deal with having a source
                if source is not None:
                    emsg += ' (from {3})'
                    eargs.append(source)
                # raise the exception
                raise SossisseConstantException(emsg.format(*eargs))
            # -----------------------------------------------------------------
            # deal with int/float checks
            # -----------------------------------------------------------------
            if isinstance(self.value, (int, float)):
                # deal with minimum value criteria (test value)
                if self.minimum is not None:
                    if self.value < self.minimum:
                        emsg = ('Constant {0}: Value={1} is less than '
                                'minimum={2}')
                        eargs = [name, value, self.minimum]
                        # deal with having a source
                        if source is not None:
                            emsg += ' (from {3})'
                            eargs.append(source)
                        # raise the exception
                        raise SossisseConstantException(emsg.format(*eargs))
                # deal with maximum value criteria set (test value)
                if self.maximum is not None:
                    if self.value > self.maximum:
                        emsg = ('Constant {0}: Value={1} is greater than '
                                'maximum={2}')
                        eargs = [name, value, self.maximum]
                        # deal with having a source
                        if source is not None:
                            emsg += ' (from {3})'
                            eargs.append(source)
                        # raise the exception
                        raise SossisseConstantException(emsg.format(*eargs))
            # -----------------------------------------------------------------
            # deal with options
            # -----------------------------------------------------------------
            if self.options is not None:
                if value not in self.options:
                    emsg = 'Constant {0}: Value={1} is not in options={2}'
                    eargs = [name, value, self.options]
                    # deal with having a source
                    if source is not None:
                        emsg += ' (from {3})'
                        eargs.append(source)
                    # raise the exception
                    raise SossisseConstantException(emsg.format(*eargs))
        # ---------------------------------------------------------------------
        # if we get here we have validated our constant
        return True


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
