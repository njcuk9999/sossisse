#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-20 at 09:56

@author: cook
"""
import os
import shutil
from typing import Any, Dict, Tuple, Union

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.exceptions'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
SossisseIOException = exceptions.SossisseIOException


# =============================================================================
# Define functions
# =============================================================================
def create_directory(path: str):
    """
    Create a directory if it does not already exist

    :param path: str, the path to create

    :raises SossisseIOException: if the path cannot be created
    :return: None, creates directory if path does not exist
    """
    # check if path already exists (in which case we do nothing)
    if os.path.exists(path):
        return
    # try to create the path
    try:
        # print that we are creating path
        misc.printc('Creating directory: {0}'.format(path), msg_type='setup')
        # make the path
        os.makedirs(path)
    except Exception as e:
        emsg = 'Cannot create directory: {0}\n\t{1}: {2}'
        eargs = [path, type(e), str(e)]
        raise SossisseIOException(emsg.format(*eargs))


def get_file(path: Union[str, None], name: str,
             required: bool = True) -> Union[str, None]:
    """
    Get a file from a path

    :param path: str, the path to the file

    :raises SossisseIOException: if the file cannot be found
    :return: str, the path to the file
    """
    # check if user has defined file
    if str(path).upper() in ['', 'NULL', 'NONE']:
        return None
    # check if path is required
    if not required:
        return path
    # check if path already exists
    if os.path.exists(path):
        msg = 'File for "{0}" exists: {1}. Will use {0} file.'
        args = [name, path]
        misc.printc(msg.format(*args), msg_type='debug')
        return path
    else:
        wmsg = 'File for "{0} does not exist: {1}. Will not use {0} file.'
        wargs = [name, path]
        misc.printc(wmsg.format(*wargs), msg_type='warning')
        return None


def copy_file(inpath: str, outpath: str):
    """
    Copy a file

    :param path: str, the path to the file

    :raises SossisseIOException: if the file cannot be copied
    :return: None, copies the file
    """
    # check if path already exists
    if os.path.exists(outpath):
        os.remove(outpath)
    # try to copy the file
    try:
        # print that we are copying file
        msg = 'Copying file: {0}-->{1}'
        margs = [inpath, outpath]
        misc.printc(msg.format(*margs), msg_type='debug')
        # copy the file
        shutil.copyfile(inpath, outpath)
    except Exception as e:
        emsg = 'Cannot copy file: {0}-->{1}\n\t{2}: {3}'
        eargs = [inpath, outpath, type(e), str(e)]
        raise SossisseIOException(emsg.format(*eargs))


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
