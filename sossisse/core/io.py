#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-20 at 09:56

@author: cook
"""
import datetime
import os
import shutil
from string import Template
from typing import Any, Dict, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table

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

# define html template file path
PACKAGE_PATH = os.path.dirname(os.path.dirname(__file__))
HTML_TEMPLATE_FILE = os.path.join(PACKAGE_PATH, 'resources',
                                  'html_template.html')


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


def load_fits(filename: str, ext: int = None, extname: str = None):
    """
    Load the data from a file

    :param filename: str, the filename to load
    :param ext: int, the extension number to load
    :param extname: str, the extension name to load

    :return: data, the loaded data
    """
    # try to get data from filename
    try:
        data = fits.getdata(filename, ext, extname)
    except Exception as e:
        emsg = 'Error loading data from file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))


def load_table(filename: str, ext: int = None):
    """
    Load the table from a file

    :param filename: str, the filename to load
    :param ext: int, the extension number to load

    :return: data, the loaded data
    """
    try:
        data = Table.read(filename, ext)
    except Exception as e:
        emsg = 'Error loading table from file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))


def save_fitsimage(filename: str, data: np.ndarray, meta: Dict[str, Any] = None):
    """
    Save the data to a file

    :param filename:
    :param data:
    :param meta:
    :return:
    """
    # print progres
    msg = 'Saving data to file: {0}'
    misc.printc(msg.format(filename), msg_type='info')
    # try to save the data
    try:
        fits.writeto(filename, data, meta, overwrite=True)
    except Exception as e:
        emsg = 'Error saving data to file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))


def save_table(filename: str, data: Table, fmt: str='csv'):
    """
    Save the table to a file

    :param filename: str, the filename to save to
    :param data: Table, the astropy table to save
    :param fmt: str, the format to save the table in
    :return:
    """
    # print progres
    msg = 'Saving table to file: {0}'
    misc.printc(msg.format(filename), msg_type='info')
    # try to save the data
    try:
        data.write(filename, format=fmt, overwrite=True)
    except Exception as e:
        emsg = 'Error saving table to file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))


def params_to_html(params: Dict[str, Any]):
    """
    Convert a dictionary of parameters to an html list
    in the form of a bulleted list as follows

    - key: value

    :param params: dict, the dictionary of parameters

    :return: str, the html string
    """
    # define the html string
    html = '<ul>'
    # loop around all parameters
    for key in params:
        # get the raw value
        rawvalue = params[key]
        # make the value a string
        if isinstance(rawvalue, (list, np.ndarray)):
            value = ', '.join([str(val) for val in rawvalue])
        else:
            value = str(rawvalue)
        # create a list item
        html += '<li><b>{0}</b>: {1}</li>'.format(key, value)
    # end the list
    html += '</ul>'
    # return the html string
    return html


def plots_to_html(params: Dict[str, Any]):
    """
    Grab the png files from the PLOT_PATH and push them into html

    :param params:
    :return:
    """
    # get all png files in the plot path
    png_files = glob.glob(os.path.join(params['PLOT_PATH'], '*.png'))
    # define the html string
    html = '<br>'
    # loop around all png files
    for png_file in png_files:
        # get the filename
        filename = os.path.basename(png_file)
        # create the html string
        html_image = '<img src="{0}" alt="{0}" style="width:"600"">'
        html_image += '<br><br><br>'
        html += html_image.format(filename)
    # return the html string
    return html


def csv_to_html(params: Dict[str, Any]):
    """
    Grab the csv files from the PLOT_PATH and push them into html

    :param params:
    :return:
    """
    # get all csv files in the plot path
    csv_files = glob.glob(os.path.join(params['PLOT_PATH'], '*.csv'))
    # define the html string
    html = '<br><ul>'
    # loop around all csv files
    for csv_file in csv_files:
        # get the filename
        filename = os.path.basename(csv_file)
        # create the html string
        html_list = '<li><a href="{0}">{0}</a><br><br><br></li>'
        html += html_list.format(filename)
    # end the list
    html += '</ul>'
    # return the html string
    return html


def summary_html(params: Dict[str, Any]):
    """
    Create a summary html file

    :param params:
    :return:
    """
    # read the html template
    with open(HTML_TEMPLATE_FILE, 'r') as template_file:
        template = Template(template_file.read())
    # define the data to insert into the template
    data = dict()
    # add the data to the dictionary for the html sections
    data['title'] = 'SOSSISSE Summary: {0}'.format(params['OBJECTNAME'])
    data['datenow'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['plots'] = plots_to_html(params)
    data['csvlist'] = csv_to_html(params)
    data['paramlist'] = params_to_html(params)
    # Substitute variables in the template
    rendered_html = template.safe_substitute(variables)
    # construct filename for html file
    html_file = os.path.join(params['PLOT_PATH'], 'index.html')
    # write the html file
    with open(html_file, 'w') as html_file:
        html_file.write(rendered_html)


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
