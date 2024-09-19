#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-20 at 09:56

@author: cook
"""
import datetime
import glob
import hashlib
import os
import shutil
import time
from string import Template
from typing import Any, Dict, List, Union

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
# Define max counter
MAX_LOCK_WAIT = 30
LOCK_WAIT = 5

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


def get_hash(string_variable: str) -> str:
    """
    Get the hash of a string variable

    :param string_variable: str, the string to hash

    :return: str, the hashed string
    """
    # get the hash of the string
    return hashlib.sha256(string_variable.encode('utf-8')).hexdigest()


def lock_wait(filename: str, unlock: bool = False):
    # get a lock file name
    lockfile = filename + '.lock'
    # unlock the lock file
    if unlock:
        os.remove(lockfile)
        return
    # get a counter
    counter = 0
    # -------------------------------------------------------------------------
    # wait for the counter to excess limit or for the lock file to be removed
    while os.path.exists(lockfile) or counter == MAX_LOCK_WAIT:
        time.sleep(LOCK_WAIT)
        msg = 'Waiting for lock file to be removed: {0} [{1}/{2}]'
        margs = [lockfile, counter+1, MAX_LOCK_WAIT]
        misc.printc(msg.format(*margs), msg_type='alert')
    # -------------------------------------------------------------------------
    # Deal with the case where the lock file still exists
    if MAX_LOCK_WAIT == counter:
        emsg = ('Lock file still exists after {0} seconds. '
                'Please remove {1} manually.')
        eargs = [LOCK_WAIT * MAX_LOCK_WAIT, lockfile]
        raise SossisseIOException(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # create the lock file
    with open(lockfile, 'w') as f:
        f.write('lock')


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
        data = fits.getdata(filename, ext=ext, extname=extname)
    except Exception as e:
        emsg = 'Error loading data from file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    # return data
    return data

def load_table(filename: str, fmt: Union[int, str] = None,
               hdu: Union[int, str] = None):
    """
    Load the table from a file

    :param filename: str, the filename to load
    :param fmt: str, the format to load the table in
    :param hdu: int or str, the extension number or extention name to load

    :return: data, the loaded data
    """
    try:
        if fmt == 'fits' or hdu is not None:
            table = Table.read(filename, format=fmt, hdu=hdu)
        else:
            table = Table.read(filename, format=fmt)
    except Exception as e:
        emsg = 'Error loading table from file: {0}\n\t{1}: {2}'
        eargs = [filename, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    return table


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


def save_fits(filename: str, datalist: List[Any], datatypes: List[str],
              datanames: List[str], meta: Dict[str, Any] = None):
    """
    Save the data to a file
    :param filename:
    :param datalist:
    :param datatypes:
    :param datanames:
    :param meta:
    :return:
    """
    # print progres
    msg = 'Saving data to file: {0}'
    misc.printc(msg.format(filename), msg_type='info')

    try:
        # open a fits hdu list
        with fits.HDUList() as hdul:
            # first hdu is primary and is blank
            hdul.append(fits.PrimaryHDU())
            # loop around all data
            for it in range(len(datalist)):
                # deal with differing datatypes
                if datatypes[it] == 'table':
                    hdu = fits.BinTableHDU(datalist[it], name=datanames[it])
                else:
                    # create a hdu
                    hdu = fits.ImageHDU(datalist[it], name=datanames[it])
                # add the hdu to the hdul
                hdul.append(hdu)
            # add the meta data
            if meta is not None:
                for key in meta:
                    hdul[0].header[key] = meta[key]
            # write the hdul to a file
            hdul.writeto(filename, overwrite=True)
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
    csv_files = glob.glob(os.path.join(params['OTHER_PATH'], '*.csv'))
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
    rendered_html = template.safe_substitute(data)
    # construct filename for html file
    html_file = os.path.join(params['PLOT_PATH'], 'index.html')
    # write the html file
    with open(html_file, 'w') as html_file:
        html_file.write(rendered_html)


    # noinspection PyUnresolvedReferences


def save_eureka(filename: str, flux: np.ndarray, flux_err: np.ndarray,
                wavegrid: np.ndarray, time_arr: np.ndarray):
    # set function name
    func_name = __NAME__ + '.save_eureka()'
    # attempt to install astraeus package
    try:
        from astraeus import xarrayIO as xrio
    except ImportError:
        emsg = ('Please install the astraeus package manually to use '
                f'"{func_name}" with the following: \n'
                'pip install git+https://github.com/kevin218/Astraeus.git')
        raise exceptions.SossisseException(emsg)
    # -----------------------------------------------------------------
    # flux and flux_error should be of shape time x wavelength
    # wavelength should be in descending order (and in microns)
    # time in BJD_TBD
    # outfile is the name you want to save your file as
    # -----------------------------------------------------------------
    # make h5 data set
    outdata = xrio.makeDataset()
    # add the spectrum extension
    outdata['optspec'] = (['time', 'x'], flux)
    outdata['optspec'].attrs['flux_units'] = 'e-/s'
    outdata['optspec'].attrs['time_units'] = 'BJD_TBD'
    # add the error extension
    outdata['opterr'] = (['time', 'x'], flux_err)
    outdata['opterr'].attrs['flux_units'] = 'e-/s'
    outdata['opterr'].attrs['time_units'] = 'BJD_TBD'
    # add the wave extension
    outdata['wave_1d'] = (['x'], wavegrid)
    # add the time array
    outdata.coords['time'] = time_arr
    # write file
    xrio.writeXR(filename, outdata)


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
