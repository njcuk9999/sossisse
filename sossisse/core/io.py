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
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from aperocore.constants.param_functions import ParamDict
from aperocore.constants.param_functions import SubParamDict
from aperocore.constants.constant_functions import ConstantsDict

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


def load_fits(filename: str, ext: int = None, extname: str = None,
              hdufix: bool = False):
    """
    Load the data from a file

    :param filename: str, the filename to load
    :param ext: int, the extension number to load
    :param extname: str, the extension name to load

    :return: data, the loaded data
    """
    # try to get data from filename
    try:
        if hdufix:
            with fits.open(filename) as hdul:
                # try to fix the data
                hdul.verify('fix')
                if ext is not None:
                    return np.array(hdul[ext].data)
                elif extname is not None:
                    return np.array(hdul[extname].data)
                else:
                    data = fits.getdata(filename, ext=ext, extname=extname)
        else:
            data = fits.getdata(filename, ext=ext, extname=extname)
    except Exception as _:
        try:
            load_fits(filename, ext, extname, hdufix=True)
        except Exception as e:
            emsg = 'Error loading data from file: {0}\n\t{1}: {2}'
            eargs = [filename, type(e), str(e)]
            raise exceptions.SossisseFileException(emsg.format(*eargs))
    # return data
    return np.array(data)


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


def save_table(filename: str, data: Table, fmt: str = 'csv'):
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


def info_html(params: Dict[str, Any]):

    # get time now
    timenow = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # get version
    version = base.__version__
    vdate = base.__date__
    # push into html
    html = '<ul>'
    html += f'<li><b>Reduction date</b>: {timenow}</li>'
    html += f'<li><b>SOSSISSE Version</b>: {version}</li>'
    html += f'<li><b>SOSSISSE date</b>: {vdate}</li>'
    html += '</ul>'
    return html


def params_to_html(params: Dict[str, Any], html: str = None):
    """
    Convert a dictionary of parameters to an html list
    in the form of a bulleted list as follows

    - key: value

    :param params: dict, the dictionary of parameters

    :return: str, the html string
    """
    # define the html string
    html = '<ul>'

    # sort keys alphabetically
    keys = np.sort(params.keys())
    # loop around all parameters
    for key in keys:
        # do not add these keys
        if key.startswith('DRS.'):
            continue
        if key.startswith('LOG.'):
            continue
        if key == 'RECIPE_SHORT':
            continue
        # get the raw value
        rawvalue = params[key]
        # ignore param dicts
        if isinstance(rawvalue, (ParamDict, SubParamDict, ConstantsDict)):
            continue
        # deal with lists
        if isinstance(rawvalue, (list, np.ndarray)):
            # quick check for valid sub-list data
            subvalues = []
            for val in rawvalue:
                if isinstance(val, (ParamDict, SubParamDict, ConstantsDict)):
                    continue
                else:
                    subvalues.append(val)
            # deal with no valid values
            if len(subvalues) == 0:
                continue
            # create a sub-list
            html += f'<li><b>{key}</b>:<ul>'
            for val in subvalues:
                html += f'<li> - {val}</li>'
            # end the list
            html += '</ul></li>'
        # deal with dictionaries
        elif isinstance(rawvalue, dict):
            # quick check for valid sub-dict data
            subvalues = dict()
            for subkey in rawvalue:
                val = rawvalue[subkey]
                if isinstance(val, (ParamDict, SubParamDict, ConstantsDict)):
                    continue
                else:
                    subvalues[subkey] = val
            # deal with no valid values
            if len(subvalues) == 0:
                continue
            # create a sub-list
            html += f'<li><b>{key}</b>:<ul>'
            for subkey in subvalues:
                html += f'<li> - {subkey}: {rawvalue[subkey]}</li>'
            # end the list
            html += '</ul></li>'
        # else convert to string
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
    png_files = glob.glob(os.path.join(params['PATHS.PLOT_PATH'], '*.png'))
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
    csv_files = glob.glob(os.path.join(params['PATHS.OTHER_PATH'], '*.csv'))
    # define the html string
    html = '<br><ul>'
    # loop around all csv files
    for csv_file in csv_files:
        # get the filename
        filename = os.path.basename(csv_file)
        # create the html string
        html_list = '<li><a href="{0}">{0}</a></li>'
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
    # get the objectname
    objname = params['INPUTS.OBJECTNAME']
    imode = params['INPUTS.INSTRUMENTMODE']
    # define the data to insert into the template
    data = dict()
    # add the data to the dictionary for the html sections
    targs = [objname, imode]
    data['title'] = 'SOSSISSE Summary: <br> {0} [{1}]'.format(*targs)
    data['info'] = info_html(params)
    data['plots'] = plots_to_html(params)
    data['csvlist'] = csv_to_html(params)
    data['paramlist'] = params_to_html(params)
    # Substitute variables in the template
    rendered_html = template.safe_substitute(data)
    # construct filename for html file
    html_file = os.path.join(params['PATHS.PLOT_PATH'], 'index.html')
    # write the html file
    with open(html_file, 'w') as html_file:
        html_file.write(rendered_html)


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
    outdata['wave_1d'].attrs['wave_units'] = 'micron'
    # add the time array
    outdata.coords['time'] = time_arr
    # write file
    xrio.writeXR(filename, outdata, verbose=False)



# =============================================================================
# Wavelength loading functions
# =============================================================================
def load_wave_ext1d(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wave from a jwst data model of the x1dints file

    :param filepath: str, path to hdf5 file

    :return: tuple, 1. np.ndarray x pixel values, 2. np.ndarray wavelength
         values for each pixel
    """
    # try to import jwst (maybe user didn't install)
    try:
        from jwst import datamodels
    except ImportError:
        emsg = ('Please install the jwst package to use WAVE_TYPE=ext1d')
        raise exceptions.SossisseException(emsg)
    # -------------------------------------------------------------------------
    # try to open the file
    try:
        fileobj = datamodels.open(filepath)
    except Exception as e:
        emsg = 'Could not open WAVE from EXT1D file: {0}\n\t{1}:{2}'
        eargs = [filepath, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    # -------------------------------------------------------------------------
    # try to load wavelength
    try:
        wavevector = fileobj.spec[0].spec_table['wavelength']
    except Exception as e:
        emsg = 'Could not read wavelength from EXT1D file: {0}\n\t{1}:{2}'
        eargs = [filepath, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    # get xpix
    xpix = np.arange(len(wavevector))
    # return the xpix and wave vector
    return xpix, wavevector


def load_wave_fits(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wave from FITS file.

    :param filepath: str, path to hdf5 file

    :return: tuple, 1. np.ndarray x pixel values, 2. np.ndarray wavelength
         values for each pixel
    """
    # try to load the wave table
    try:
        wave_table = Table.read(filepath)
    except Exception as e:
        emsg = 'Could not load WAVE FITS file: {0}\n\t{1}: {2}'
        eargs = [filepath, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    # try to get the vectors
    try:
        xpix = np.array(wave_table['xpix'])
        wavevector = np.array(wave_table['wavevector'])
    except Exception as e:
        emsg = ('FITS file: {0} in wrong format (requires xpix and '
                'wavevector columns)'.format(filepath))
        raise exceptions.SossisseFileException(emsg)
    # return the xpix and wave vector
    return xpix, wavevector


def load_wave_hdf5(filepath: str, xsize: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wave from HDF5 file.

    :param filepath: str, path to hdf5 file
    :param xsize: int, size of x axis

    :return: tuple, 1. np.ndarray x pixel values, 2. np.ndarray wavelength
         values for each pixel
    """
    # try to load the hdf5 file
    try:
        hf = h5py.File(filepath, 'r')
    except Exception as e:
        emsg = 'Could not open HDF5 file: {0}\n\t{1}:{2}'
        eargs = [filepath, type(e), str(e)]
        raise exceptions.SossisseFileException(emsg.format(*eargs))
    # try to get the vectors
    try:
        xpix, wave = hf['x'], hf['wave_1d']
    except Exception as e:
        emsg = ('HDF5 file: {0} in wrong format (requires x and '
                'wave_1d vectors)'.format(filepath))
        raise exceptions.SossisseFileException(emsg)
    # set up a wave vector across the x direction
    wavevector = np.full(xsize, np.nan)
    # push values into wave vector at correct position
    wavevector[np.array(xpix, dtype=int)] = wave
    # return the xpix and wave vector
    return xpix, wavevector


def load_wave_posfile(tbl_ref: Table, xsize, xtraceoffset: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wavevector from position file

    :param tbl_ref: Table, the position file astropy.table.Table
    :param xsize: int, size of the wavevector expected
    :param xtraceoffset: int, offset of the wavevector

    :return: tuple, 1. np.ndarray x pixel values, 2. np.ndarray wavelength
             values for each pixel
    """
    # get the valid pixels
    valid = tbl_ref['X'] > 0
    valid &= tbl_ref['X'] < xsize - 1
    valid &= np.isfinite(np.array(tbl_ref['WAVELENGTH']))
    # mask the table by these valid positions
    tbl_ref = tbl_ref[valid]
    # sort by the x positions
    tbl_ref = tbl_ref[np.argsort(tbl_ref['X'])]
    # spline the wave grid
    spl_wave = ius(tbl_ref['X'], tbl_ref['WAVELENGTH'], ext=1, k=1)
    # push onto our wave grid
    wavevector = spl_wave(np.arange(xsize) - xtraceoffset)
    # deal with zeros
    wavevector[wavevector == 0] = np.nan
    # get xpix
    xpix = np.arange(xsize)
    # return the xpix and wave vector
    return xpix, wavevector



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
