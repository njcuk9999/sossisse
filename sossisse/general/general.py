#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.general.py

General functionality in here

Created on 2022-09-20

@author: cook
"""
import glob
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from sossisse.core import base

from sossisse.core import math
from sossisse.core import misc
from sossisse.core import io
from sossisse.instruments import Instrument
from sossisse.general import plots


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.misc'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
def white_light_curve(inst: Instrument):
    """
    White light curve functionality

    :param inst: Instrument, the instrument object
    :return:
    """
    # set the function name
    func_name = f'{__NAME__}.white_light_curve'
    # print the splash
    misc.sossart()
    # get parameters from instrumental parameters
    objname = inst.params['OBJECTNAME']
    # print the white light curve splash
    print(misc.art('White light curve ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # load the image, error and data quality
    cube, err = inst.load_data_with_dq()
    # -------------------------------------------------------------------------
    # for each slice of the cube, isolated bad pixels are interpolated with the
    # value of their 4 neighbours.
    cube = inst.patch_isolated_bads(cube)
    # -------------------------------------------------------------------------
    # get the trace map
    tracemap = inst.get_trace_map()
    # -------------------------------------------------------------------------
    # if you want to subtract a higher order polynomial to the 1/f noise, change
    # the value of fit_order
    out_c1f = inst.clean_1f(cube, err, tracemap)
    cube, med, med_diff, transit_invsout, pcas = out_c1f
    # -------------------------------------------------------------------------
    # recenter the trace position
    tracemap = inst.recenter_trace_position(tracemap, med)
    # -------------------------------------------------------------------------
    # Part of the code that does rotation/shift/amplitude
    # -------------------------------------------------------------------------
    # get the gradients
    dx, dy, rotxy, ddy, med_clean = inst.get_gradients(med)
    # set up the mask for trace position
    mask_out = inst.get_mask_trace_pos(med, tracemap)
    mask_trace_pos, x_order0, y_order0, x_trace_pos, y_trace_pos = mask_out
    # setup the linear reconstruction vector based on the input parameters
    lvector = inst.setup_linear_reconstruction(med, dx, dy, rotxy, ddy,
                                               pcas, med_diff)
    # -------------------------------------------------------------------------
    # find the best linear combination of scale/dx/dy/rotation from lvector
    # amps is a vector with the amplitude of all 4 fitted terms
    # amps[0] -> amplitude of trace
    # amps[1] -> dx normalized on reference trace
    # amps[2] -> dy normalized on reference trace
    # amps[3] -> rotation (in radians) normalized on reference trace
    # amps[4] -> 2nd derivative in y [if option activated]
    # -------------------------------------------------------------------------
    amp_out = inst.apply_amp_recon(cube, err, med, mask_trace_pos,
                                   lvector, x_trace_pos, y_trace_pos,
                                   x_order0, y_order0)
    # get outputs of apply_amp_recon
    l_table, lrecon, valid_cube = amp_out
    # -------------------------------------------------------------------------
    # normalize the trace but a normalization factor
    l_table = inst.normalize_sum_trace(l_table)
    # -------------------------------------------------------------------------
    # per pixel baseline
    if inst.params['PER_PIXEL_BASELINE_CORRECTION']:
        misc.printc('Performing per-pixel baseline subtraction', 'info')
        cube = inst.per_pixel_baseline(cube, valid_cube)
    # -------------------------------------------------------------------------
    # print the rms baseline for all methods
    for method in inst.get_rms_baseline():
        # calculate the rms for this method
        rms_method = inst.get_rms_baseline(l_table['amplitude'], method=method)
        # print this
        msg = '{0}, rms = {1:.1f}ppm'.format(method, rms_method * 1e6)
        misc.printc(msg, 'number')
    # -------------------------------------------------------------------------
    # get the effective wavelength
    photo_weighted_mean, energy_weighted_mean = inst.get_effective_wavelength()
    # =========================================================================
    # write files
    # =========================================================================
    # get the meta data
    meta_data = inst.get_variable('META', func_name)
    # -------------------------------------------------------------------------
    # write the error map
    errfile = inst.get_variable('WLC_ERR_FILE', func_name)
    io.save_fitsimage(errfile, err, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the residual map
    resfile = inst.get_variable('WLC_RES_FILE', func_name)
    io.save_fitsimage(resfile, cube, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the recon
    reconfile = inst.get_variable('WLC_RECON_FILE', func_name)
    io.save_fitsimage(reconfile, lrecon, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the table to the csv path
    ltbl_file = inst.get_variable('WLC_LTBL_FILE', func_name)
    io.save_table(ltbl_file, l_table, fmt='csv')
    # =========================================================================
    # Plots and Summary HTML
    # =========================================================================
    # plot the stability plot
    plots.plot_stability(inst.params, l_table)
    # -------------------------------------------------------------------------
    # plot the transit plot
    plots.plot_transit(inst.params, l_table)
    # -------------------------------------------------------------------------
    # write the yaml file to html
    io.summary_html(inst.params)


def spectral_extraction(inst: Instrument):
    """
    White light curve functionality

    :param inst: Instrument, the instrument object
    :return:
    """
    # set the function name
    func_name = f'{__NAME__}.spectral_extraction'
    # print the splash
    misc.sossart()
    # get parameters from instrumental parameters
    objname = inst.params['OBJECTNAME']
    # print the white light curve splash
    print(misc.art('Spectral timeseries ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # load the median image
    med_file = self.get_variable('MEIDAN_IMAGE_FILE', func_name)
    med = io.load_fits(med_file)
    # load the residuals
    res_file = self.get_variable('WLC_RES_FILE', func_name)
    residual = io.load_fits(res_file)
    # load the error file
    err_file = self.get_variable('WLC_ERR_FILE', func_name)
    err = io.load_fits(err_file)
    # load the residuals
    recon_file = self.get_variable('WLC_RECON_FILE', func_name)
    recon = io.load_fits(recon_file)
    # load the linear fit table
    ltable_file = self.get_variable('WLC_LTBL_FILE', func_name)
    ltable = io.load_table(ltable_file)
    # -------------------------------------------------------------------------
    # plot / save storage
    storage = dict()
    # -------------------------------------------------------------------------
    # loop around trace orders
    for trace_order in inst.params['TRACE_ORDERS']:
        # create the SED
        sed_tbl = inst.create_sed(med, residual, trace_order)
        # ---------------------------------------------------------------------
        # load the model (and deal with masking order zero if required)
        model = inst.load_model(recon, med, residual)
        # ---------------------------------------------------------------------
        # spectrum is the ratio of the residual to the trace model
        spec, spec_err = inst.ratio_residual_to_trace(model, err, res,
                                                      trace_order)
        # ---------------------------------------------------------------------
        # remove the out-of-transit trend
        if inst.params['REMOVE_TREND']:
            spec, ltable = inst.remove_trend(spec, ltable)
        # ---------------------------------------------------------------------
        # compute or set transit depth
        transit_depth = inst.get_transit_depth(ltable)
        # ---------------------------------------------------------------------
        # get the in-transit spectrum
        isout = inst.intransit_spectrum(spec, spec_err)
        spec_in, spec_err_in, spec_err_out = isout
        # ---------------------------------------------------------------------
        # bin the data by RESOLUTION_BIN
        wave_bin, flux_bin, flux_bin_err = inst.bin_spectrum(wavegrid,
                                                             spec_in,
                                                             spec_err_in)
        # ---------------------------------------------------------------------
        # save for plotting (outside the trace_order loop) / saving
        storage_it = dict()
        storage_it['wavegrid'] = wavegrid
        storage_it['sp_sed'] = sp_sed
        storage_it['throughput'] = throughput
        storage_it['spec_in'] = spec_in
        storage_it['spec_err_in'] = spec_err_in
        storage_it['transit_depth'] = transit_depth
        storage_it['wave_bin'] = wave_bin
        storage_it['flux_bin'] = flux_bin
        storage_it['flux_bin_err'] = flux_bin_err
        # append to plot storage
        storage[trace_order] = plot_storage_it
        # ---------------------------------------------------------------------
        inst.save_results(storage_it)
    # -------------------------------------------------------------------------
    # plot the SED
    plots.plot_full_sed(inst.params, storage)
    # -------------------------------------------------------------------------
    # convert sossisse to eureka products
    inst.to_eureka()

