#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.general.py

General functionality in here

Created on 2022-09-20

@author: cook
"""
import os

import numpy as np

from sossisse.core import base
from sossisse.core import io
from sossisse.core import misc
from sossisse.general import plots
from sossisse.instruments import Instrument

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
def white_light_curve(inst: Instrument) -> Instrument:
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
    objname = inst.params['INPUTS']['OBJECTNAME']
    # print the white light curve splash
    print(misc.art('White light curve ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # get the stabiblity table file name
    wlc_ltbl_file = inst.get_variable('WLC_LTBL_FILE', func_name)
    # return if we have the soss_stablity file
    if os.path.exists(wlc_ltbl_file):
        msg = 'File {0} exists we skip white light curve step'
        misc.printc(msg.format(wlc_ltbl_file), 'info')
        return inst
    # -------------------------------------------------------------------------
    # fancy centering - recalculate trace file
    inst.fancy_centering()
    # -------------------------------------------------------------------------
    # load the image, error and data quality
    cube, err, dq = inst.load_data_with_dq()
    # -------------------------------------------------------------------------
    # remove the background
    cube, err = inst.remove_background(cube, err, dq)
    # -------------------------------------------------------------------------
    # for each slice of the cube, isolated bad pixels are interpolated with the
    # value of their 4 neighbours.
    cube = inst.patch_isolated_bads(cube)
    # -------------------------------------------------------------------------
    # remove cosmic rays with a sigma cut
    cube = inst.remove_cosmic_rays(cube)
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
    # -----------------------------------------------------inst--------------------
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
    ltable, lrecon, valid_cube = amp_out
    # -------------------------------------------------------------------------
    # normalize the trace but a normalization factor
    ltable = inst.normalize_sum_trace(ltable)
    # -------------------------------------------------------------------------
    # per pixel baseline
    if inst.params['WLC']['GENERAL']['PER_PIXEL_BASELINE_CORRECTION']:
        misc.printc('Performing per-pixel baseline subtraction', 'info')
        cube = inst.per_pixel_baseline(cube, valid_cube)
    # -------------------------------------------------------------------------
    # print the rms baseline for all methods
    for method in inst.get_rms_baseline():
        # calculate the rms for this method
        rms_method = inst.get_rms_baseline(ltable['amplitude'], method=method)
        # print this
        msg = '{0}, rms = {1:.1f}ppm'.format(method, rms_method * 1e6)
        misc.printc(msg, 'number')
    # -------------------------------------------------------------------------
    # get the effective wavelength
    photo_weighted_mean, energy_weighted_mean = inst.get_effective_wavelength()
    # =========================================================================
    # write files
    # =========================================================================
    inst.save_wlc_results(cube, err, lrecon, ltable)
    # =========================================================================
    # Plots and Summary HTML
    # =========================================================================
    # plot the stability plot
    plots.plot_stability(inst, ltable)
    # -------------------------------------------------------------------------
    # plot the transit plot
    plots.plot_transit(inst, ltable)
    # -------------------------------------------------------------------------
    # write the yaml file to html
    io.summary_html(inst.params)
    # -------------------------------------------------------------------------
    # return the instrument object
    return inst


def spectral_extraction(inst: Instrument) -> Instrument:
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
    objname = inst.params['INPUTS']['OBJECTNAME']
    # print the white light curve splash
    print(misc.art('Spectral timeseries ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # plot / save storage
    storage = dict()
    # -------------------------------------------------------------------------
    # loop around trace orders
    for trace_order in inst.params['GENERAL']['TRACE_ORDERS']:
        # print progress
        misc.printc('Processing trace order {0}'.format(trace_order), 'alert')
        # load the median image
        med_file = inst.get_variable('MEDIAN_IMAGE_FILE', func_name)
        med = io.load_fits(med_file)
        # get clean median trace for spectrum
        dx, dy, rotxy, ddy, med_clean = inst.get_gradients(med)
        # load the residuals
        res_file = inst.get_variable('WLC_RES_FILE', func_name)
        residual = io.load_fits(res_file)
        # load the error file
        err_file = inst.get_variable('WLC_ERR_FILE', func_name)
        err = io.load_fits(err_file)
        # load the residuals
        recon_file = inst.get_variable('WLC_RECON_FILE', func_name)
        recon = io.load_fits(recon_file)
        # load the linear fit table
        ltable_file = inst.get_variable('WLC_LTBL_FILE', func_name)
        ltable = io.load_table(ltable_file)
        # for future reference in the code, we keep track of data size
        inst.set_variable('DATA_X_SIZE', med.shape[1], func_name)
        inst.set_variable('DATA_Y_SIZE', med.shape[0], func_name)
        # get the trace position
        posmax, throughput = inst.get_trace_pos(order_num=trace_order)
        # get wave grid
        wavegrid = inst.get_wavegrid(order_num=trace_order)
        # create the SED
        sp_sed = inst.create_sed(med, residual, wavegrid, posmax, throughput,
                                  med_clean, trace_order)
        # ---------------------------------------------------------------------
        # load the model (and deal with masking order zero if required)
        model = inst.load_model(recon, med)
        # ---------------------------------------------------------------------
        # spectrum is the ratio of the residual to the trace model
        spec, spec_err = inst.ratio_residual_to_trace(model, err, residual,
                                                      posmax)
        # ---------------------------------------------------------------------
        # remove the out-of-transit trend on the spectrum
        if inst.params['SPEC_EXT']['REMOVE_TREND']:
            spec = inst.remove_trend_spec(spec)
        # -----------------------------------------------------------------
        # reshape the amplitudes into an image
        amp_image = np.repeat(np.array(ltable['amplitude']), spec.shape[1])
        amp_image = amp_image.reshape(spec.shape)
        # add this gray component onto the spectrum
        spec2 = spec + amp_image
        # ---------------------------------------------------------------------
        # remove the out-of-transit trend on the photometric time series
        if inst.params['SPEC_EXT']['REMOVE_TREND']:
            ltable = inst.remove_trend_phot(spec, ltable)
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
        # reshape the wave grid into an image
        wavegrid_2d = np.tile(wavegrid, (spec.shape[0], 1))
        # save for plotting (outside the trace_order loop) / saving
        storage_it = dict()
        storage_it['wavegrid'] = wavegrid
        storage_it['sp_sed'] = sp_sed
        storage_it['throughput'] = throughput
        storage_it['spec'] = spec
        storage_it['spec_err'] = spec_err
        storage_it['ltable'] = ltable
        storage_it['spec2'] = spec2
        storage_it['wavegrid_2d'] = wavegrid_2d
        storage_it['spec_in'] = spec_in
        storage_it['spec_err_in'] = spec_err_in
        storage_it['transit_depth'] = transit_depth
        storage_it['wave_bin'] = wave_bin
        storage_it['flux_bin'] = flux_bin
        storage_it['flux_bin_err'] = flux_bin_err
        # append to plot storage
        storage[trace_order] = storage_it
        # ---------------------------------------------------------------------
        inst.save_spe_results(storage_it, trace_order)
    # -------------------------------------------------------------------------
    # plot the SED
    plots.plot_full_sed(inst, storage)
    # -------------------------------------------------------------------------
    # convert sossisse to eureka products
    inst.to_eureka(storage)
    # -------------------------------------------------------------------------
    # write the yaml file to html
    io.summary_html(inst.params)
    # -------------------------------------------------------------------------
    # return the instrument object
    return inst


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
