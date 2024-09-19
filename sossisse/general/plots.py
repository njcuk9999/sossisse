#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from sossisse.core import base
from sossisse.core import math
from sossisse.core import math as mp
from sossisse.core import misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.plots'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions used by plots
# =============================================================================
def save_show_plot(params: Dict[str, Any], outname: str):
    """
    Save and show the plot
    :param params: dict, the parameters for the instrument
    :param outname: str, the output name for the plot

    :return:
    """
    # loop around figure types
    for figtype in params['FIGURE_TYPES']:
        # construct the basename with extension
        basename = f'{outname}.{figtype}'
        # contstruct the full path
        abspath = os.path.join(params['PLOT_PATH'], basename)
        # say that we are plotting graph
        msg = f'Plotting graph: {basename}'
        misc.printc(msg, msg_type='info')
        # save the figure
        plt.savefig(abspath)
    # if we want to show the plot do it now
    if params['SHOW_PLOTS']:
        # show the plot
        plt.show()
    # finally close the plot
    plt.close()


def cal_y_limits(value: np.ndarray, errvalue: np.ndarray) -> List[float]:
    """
    Calculate the y limits for a plot

    :param value: np.ndarray, the value array
    :param errvalue: np.ndarray, the error value array

    :return: list of floats, the y limits [min, max]
    """
    # calculate the ylimit based on the 0.5th and 99.5th percentiles
    y0 = np.nanpercentile(value - errvalue, 0.5)
    y1 = np.nanpercentile(value + errvalue, 99.5)
    diff_y = y1 - y0
    # we set the limits an extra 8th of the difference above and below
    ylim = [y0 - diff_y / 8, y1 + diff_y / 8]
    # return the y limits
    return ylim


# =============================================================================
# Define plot functions
# =============================================================================
def pca_plot(inst: Any, n_comp: int, pcas: np.ndarray,
             variance_ratio: np.ndarray):
    """
    Plot the PCA components

    :param inst: Instrument instance
    :param n_comp: int, the number of PCA components we have
    :param pcas: np.ndarray, the PCA components
    :param variance_ratio: np.ndarray, ratio of variance normaliszed to the
                           first component

    :return: None, plots graph
    """
    # set up figure
    fig, frames = plt.subplots(nrows=n_comp, ncols=1, sharex='all',
                               sharey='all', figsize=[8, 4 * n_comp])
    # deal with single component (frames is a single axis)
    if n_comp == 1:
        frames = [frames]
    # -------------------------------------------------------------------------
    # loop around components
    for icomp in range(n_comp):
        i_pca = pcas[icomp]
        # plot the component
        frames[icomp].imshow(i_pca, aspect='auto',
                             vmin=np.nanpercentile(i_pca, 0.5),
                             vmax=np.nanpercentile(i_pca, 99.5),
                             origin='lower')
        # set the title of the plot
        title = f'PCA {icomp + 1}, variance {variance_ratio[icomp]:.4f}'
        frames[icomp].set(title=title)
    # -------------------------------------------------------------------------
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'file_temporary_pcas')


def gradient_plot(inst: Any, dx: np.ndarray, dy: np.ndarray,
                  rotxy: np.ndarray):
    """
    Plot the gradients

    :param inst: Instrument instance
    :param dx: np.ndarray, the gradient in x
    :param dy: np.ndarray, the gradient in y
    :param rotxy: np.ndarray, the rotation between x and y

    :return: None, plots graph
    """
    # set function name
    func_name = f'{__NAME__}.gradient_plot()'
    # set up figure
    fig, frames = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all')
    # -------------------------------------------------------------------------
    # work out the rms of dx
    rms = np.nanpercentile(dx, [5, 95])
    rms = rms[1] - rms[0]
    # -------------------------------------------------------------------------
    # plot dx
    frames[0].imshow(dx, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
    rms = np.nanpercentile(dy, [5, 95])
    # work out the rms of dy
    rms = rms[1] - rms[0]
    # -------------------------------------------------------------------------
    # plot dy
    frames[1].imshow(dy, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
    # work out the rms of rotxy
    rms = np.nanpercentile(rotxy, [5, 95])
    rms = rms[1] - rms[0]
    # -------------------------------------------------------------------------
    # plot rotxy
    frames[2].imshow(rotxy, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'derivatives{0}'.format(tag))


def mask_order0_plot(inst: Any, diff: np.ndarray, sigmask: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.mask_order0_plot()'
    # set up figure
    fig, frames = plt.subplots(nrows=2, ncols=1)
    # -------------------------------------------------------------------------
    # plot the diff
    frames[0].imshow(diff, aspect='auto', origin='lower',
                     vmin=np.nanpercentile(diff, 2),
                     vmax=np.nanpercentile(diff, 80),
                     interpolation='none')
    # -------------------------------------------------------------------------
    # plot the sigmask
    frames[1].imshow(sigmask, aspect='auto', origin='lower',
                     interpolation='none')
    # set titles
    frames[0].set(title='median residual in-out')
    frames[1].set(title='mask')
    # -------------------------------------------------------------------------
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'masking_order0_{0}'.format(tag))


def trace_correction_sample(inst: Any, iframe: int,
                            cube: np.ndarray, recon: np.ndarray,
                            x_trace_pos: np.ndarray, y_trace_pos: np.ndarray,
                            x_order0: np.ndarray, y_order0: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.trace_correction'
    # setup the figure
    fig, frames = plt.subplots(nrows=2, ncols=1, figsize=[12, 12])
    # plot the cube
    frames[0].imshow(cube[iframe], aspect='auto', origin='lower',
                     vmin=np.nanpercentile(cube[iframe], 1),
                     vmax=np.nanpercentile(cube[iframe], 95))
    frames[0].set(title='Sample Image')
    # -------------------------------------------------------------------------
    # remove the recon temporarily for the plot
    tmp = cube[iframe] - recon
    # plot the cube minus the recon
    frames[1].imshow(tmp, aspect='auto', origin='lower',
                     vmin=np.nanpercentile(tmp, 5),
                     vmax=np.nanpercentile(tmp, 95))
    # -------------------------------------------------------------------------
    # plot the trace positions
    frames[0].plot(x_trace_pos, y_trace_pos, '.', color='orange', alpha=0.2)
    frames[1].plot(x_trace_pos, y_trace_pos, '.', color='orange', alpha=0.2,
                   label='trace mask')
    # -------------------------------------------------------------------------
    # plot the order0 positions (if given)
    if len(x_order0) > 2:
        frames[0].plot(x_order0, y_order0, 'r.', alpha=0.1)
        frames[1].plot(x_order0, y_order0, 'r.', alpha=0.1, label='order 0')
    # -------------------------------------------------------------------------
    # setup the legend and title
    frames[1].legend()
    frames[1].set(title='Residual')
    # -------------------------------------------------------------------------
    # remove the x and y axis labels
    frames[0].get_xaxis().set_visible(False)
    frames[0].get_yaxis().set_visible(False)
    frames[1].get_xaxis().set_visible(False)
    frames[1].get_yaxis().set_visible(False)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'sample_frame{0}_{1}'.format(iframe, tag))


def aperture_correction_plot(inst: Any, outputs: Dict[str, Any],
                             trace_corr: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.aperture_correction_plot()'
    # get values from outputs
    xpix = np.arange(len(outputs['amplitude_uncorrected']))
    amp_uncorr = outputs['amplitude_uncorrected']
    amp_corr = outputs['amplitude']
    yerr = outputs['amplitude_error']
    # scale the trace correction
    tmp_trace_corr = 1e3 * (trace_corr - 1)
    # get the limits of the trace correction
    p12 = np.nanpercentile(tmp_trace_corr, [1, 99])
    # -------------------------------------------------------------------------
    # setup the figure
    fig, frames = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=[10, 10])
    # -------------------------------------------------------------------------
    # plot the uncorrected amplitude
    frames[0].errorbar(xpix, amp_uncorr, yerr=yerr, fmt='r.',
                       alpha=0.3, label='uncorrected')
    # -------------------------------------------------------------------------
    # plot the corrected amplitude
    frames[0].errorbar(xpix + 0.5, amp_corr, yerr=yerr,
                       fmt='g.', alpha=0.3, label='corrected')
    # -------------------------------------------------------------------------
    # setup the title and legend
    frames[0].set(title='amplitude')
    frames[0].legend()
    # -------------------------------------------------------------------------
    # plot the trace correction
    frames[1].plot(tmp_trace_corr, 'r.', alpha=0.3)
    # convert the trace limits in to y limits on the graph
    ylim = [p12[0] - 0.3 * (p12[1] - p12[0]), p12[1] + 0.3 * (p12[1] - p12[0])]
    # set the title, labels and limits
    frames[1].set(title='Apperture correction', ylabel='corr [ppt]', ylim=ylim)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'apperture_correction{0}'.format(tag))


def plot_stability(inst: Any, table: Table):
    # set function name
    func_name = f'{__NAME__}.plot_stability()'
    # validate out-of-transit domain
    inst.get_valid_oot()
    has_oot = inst.get_variable('HAS_OOT', func_name)
    oot_domain = inst.get_variable('OOT_DOMAIN', func_name)
    # -------------------------------------------------------------------------
    # get the output names, units and factors
    output_names = inst.get_variable('OUTPUT_NAMES', func_name)
    output_units = inst.get_variable('OUTPUT_UNITS', func_name)
    output_factor = inst.get_variable('OUTPUT_FACTOR', func_name)
    # -------------------------------------------------------------------------
    # get the number of outputs
    noutputs = len(output_names)
    # get the number of points
    npoints = len(table['amplitude'])
    # force output factors to be floats
    output_factor = np.array(output_factor, dtype=float)
    # -------------------------------------------------------------------------
    # calculate the rms photon noise
    rms_phot = inst.get_rms_baseline(table['amplitude'],
                                     method='quadratic_sigma')
    # -------------------------------------------------------------------------
    # set up the plot
    fig, frames = plt.subplots(nrows=noutputs, ncols=1, sharex='all',
                               figsize=[8, 12])
    # -------------------------------------------------------------------------
    # set the alpha level
    alpha = np.min([np.sqrt(200 / npoints), 1])
    # -------------------------------------------------------------------------
    # get a index array
    index = np.arange(npoints)
    # -------------------------------------------------------------------------
    # get the domain text
    if inst.params['WLC_DOMAIN'] is not None:
        dargs = [inst.params['WLC_DOMAIN'][0], inst.params['WLC_DOMAIN'][1],
                 inst.params['SID']]
        domain = '({0:.2f} - {1:.2f}µm)\nunique ID {2}\n'.format(*dargs)
    else:
        domain = ''
    # -------------------------------------------------------------------------
    # set the x label
    xlabel = 'N$^{th}$ integration'
    # -------------------------------------------------------------------------
    # loop around parameters and plot them
    for it in range(noutputs):
        # get this iterations name/unit/factor
        name_it = output_names[it]
        unit_it = output_units[it]
        factor_it = output_factor[it]
        # ---------------------------------------------------------------------
        # get the value and error value from the table and scale it accordingly
        value = table[name_it] * factor_it
        errvalue = table[name_it + '_error'] * factor_it
        # ---------------------------------------------------------------------
        # deal with having out of transit points
        if has_oot:
            # plot the out of transit points
            frames[it].errorbar(index[oot_domain], value[oot_domain],
                                yerr=errvalue[oot_domain],
                                fmt='.', color='green', alpha=alpha,
                                label='out-of-transit')
            # plot the in transit points
            frames[it].errorbar(index[~oot_domain], value[~oot_domain],
                                yerr=errvalue[~oot_domain],
                                fmt='.', color='red', alpha=alpha,
                                label='in-transit')
            # only plot the legend for the first plot frame
            if it == 0:
                frames[it].legend()
        # otherwise we just plot everything
        else:
            # plot all points
            frames[it].errorbar(index, value, yerr=errvalue,
                                fmt='g.', alpha=0.4)
        # ---------------------------------------------------------------------
        # axis labels, title and grid
        # ---------------------------------------------------------------------
        # get the y limits
        ylim = cal_y_limits(value, errvalue)
        # get the rms for this output
        rms_it = mp.estimate_sigma(table[name_it] * factor_it)
        # ---------------------------------------------------------------------
        # get the title for the plot
        if it == 0:
            title = f'{inst.params["OBJECTNAME"]} -- {inst.params["SUFFIX"]}\n'
            title += domain
            title += f'rms: {rms_phot: .2f} ppm'
        else:
            title = f'rms: {rms_it:.4f} {unit_it}'
        # ---------------------------------------------------------------------
        # construct the y label
        ylabel = f'{name_it} [{unit_it}]'
        # ---------------------------------------------------------------------
        # push all the settings to the plot frame
        frames[it].set(xlabel=xlabel, ylabel=ylabel, ylim=ylim, title=title)
        frames[it].grid(color='grey', linestyle='--', alpha=alpha, linewidth=2)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'stability{0}'.format(tag))


def plot_transit(inst: Any, table: Table):
    # set function name
    func_name = f'{__NAME__}.plot_transit()'
    # validate out-of-transit domain
    inst.get_valid_oot()
    has_oot = inst.get_variable('HAS_OOT', func_name)
    oot_domain = inst.get_variable('OOT_DOMAIN', func_name)
    # get the polynomial degree for the transit baseline
    poly_order = inst.params['TRANSIT_BASELINE_POLYORD']
    # -------------------------------------------------------------------------
    # get the contact points
    cframes = inst.params['CONTACT_FRAMES']
    # get the number of points
    npoints = len(table['amplitude'])
    # -------------------------------------------------------------------------
    # get the amplitude and error values
    value = table['amplitude']
    errvalue = table['amplitude_error']
    # get the index of the pixels
    index = np.arange(npoints)
    # -------------------------------------------------------------------------
    # deal with no out-of-transit defined
    if not has_oot:
        oot_domain = np.ones_like(index, dtype=bool)
    # -------------------------------------------------------------------------
    # 5-sigma robust poly fit of the continuum
    ampfit, _ = mp.robust_polyfit(index[oot_domain], value[oot_domain],
                                  degree=poly_order, nsigcut=5)
    # remove this fit from the amplitude
    value = value / np.polyval(ampfit, index)
    # -------------------------------------------------------------------------
    # deal with no out-of-transit defined (can't workout mid-transit)
    if not has_oot:
        mid_transit = np.ones_like(len(index))
    else:
        # calculate the mid-transit frames
        norm_index = index - (cframes[0] + cframes[3]) / 2
        mid_transit = np.abs(norm_index) < 0.3 * (cframes[3] - cframes[0])
    # -------------------------------------------------------------------------
    # fit the mid transit frames
    fit_mid, _ = math.robust_polyfit(index[mid_transit], value[mid_transit],
                                     degree=2, nsigcut=5)
    # calculate the mid-transit point and depth
    mid_transit_point = -0.5 * fit_mid[1] / fit_mid[0]
    mid_transit_depth = np.polyval(fit_mid, mid_transit_point)
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frame = plt.subplots(nrows=1, ncols=1, figsize=[8, 4])
    # -------------------------------------------------------------------------
    # plot the transit + fit (if we have out-of-transit defined)
    if has_oot:
        frame.errorbar(index[oot_domain], value[oot_domain],
                       yerr=errvalue[oot_domain],
                       fmt='.', color='green', alpha=0.4, label='oot', zorder=3)
        frame.errorbar(index[~oot_domain], value[~oot_domain],
                       yerr=errvalue[~oot_domain],
                       fmt='.', color='red', alpha=0.4, label='it', zorder=2)
        # plot the transit fit
        frame.plot(index[mid_transit], np.polyval(fit_mid, index[mid_transit]),
                   'k--', zorder=10)
        # add a legend
        frame.legend()
    # otherwise just plot the transit
    else:
        frame.errorbar(index, value, yerr=errvalue,
                       fmt='.', color='green', alpha=0.4, label='oot', zorder=3)

    # ---------------------------------------------------------------------
    # axis labels, title and grid
    # ---------------------------------------------------------------------
    # get the y limits
    ylim = cal_y_limits(value, errvalue)
    # get the title for the plot
    title = f'{inst.params["OBJECTNAME"]} -- {inst.params["SUFFIX"]}\n'
    title += f'{mid_transit_depth * 1e6:.0f} ppm'

    # set the axis
    frame.set(xlabel='Nth frame', ylabel='Baseline-corrected flux', ylim=ylim,
              title=title)
    # set up the grid
    frame.grid(linestyle='--', color='grey', zorder=-99)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # get tag
    tag = inst.get_variable('TAG1', func_name)
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'transit_{0}'.format(tag))


def plot_sed(inst: Any, wavegrid: np.ndarray, sed: np.ndarray,
             trace_order: int):
    # set function name
    # func_name = f'{__NAME__}.plot_sed()'
    # get object name and suffix
    objname = inst.params['OBJECTNAME']
    suffix = inst.params['SUFFIX']
    # set up the plot
    fig, frame = plt.subplots(nrows=1, ncols=1)
    # plot the SED
    frame.plot(wavegrid, sed)
    # construct title
    title = f'{objname} -- {suffix} order={trace_order}'
    # set the axis labels
    frame.set(xlabel='Wavelength [nm]',
              ylabel='Flux\nthroughput-corrected',
              title=title)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'sed_{0}_ord{1}'.format(objname, trace_order))


def plot_full_sed(inst: Any, plot_storage: Dict[str, Dict[str, Any]]):
    # set up the plot
    fig, frame = plt.subplots(nrows=1, ncols=1)
    # loop around tarce orders
    for trace_order in plot_storage.keys():
        # deal with trace order
        if trace_order == 1:
            fmt1 = 'g.'
            fmt2 = 'ro--'
        elif trace_order == 2:
            fmt1 = 'c.'
            fmt2 = 'mo--'
        else:
            continue
        # get this trace orders parameters
        wavegrid = plot_storage[trace_order]['wavegrid']
        sed_spec = plot_storage[trace_order]['sp_sed']
        throughtput = plot_storage[trace_order]['throughput']
        spec_in = plot_storage[trace_order]['spec_in']
        spec_err_in = plot_storage[trace_order]['spec_err_in']
        transit_depth = plot_storage[trace_order]['transit_depth']
        wave_bin = plot_storage[trace_order]['wave_bin']
        flux_bin = plot_storage[trace_order]['flux_bin']
        flux_bin_err = plot_storage[trace_order]['flux_bin_err']
        # plot the SED
        frame.plot(wavegrid, sed_spec / throughtput,
                   label='Flux, throughput-corrected, '
                         'order {0}'.format(trace_order))
        # plot the in-transit spectrum
        frame.errorbar(wavegrid, (spec_in + transit_depth) * 1e6,
                    yerr=spec_err_in * 1e6,
                    fmt=fmt1, alpha=0.25,
                    label='in-transit, order {}'.format(trace_order))
        # plot the binned in-transit spectrum
        binlabelargs = [inst.params['RESOLUTION_BIN'], trace_order]
        binlabel = 'Resolution {}, order {}'.format(*binlabelargs)
        frame.errorbar(wave_bin, (flux_bin + transit_depth) * 1e6,
                    yerr=flux_bin_err * 1e6, fmt=fmt2,
                    label=binlabel)
    # -------------------------------------------------------------------------
    # construct title
    objname = inst.params['OBJECTNAME']
    title = f'{objname} -- {inst.params["SUFFIX"]}'
    # set the axis labels
    frame.set(xlabel=r'Wavelength [$\mu$m]', ylabel='ppm',
              title=title)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'sed_{0}'.format(objname))
