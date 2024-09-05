#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from sossisse.core import base
from sossisse.core import math
from sossisse.core import misc
from sossisse.core import math as mp
from sossisse.general import science


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
    :return:
    """
    # loop around figure types
    for figtype in params['FIGURE_TYPES']:
        # construct the basename with extension
        basename = f'{outname}.{figtype}'
        # contstruct the full path
        abspath = os.path.join(params['PLOT_PATH'], basename)
        # say that we are plotting graph
        msg = f'Plotting graph: {abspath}'
        misc.printc(msg, msg_type='info')
        # save the figure
        plt.savefig(abspath)
    # if we want to show the plot do it now
    if params['SHOW_PLOTS']:
        # show the plot
        plt.show()
    # finally close the plot
    plt.close()



# =============================================================================
# Define plot functions
# =============================================================================
def pca_plot(params: Dict[str, Any], n_comp: int, pcas: np.ndarray,
             variance_ratio: np.ndarray):
    """
    Plot the PCA components

    :param params: Dict[str, Any], the parameters for the instrument
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
    save_show_plot(params, 'file_temporary_pcas')


def gradient_plot(params: Dict[str, Any], dx: np.ndarray, dy: np.ndarray,
                  rotxy: np.ndarray):
    """
    Plot the gradients

    :param params: Dict[str, Any], the parameters for the instrument
    :param dx: np.ndarray, the gradient in x
    :param dy: np.ndarray, the gradient in y
    :param rotxy: np.ndarray, the rotation between x and y

    :return: None, plots graph
    """
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
    # standard save/show plot for SOSSISSE
    save_show_plot(params, 'derivatives{0}'.format(params['tag']))


def mask_order0_plot(params: Dict[str, Any], diff: np.ndarray,
                     sigmask: np.ndarray):
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
    # standard save/show plot for SOSSISSE
    save_show_plot(params, 'masking_order0_{0}'.format(params['tag']))


def trace_correction_sample(params: Dict[str, Any], iframe: int, 
                            cube: np.ndarray, recon: np.ndarray,
                            x_trace_pos: np.ndarray, y_trace_pos: np.ndarray,
                            x_order0: np.ndarray, y_order0: np.ndarray):
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
    # standard save/show plot for SOSSISSE
    save_show_plot(params, 'sample_frame{0}_{1}'.format(iframe, params['tag']))
    

def aperture_correction_plot(params: Dict[str, Any],
                             outputs: Dict[str, Any], trace_corr: np.ndarray):
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
    # standard save/show plot for SOSSISSE
    save_show_plot(params, 'apperture_correction{0}'.format(params['tag']))


# TODO: re-write
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
    if  inst.params['WLC_DOMAIN'] is not None:
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
        errval = table[name_it + '_error'] * factor_it
        # ---------------------------------------------------------------------
        # deal with having out of transit points
        if has_oot:
            # plot the out of transit points
            frames[it].errorbar(index[oot_domain], value[oot_domain],
                                yerr=errval[oot_domain],
                                fmt='.', color='green', alpha=alpha,
                                label='out-of-transit')
            # plot the in transit points
            frames[it].errorbar(index[~oot_domain], value[~oot_domain],
                                yerr=errval[~oot_domain],
                                fmt='.', color='red', alpha=alpha,
                                label='in-transit')
            # only plot the legend for the first plot frame
            if it == 0:
                frames[it].legend()
        # otherwise we just plot everything
        else:
            # plot all points
            frames[it].errorbar(index, value, yerr=errval,
                                fmt='g.', alpha=0.4)
        # ---------------------------------------------------------------------
        # axis labels, title and grid
        # ---------------------------------------------------------------------
        # calculate the ylimit based on the 0.5th and 99.5th percentiles
        y0 = np.nanpercentile(value - errval, 0.5)
        y1 = np.nanpercentile(value + errval, 99.5)
        diff_y = y1 - y0
        # we set the limits an extra 8th of the difference above and below
        ylim = [y0 - diff_y / 8, y1 + diff_y / 8]
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
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'stability{0}'.format(inst.params['tag']))



# TODO: re-write
def plot_transit(tbl, params):
    params = science.get_valid_oot(params)
    val = tbl['amplitude']
    errval = tbl['amplitude_error']

    oot = params['oot_domain']
    index = np.arange(len(tbl))

    # 5-sigma
    fit = math.robust_polyfit(index[oot], val[oot], params['transit_baseline_polyord'], 5)[0]
    val = val / np.polyval(fit, index)

    y0 = np.nanpercentile(val - errval, 0.5)
    y1 = np.nanpercentile(val + errval, 99.5)
    dy = y1 - y0
    ylim = [y0 - dy / 2, y1 + dy / 2]

    mid_transit = np.abs(index - (params['it'][0] + params['it'][3]) / 2) < 0.3 * (params['it'][3] - params['it'][0])

    fit_mid = math.robust_polyfit(index[mid_transit], val[mid_transit], 2, 5)[0]

    mid_transit_point = -.5 * fit_mid[1] / fit_mid[0]
    mid_transit_depth = np.polyval(fit_mid, mid_transit_point)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 4])
    ax.set(ylim=ylim)
    ax.errorbar(index[oot], val[oot], yerr=errval[oot],
                fmt='.', color='green', alpha=0.4, label='oot', zorder=3)
    ax.errorbar(index[~oot], val[~oot], yerr=errval[~oot],
                fmt='.', color='red', alpha=0.4, label='it', zorder=2)

    ax.plot(index[mid_transit], np.polyval(fit_mid, index[mid_transit]), 'k--', zorder=10)
    ax.set(title='{0} -- {1}\nMid-transit depth : {2:.0f} ppm'.format(params['object'],
                                                                      params['suffix'], (1 - mid_transit_depth) * 1e6))
    ax.set(xlabel='Nth frame')
    ax.set(ylabel='Baseline-corrected flux')
    ax.grid(linestyle='--', color='grey', zorder=-99)
    ax.legend()

    plt.tight_layout()
    for figtype in params['figure_types']:
        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        plt.savefig('{}/transit_{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype))
    if params['show_plots']:
        plt.show()
    plt.close()
