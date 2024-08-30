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

from sossisse.core import base
from sossisse.core import math
from sossisse.core import misc
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
    # set up figure
    fig, frames = plt.subplots(nrows=n_comp, ncols=1, sharex='all',
                               sharey='all', figsize=[8, 4 * n_comp])
    # deal with single component (frames is a single axis)
    if n_comp == 1:
        frames = [frames]
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

    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(params, 'file_temporary_pcas')




def plot_sossice(tbl, params):
    params = science.get_valid_oot(params)
    params['output_factor'] = np.array(params['output_factor'], dtype=float)

    nrows = len(params['output_names'])

    # just for fun, a graph with stats on the trace rotation+scale
    rms_phot = science.get_rms_baseline(tbl['amplitude'], method='quadratic_sigma')
    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex='all', figsize=[8, 12])

    alpha = np.min([np.sqrt(200 / len(tbl)), 1])

    for i in range(nrows):
        val = tbl[params['output_names'][i]] * params['output_factor'][i]
        errval = tbl[params['output_names'][i] + '_error'] * params['output_factor'][i]

        y0 = np.nanpercentile(val - errval, 0.5)
        y1 = np.nanpercentile(val + errval, 99.5)
        dy = y1 - y0
        ylim = [y0 - dy / 8, y1 + dy / 8]
        ax[i].set(ylim=ylim)
        if 'oot_domain' in params.keys():
            oot = params['oot_domain']
            ax[i].errorbar(np.arange(len(tbl))[oot], val[oot], yerr=errval[oot],
                           fmt='.', color='green', alpha=alpha, label='oot')
            ax[i].errorbar(np.arange(len(tbl))[~oot], val[~oot], yerr=errval[~oot],
                           fmt='.', color='red', alpha=alpha, label='it')
            if i == 0:
                ax[i].legend()
        else:
            ax[i].errorbar(np.arange(len(tbl)), val, yerr=errval,
                           fmt='g.', alpha=0.4)

    xlabel = 'N$^{th}$ integration'

    if 'wlc_domain' in params.keys():
        domain = '({0:.2f} - {1:.2f}µm)\nunique ID {2}'.format(params['wlc_domain'][0],
                                                               params['wlc_domain'][1], params['checksum'])
    else:
        domain = ''

    for i in range(nrows):

        if i == 0:
            title = '{0} -- {1}\nrms : {3:.2f} ppm'.format(params['object'], params['suffix'], domain, rms_phot * 1e6)
        else:
            title = 'rms : {:.4f}'.format(math.estimate_sigma(tbl[params['output_names'][i]]) * params['output_factor'][i])

        ylabel = '{} [{}]'.format(params['output_names'][i], params['output_units'][i])
        ax[i].set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax[i].grid(color='grey', linestyle='--', alpha=alpha, linewidth=2)

    plt.tight_layout()
    for figtype in params['figure_types']:
        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        plt.savefig('{}/stability_soss{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype))
    if params['show_plots']:
        plt.show()
    plt.close()


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
