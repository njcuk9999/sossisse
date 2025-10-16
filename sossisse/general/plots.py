#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:23

@author: cook
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from astropy.table import Table
from astropy.visualization import ImageNormalize
from astropy.visualization import interval as interval_mod
from astropy.visualization import stretch as stretch_mod

from aperocore import math as mp
from aperocore.base import base as aperobase
from sossisse.core import base
from sossisse.core import misc

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.plots'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define general function used by plots
# =============================================================================
def plot_fmt(val, spec=".2e"):
    return format(val, spec) if val is not None else 'None'


def plot_normalization(data: np.ndarray, 
                       interval: str = 'base', stretch: str = 'base',
                       vlims: List[float] = [0, 100],
                       vtype: str = 'percentile'
                       ) -> Tuple[ImageNormalize, str]:
    """
    Create a ds9 normalization object based on the interval and stretch type.
    

    :param data: np.ndarray, the data to normalize
    :param interval: str, the type of interval to use, currently supported are:
        - 'base': BaseInterval
        - 'minmax': MinMaxInterval
        - 'zscale': ZScaleInterval
    :param stretch: str, the type of stretch to use, currently supported are:
        - 'base': BaseStretch
        - 'linear': LinearStretch
        - 'sqrt': SqrtStretch
        - 'log': LogStretch
    :param vmin: float, the minimum value for the normalization
    :param vmax: float, the maximum value for the normalization
    :param vtype: str, the type of value to use for normalization, 
        currently supported are:
        - 'percentile': use percentiles for normalization
        - 'absolute': use min/max values for normalization

    :return: tuple, 1. ImageNormalize object, 2. str, the normalization text
    """
    # -------------------------------------------------------------------------
    if interval == 'base':
        interval_inst = interval_mod.BaseInterval()
    elif interval == 'minmax':
        interval_inst = interval_mod.MinMaxInterval()
    elif interval == 'zscale':
        interval_inst = interval_mod.ZScaleInterval()
    # else not supported
    else:
        raise ValueError(f'Unsupported interval type: {interval}')
    # -------------------------------------------------------------------------
    if stretch == 'base':
        stretch_inst = stretch_mod.BaseStretch()
    elif stretch == 'linear':
        stretch_inst = stretch_mod.LinearStretch()
    elif stretch == 'sqrt':
        stretch_inst = stretch_mod.SqrtStretch()
    elif stretch == 'log':
        stretch_inst = stretch_mod.LogStretch()
    # else not supported
    else:
        raise ValueError(f'Unsupported stretch type: {stretch}')
    # -------------------------------------------------------------------------
    # deal with vmin, vmax and vtype
    if vtype == 'percentile':
        # deal with vmin
        if vlims[0] <= 0:
            vmin = None
        else:
            vmin = vlims[0]
        # deal with vmax
        if vlims[1] >= 100:
            vmax = None
        else:
            vmax = vlims[1]
        # ---------------------------------------------------------------------
        # deal with calculating a percentile interval
        if vmin is not None and vmax is not None:
            vmin, vmax = np.nanpercentile(data, vlims)
        elif vmin is not None:
            vmin = np.nanpercentile(data, vmin)
        elif vmax is not None:
            vmax = np.nanpercentile(data, vmax)
    # otherwise we assume absolute vmin/vmax
    else:
        vmin, vmax = vlims
    # get the image normalization
    norm =  ImageNormalize(stretch=stretch_inst,   # type: ignore
                           interval=interval_inst,
                           vmin=vmin, vmax=vmax)

    # get the normalization text 
    ntext = (f'NORM[vmin={plot_fmt(vmin, ".2e")}, ' 
             f'vmax={plot_fmt(vmax, ".2e")}, '
             f'interval={interval}, stretch={stretch}, '
             f'min={np.nanmin(data):.2e}, max={np.nanmax(data):.2e}]')
    # return the normalization and text
    return norm, ntext


def add_footer_text(fig, text, fontsize=10, pad=0.02):
    """
    Adds a line of text at the very bottom of a matplotlib figure.

    Parameters:
        fig      : The matplotlib figure object.
        text     : The string to display as footer.
        fontsize : Size of the text.
        pad      : Padding from the bottom (in figure coordinates, default ~2%).
    """
    fig.text(0.5, pad, text, ha='center', va='bottom', fontsize=fontsize)


# =============================================================================
# Define functions used by plots
# =============================================================================
def plot_file(params: Dict[str, Any], outname: str, title,
              description: str = ''):
    # get plot file
    plotfile = os.path.join(params['PATHS.PLOT_PATH'], 'plots.yaml')
    # read yaml file
    if os.path.exists(plotfile):
        blocks = aperobase.load_yaml(plotfile)
    else:
        blocks = dict()
    # ------------------------------------------------------------------------
    # empty this block
    blocks[outname] = dict()
    # prepare yaml block
    blocks[outname]['title'] = title
    blocks[outname]['description'] = description
    blocks[outname]['time'] = float(time.time())
    # ------------------------------------------------------------------------
    # save the yaml file
    aperobase.write_yaml(blocks, plotfile)


def save_show_plot(params: Dict[str, Any], outname: str, title: str = '', 
                   description: str = ''):
    """
    Save and show the plot
    :param params: dict, the parameters for the instrument
    :param outname: str, the output name for the plot

    :return:
    """
    # save to yaml file (for html writing)
    plot_file(params, outname, title, description)
    # loop around figure types
    for figtype in params['PLOTS.FIGURE_TYPES']:
        # construct the basename with extension
        basename = f'{outname}.{figtype}'
        # contstruct the full path
        abspath = os.path.join(params['PATHS.PLOT_PATH'], basename)
        # say that we are plotting graph
        msg = f'Plotting graph: {basename}'
        misc.printc(msg, msg_type='info')
        # save the figure
        plt.savefig(abspath)
    # if we want to show the plot do it now
    if params['PLOTS.SHOW']:
        # deal with description
        if len(description) > 0:
            misc.printc(f'PLOT: {title} \n\n{description}', 'plot')
        # show the plot
        plt.show(block=True)
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
    # set title and description
    title = f'PCA components (n_comp={n_comp})'
    description = ('PCA components used to model the 1/f noise.')
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
    save_show_plot(inst.params, 'file_temporary_pcas', title, description)


def gradient_plot(inst: Any, data: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                  rotxy: np.ndarray, ddy: np.ndarray):
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
    # set title and description
    title = 'Gradients of median trace image'
    description = ('Gradients of the median trace image used to model '
                   'the white light curve flux variations.')
    # get the image normalization
    vlims = inst.params['WLC.PLOT.GRADIENT_VLIM']
    vtype = inst.params['WLC.PLOT.GRADIENT_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.GRADIENT_INTERVAL']
    stretch = inst.params['WLC.PLOT.GRADIENT_STRETCH']
    # set up figure
    fig, frames = plt.subplots(nrows=5, ncols=1, sharex='all', sharey='all',
                               figsize=[12, 12])
    # -------------------------------------------------------------------------
    ntexts = dict()
    # plot data
    norm, ntext = plot_normalization(data, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts['data'] = ntext
    frames[0].imshow(data, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[0].set(title='data (median trace image)')
    # -------------------------------------------------------------------------
    # plot dx
    norm, ntext = plot_normalization(dx, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts['dx'] = ntext
    frames[1].imshow(dx, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[1].set(title='derivative of median trace w.r.t. x  (dM/dx)')
    # -------------------------------------------------------------------------
    # plot dy
    norm, ntext = plot_normalization(dy, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts['dy'] = ntext
    frames[2].imshow(dy, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[2].set(title='derivative of median trace w.r.t. y  (dM/dy)')
    # -------------------------------------------------------------------------
    # plot rotxy
    norm, ntext = plot_normalization(rotxy, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts['rotxy'] = ntext
    frames[3].imshow(rotxy, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[3].set(title=r'derivative of median trace w.r.t. rotation '
                        r'(dM/d$\theta$)')
    # -------------------------------------------------------------------------
    # plot ddy
    norm, ntext = plot_normalization(ddy, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts['ddy'] = ntext
    frames[4].imshow(ddy, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[4].set(title=r'second derivative of median trace w.r.t. y '
                        r'($\partial^{2}M/\partial y^{2}$)')
    # -------------------------------------------------------------------------
    # deal with ntext
    ntext = ''
    for key in ntexts:
        ntext += f'\n{key}: {ntexts[key]} '
    # add footer text with normalization info
    add_footer_text(fig, ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'derivatives', title, description)


def plot_subtract_1f_scorr(inst: Any, scorr: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.plot_subtract_1f_scorr()'
    # set title and description
    title = '1/f noise correction values'
    description = ('1/f noise correction values used to correct the data. '
                   'Top panel is the correction values for the first '
                   'integration, middle panel is the correction values for '
                   'the last integration, bottom panel is the median '
                   'correction values across all integrations.')
    # get the degree for the 1/f polynomial fit
    degree_1f_corr = inst.params['WLC.GENERAL.DEGREE_1F_CORR']

    if degree_1f_corr == 0:
        mode = 'correction (for 1/f) = nanmedian(residuals) degree_1f_corr = 0'
    else:
        mode = 'correction (for 1/f)  = fit(residuals) degree_1f_corr = {0}'
        mode = mode.format(degree_1f_corr)

    # calculate the median scorr
    med_scorr = np.nanmedian(scorr, axis=0)

    # get the image normalization
    vlims = inst.params['WLC.PLOT.GRADIENT_VLIM']
    vtype = inst.params['WLC.PLOT.GRADIENT_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.GRADIENT_INTERVAL']
    stretch = inst.params['WLC.PLOT.GRADIENT_STRETCH']
    # get normaliz
    ntexts = dict()
    norm1, ntext1 = plot_normalization(scorr[0], interval=interval,
                                       stretch=stretch, vlims=vlims,
                                       vtype=vtype)
    ntexts['First'] = ntext1
    norm2, ntext2 = plot_normalization(scorr[-1], interval=interval,
                                       stretch=stretch, vlims=vlims,
                                       vtype=vtype)
    ntexts['Last'] = ntext2
    norm3, ntext3 = plot_normalization(med_scorr, interval=interval,
                                       stretch=stretch, vlims=vlims,
                                       vtype=vtype)
    ntexts['Median'] = ntext3
    # setup the figure
    fig, frames = plt.subplots(nrows=3, ncols=1, figsize=[8, 6])
    # plot the first integration
    im0 = frames[0].imshow(scorr[0], origin='lower', cmap='inferno',
                     aspect='auto', norm=norm1)
    frames[0].set(title='1/f correction values for first integration')
    # -------------------------------------------------------------------------
    # colorbar
    plt.colorbar(im0, ax=frames[0], orientation='vertical',
                 pad=0.01, fraction=0.05, label='1/f correction value')
    # plot the last integration
    im1 = frames[1].imshow(scorr[-1], origin='lower', cmap='inferno',
                           aspect='auto', norm=norm2)
    frames[1].set(title='1/f correction values for last integration')
    # colorbar
    plt.colorbar(im1, ax=frames[1], orientation='vertical',
                 pad=0.01, fraction=0.05, label='1/f correction value')
    # -------------------------------------------------------------------------
    # plot the median of all integrations
    im2 = frames[2].imshow(med_scorr, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm3)
    frames[2].set(title='Median 1/f correction values across all integrations')
    # colorbar
    plt.colorbar(im2, ax=frames[2], orientation='vertical',
                 pad=0.01, fraction=0.05, label='1/f correction value')
    # -------------------------------------------------------------------------
    # set the overall title
    plt.suptitle(mode)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # deal with ntext
    ntext = ''
    for key in ntexts:
        ntext += f'\n{key}: {ntexts[key]} '
    # add footer text with normalization info
    add_footer_text(fig, ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'subtract_1f_scorr', title, description)


def plot_subtract_1f_comp(inst: Any, cube0: np.ndarray, cube1: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.plot_subtract_1f_comp()'
    # set title and description
    title = '1/f noise correction example'
    description = ('Example of the 1/f noise correction on integration 0. '
                   'Top panel is before correction, bottom panel is after '
                   'correction.')
    # setup the figure
    fig, frames = plt.subplots(nrows=2, ncols=1, figsize=[12, 12])
    # get the image normalization parameters
    vlims = inst.params['WLC.PLOT.SUB1F_COMP_VLIM']
    vtype = inst.params['WLC.PLOT.SUB1F_COMP_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.SUB1F_COMP_INTERVAL']
    stretch = inst.params['WLC.PLOT.SUB1F_COMP_STRETCH']
    # choose the frame to plot
    iframe = 0
    # use same normalization for both plots
    norm, ntext = plot_normalization(cube0, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    # plot the cube0
    frames[0].imshow(cube0[iframe], origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[0].set(title='Integration 0 before 1/f subtraction')
    # -------------------------------------------------------------------------
    # plot the cube1
    frames[1].imshow(cube1[iframe], origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[1].set(title='Integration 0 after 1/f subtraction')
    # -------------------------------------------------------------------------
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'subtract_1f_comp', title, description)


def mask_order0_plot(inst: Any, diff0: np.ndarray, diff1: np.ndarray, 
                     diff2: np.ndarray, diff3: np.ndarray, diff4: np.ndarray, 
                     all_labels: np.ndarray, sigmask: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.mask_order0_plot()'
    # set title and description
    title = 'Order 0 masking'
    description = ('Order 0 masking steps. From top to bottom: '
                   'Original median trace image, straightened image, '
                   'low pass filtered image, unstraightened image, '
                   'Original - unstraightened, all clusters found, '
                   'Order 0 mask.')
    # set up figure
    fig, frames = plt.subplots(nrows=7, ncols=1, figsize=(12, 20))
    # loop around diffs
    diffs = [diff0, diff1, diff2, diff3, diff4]
    titles = ['Original', 'Straight', 'low pass', 'unstraightened',
              'Original - unstraightened']
    # -------------------------------------------------------------------------
    # plot the diff
    for it, _diff in enumerate(diffs):
        frames[it].imshow(_diff, aspect='auto', origin='lower',
                          vmin=np.nanpercentile(_diff, 2),
                          vmax=np.nanpercentile(_diff, 80),
                          interpolation='none')
        frames[it].set(title=titles[it])
    # -------------------------------------------------------------------------
    # plot all labels
    frames[5].imshow(all_labels > 0, aspect='auto', origin='lower',
                     interpolation='none')
        # set titles
    frames[5].set(title='All clusters found')
    # -------------------------------------------------------------------------
    # plot the sigmask
    frames[6].imshow(sigmask, aspect='auto', origin='lower',
                     interpolation='none')
    # set titles
    frames[6].set(title='Order 0 mask')
    # -------------------------------------------------------------------------
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'masking_order0', title, description)


def plot_trace_mask(inst: Any, trace_map: np.ndarray,
                    images: Optional[List[np.ndarray]] = None,
                    labels: Optional[List[str]] = None):
    # set function name
    func_name = f'{__NAME__}.plot_trace_mask()'
    # set title and description
    title = 'Trace mask'
    description = ('Trace mask used to extract the white light curve. '
                   'Hatched region is the trace mask.')
    # deal with no image or labels and set up figure
    if images is None or labels is None:
        _images, _labels = [None], [None]
        fig, frame = plt.subplots(nrows=1, ncols=1, figsize=(12, 20))
        frames = [frame]
    else:
        _images, _labels = images, labels
        fig, frames = plt.subplots(nrows=len(images), ncols=1, figsize=(12, 20))
    # -------------------------------------------------------------------------
    ntexts = dict()
    # loop around images
    for it, _image in enumerate(_images):
        # plot the background image
        if _image is not None:
            # get the image normalization
            norm, ntext = plot_normalization(_image, interval='minmax',
                                             stretch='log', vlims=[5, 95],
                                             vtype='percentile')
            # add ntext to ntexts
            ntexts[_labels[it]] = ntext
            # plot the image
            frames[it].imshow(_image, aspect='auto', origin='lower',
                              norm=norm, interpolation='none')
            frames[it].set(title=_labels[it])

        # plot the trace map on top
        frames[it].contourf(trace_map, levels=[0.5, 1.5], colors='none',
                            hatches=['////'], alpha=0)
        frames[it].contour(trace_map, levels=[0.5], colors='orange',
                           linewidths=2)
    # -------------------------------------------------------------------------
    # deal with ntext
    ntext = ''
    for key in ntexts:
        ntext += f'\n{key}: {ntexts[key]} '
    # add footer text with normalization info
    add_footer_text(fig, ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'trace_mask', title, description)

def trace_correction_sample(inst: Any, iframe: int,
                            cube: np.ndarray, recon: np.ndarray,
                            x_trace_pos: np.ndarray, y_trace_pos: np.ndarray,
                            x_order0: np.ndarray, y_order0: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.trace_correction'
    # set title and description
    title = 'Trace correction'
    description = (f'Example of the trace correction on integration {iframe}. '
                   'Top panel is the original integration, bottom panel is '
                   'the integration minus the linear reconstruction. '
                   'Orange points are the trace mask, red points are the '
                   'order 0 mask (if given).')
    # setup the figure
    fig, frames = plt.subplots(nrows=2, ncols=1, figsize=[12, 12])
    # plot the cube
    im0 = frames[0].imshow(cube[iframe], aspect='auto', origin='lower',
                           vmin=np.nanpercentile(cube[iframe], 1),
                           vmax=np.nanpercentile(cube[iframe], 95))
    frames[0].set(title=f'Integration {iframe}')
    # plot colorbar
    plt.colorbar(im0, ax=frames[0], orientation='vertical')
    # -------------------------------------------------------------------------
    # remove the recon temporarily for the plot
    tmp = cube[iframe] - recon
    # plot the cube minus the recon
    im1 = frames[1].imshow(tmp, aspect='auto', origin='lower',
                           vmin=np.nanpercentile(tmp, 5),
                           vmax=np.nanpercentile(tmp, 95))
    # plot colorbar
    plt.colorbar(im1, ax=frames[1], orientation='vertical')
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
    frames[1].set(title=f'Integration {iframe} - linear reconstruction')
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
    save_show_plot(inst.params, 'sample_frame{0}'.format(iframe),
                   title, description)


def aperture_correction_plot(inst: Any, outputs: Dict[str, Any],
                             trace_corr: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.aperture_correction_plot()'
    # set title and description
    title = 'Aperture correction'
    description = ('Aperture correction applied to the white light curve '
                   'amplitude coefficients. Top panel shows the amplitude '
                   'coefficients before and after correction. Bottom panel '
                   'shows the aperture correction applied (in ppt).')
    # -------------------------------------------------------------------------
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
    frames[0].set(title='Amplitude coefficient before and after correction.')
    frames[0].legend()
    # -------------------------------------------------------------------------
    # plot the trace correction
    frames[1].plot(tmp_trace_corr, 'r.', alpha=0.3)
    # convert the trace limits in to y limits on the graph
    ylim = [p12[0] - 0.3 * (p12[1] - p12[0]), p12[1] + 0.3 * (p12[1] - p12[0])]
    # set the title, labels and limits
    frames[1].set(title='Aperture correction', ylabel='corr [ppt]', ylim=ylim)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'aperture_correction', title, description)


def plot_trace_flux_loss(inst: Any, sums: np.ndarray,
                         dxs: np.ndarray, dys: np.ndarray,
                         xmax: int, ymax: int, loss_ppt: np.ndarray, 
                         tracemap: np.ndarray,
                         med: np.ndarray, best_dx: float, best_dy: float):
    # set function name
    func_name = f'{__NAME__}.plot_trace_flux_loss()'
    # set title and description
    title = 'Trace flux loss'
    description = ('Flux lost/gained in the white light curve aperture '
                   'as a function of trace position offset. Top panel is '
                   'the flux gained/lost as a function of x offset, middle '
                   'panel is the flux gained/lost as a function of y offset, '
                   'bottom panel is the 2D map of flux in the aperture as a '
                   'function of x and y offset. The best position is shown '
                   'as a red point in the bottom panel.')
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=3, ncols=1, figsize=[12, 12])
        # plot the offset of the trace
    frames[0].plot(dxs, loss_ppt[:, ymax])
    # set frame labels
    frames[0].set(xlabel='x offset of trace', ylabel='flux gained in ppt')
    # ---------------------------------------------------------------------
    # plot the offset of the trace
    frames[1].plot(dys, loss_ppt[xmax, :])
    # set frame labels
    frames[1].set(xlabel='y offset of trace', ylabel='flux gained in ppt')
    # ---------------------------------------------------------------------
    # get the limits of the sum array from dxs and dys
    extent = [np.min(dxs), np.max(dxs), np.min(dys), np.max(dys)]
    # plot the best_dx vs best_dy
    im = frames[2].imshow(sums.T, aspect='auto', extent=extent, origin='lower')
    # add the best point in red
    best_label = 'best position (x={0}, y={1})'.format(best_dx, best_dy)
    frames[2].plot(best_dx, best_dy, 'ro', label=best_label)
    # set frame labels
    frames[2].set(xlabel='dx', ylabel='dy', title='flux in aperture')
    # add a color bar
    plt.colorbar(im, ax=frames[2], orientation='vertical',
                 label='sum of white light flux')
    # add legend
    frames[2].legend()
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'trace_flux_loss', title, description)


def plot_fancy_centering1(inst: Any, xpix: np.ndarray, tracepos: np.ndarray,
                          traceois_fit: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.plot_fancy_centering1()'
    # set title and description
    title = 'Fancy centering step 1'
    description = ('Example of the fancy centering step 1. Top panel is the '
                   'trace position (red) and the fitted trace position (blue). '
                   'Bottom panel is the residuals between the two.')
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=2, ncols=1, sharex='all')
    # -------------------------------------------------------------------------
    frames[0].plot(xpix, tracepos, 'r-')
    frames[0].plot(xpix, traceois_fit, 'b-')
    frames[0].set(xlabel='xpix', ylabel='Trace position [pix]')
    # -------------------------------------------------------------------------
    frames[1].plot(xpix, tracepos - traceois_fit, 'g-')
    frames[1].set(xlabel='xpix', ylabel='Residuals [pix]')
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'fancy_centering1', title, description)


def plot_fancy_centering2(inst: Any, med: np.ndarray,
                          wave: np.ndarray, spectrum: np.ndarray,
                          x1: np.ndarray, y1: np.ndarray,
                          x2: np.ndarray, y2: np.ndarray):
    # set function name
    func_name = f'{__NAME__}.plot_fancy_centering2()'
    # set title and description
    title = 'Fancy centering step 2'
    description = ('Example of the fancy centering step 2. Top panel is the '
                   'extracted spectrum (black). Bottom panel is the '
                   'sqrt of the absolute median flux image (gray scale) '
                   'with the two traces overplotted (green and red).')
    # -------------------------------------------------------------------------
    # we want the sqrt of the absolute median flux
    sqrtabsim = np.sqrt(np.abs(med))
    # get the vmin and vmax
    vmin, vmax = np.nanpercentile(sqrtabsim, [1, 99])
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=2, ncols=1)
    # -------------------------------------------------------------------------
    frames[0].plot(wave, spectrum, 'k-')

    frames[1].imshow(sqrtabsim, origin='lower', cmap='gray',
                     aspect='auto', vmin=vmin, vmax=vmax)
    frames[1].plot(x1, y1, 'g-')
    frames[1].plot(x2, y2, 'r-')
    # -------------------------------------------------------------------------
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'fancy_centering2', title, description)


def plot_background(inst, frame0_before, frame0_after):
    # set function name
    func_name = f'{__NAME__}.plot_background1()'
    # set title and description
    title = 'Background correction'
    description = ('Example of the background correction on integration 0. '
                   'Top panel is before background correction, bottom panel '
                   'is after background correction.')
    # -------------------------------------------------------------------------
    inst.params['WLC.PLOT.BACKGROUND_VLIM'] = [1, 70]
    inst.params['WLC.PLOT.BACKGROUND_INTERVAL'] = 'zscale'

    # get the image normalization
    vlims = inst.params['WLC.PLOT.BACKGROUND_VLIM']
    vtype = inst.params['WLC.PLOT.BACKGROUND_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.BACKGROUND_INTERVAL']
    stretch = inst.params['WLC.PLOT.BACKGROUND_STRETCH']
    norm1, ntext1 = plot_normalization(frame0_before, interval=interval,
                                       stretch=stretch, vlims=vlims, vtype=vtype)
    norm2, ntext2 = plot_normalization(frame0_after, interval=interval,
                                       stretch=stretch, vlims=vlims, vtype=vtype)
    ntexts = dict()
    ntexts['before'] = ntext1
    ntexts['after'] = ntext2
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=2, ncols=1)
    # -------------------------------------------------------------------------
    # plot the before/after frames
    im0 = frames[0].imshow(frame0_before, origin='lower', cmap='inferno',
                           aspect='auto', norm=norm1)
    im1 = frames[1].imshow(frame0_after, origin='lower', cmap='inferno',
                           aspect='auto', norm=norm2)
    # plot colorbars
    plt.colorbar(im0, ax=frames[0], orientation='vertical')
    plt.colorbar(im1, ax=frames[1], orientation='vertical')
    # set title
    frames[0].set(title='Before background')
    frames[1].set(title='After background')
    # -------------------------------------------------------------------------
    # add footer text with normalization info
    ntext_str = ''
    for key in ntexts:
        ntext_str += f'\n{key}: {ntexts[key]} '
    add_footer_text(fig, ntext_str, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # force a tight layout leaving space at bottom for footer
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'background_corr', title, description)


def plot_lowpass(inst, frame0_before, frame0_after, sum_cube_tile):
    # set function name
    func_name = f'{__NAME__}.plot_lowpass()'
    # set title and description
    title = 'Low pass filter correction'
    description = ('Example of the low pass filter correction on integration 0. '
                   'Top panel is before low pass correction, middle panel is '
                   'after low pass correction, bottom panel is the average '
                   'low pass filter corrections applied to all integrations.')
    # -------------------------------------------------------------------------
    # get the image normalization
    vlims = inst.params['WLC.PLOT.LOWPASS_VLIM']
    vtype = inst.params['WLC.PLOT.LOWPASS_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.LOWPASS_INTERVAL']
    stretch = inst.params['WLC.PLOT.LOWPASS_STRETCH']
    norm, ntext = plot_normalization(frame0_before, interval=interval, 
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    # -------------------------------------------------------------------------
    # plot the before/after frames
    im0 = frames[0].imshow(frame0_before, origin='lower', cmap='inferno',
                           aspect='auto', norm=norm)
    im1 = frames[1].imshow(frame0_after, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    # set title
    frames[0].set(title='Before low pass')
    frames[1].set(title='After low pass')
    # plot color bars
    plt.colorbar(im0, ax=frames[0], orientation='vertical')
    plt.colorbar(im1, ax=frames[1], orientation='vertical')
    # -------------------------------------------------------------------------
    # plot the sum of the low pass filter
    im2 = frames[2].imshow(sum_cube_tile, origin='lower', cmap='inferno',
                           aspect='auto')
    frames[2].set(title='Average low pass filter corrections')
    # add a colorbar to frames 2
    plt.colorbar(im2, ax=frames[2], orientation='vertical')
    # -------------------------------------------------------------------------
    # add footer text with normalization info
    add_footer_text(fig, ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # force a tight layout leaving space at bottom for footer
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'lowpass_corr', title, description)


def plot_flat_field(inst, frame0_before, frame0_after):
    # set function name
    func_name = f'{__NAME__}.plot_flat_field()'
    # set title and description
    title = 'Flat field correction'
    description = ('Example of the flat field correction on integration 0. '
                   'Top panel is before flat field correction, bottom panel is '
                   'after flat field correction.')
    # -------------------------------------------------------------------------
    # get the image normalization
    vlims = inst.params['WLC.PLOT.FLAT_VLIM']
    vtype = inst.params['WLC.PLOT.FLAT_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.FLAT_INTERVAL']
    stretch = inst.params['WLC.PLOT.FLAT_STRETCH']
    norm, ntext = plot_normalization(frame0_before, interval=interval, 
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    # -------------------------------------------------------------------------
    # setup the plot
    fig, frames = plt.subplots(nrows=2, ncols=1)
    # -------------------------------------------------------------------------
    # plot the before/after frames
    frames[0].imshow(frame0_before, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    frames[1].imshow(frame0_after, origin='lower', cmap='inferno',
                     aspect='auto', norm=norm)
    # set title
    frames[0].set(title='Before flatfield')
    frames[1].set(title='After flatfield')
    # -------------------------------------------------------------------------
    # add footer text with normalization info
    add_footer_text(fig, ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # force a tight layout leaving space at bottom for footer
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'flatfield', title, description)


def plot_heatmap(inst: Any, heat_map: np.ndarray, iframe_before: np.ndarray,
                 iframe_after: np.ndarray,
                 title: str, outname: str, clabel: str):
    """
    Plot a heatmap of the bad pixels

    :param inst: Instrument instance
    :param heat_map: np.ndarray, the heat map of the bad pixels
    :param iframe: np.ndarray, a comparison frame
    :param title: str, the title of the plot
    :param outname: str, the output name for the plot

    :return: None, plots graph
    """
    # set title and description
    description = (f'Top panel is the heat map of {title}. '
                   f'Middle panel is a comparison frame before '
                   'correction, bottom panel is the same frame after '
                   'correction.')
    # set up figure
    fig, frames = plt.subplots(ncols=1, nrows=3, figsize=(12, 12))
    # -------------------------------------------------------------------------
    # get the image normalization
    vlims = inst.params['WLC.PLOT.FRAME_VLIM']
    vtype = inst.params['WLC.PLOT.FRAME_VLIM_TYPE']
    interval = inst.params['WLC.PLOT.FRAME_INTERVAL']
    stretch = inst.params['WLC.PLOT.FRAME_STRETCH']
    norm, ntext = plot_normalization(iframe_before, interval=interval,
                                     stretch=stretch, vlims=vlims, vtype=vtype)
    # -------------------------------------------------------------------------
    # plot the heat map
    im0 = frames[0].imshow(heat_map, origin='lower', cmap='inferno',
                          aspect='auto', interpolation='none')
    # add a color bar
    plt.colorbar(im0, ax=frames[0], orientation='horizontal',
                 label=clabel)
    # set the title
    frames[0].set(title=title)
    # -------------------------------------------------------------------------
    # plot a comparison frame (before)
    im1 = frames[1].imshow(iframe_before, origin='lower', cmap='inferno',
                          aspect='auto', interpolation='none',
                          norm=norm)
    frames[1].set(title='Before correction')
    # -------------------------------------------------------------------------
    # plot a comparison frame (before)
    im2 = frames[2].imshow(iframe_after, origin='lower', cmap='inferno',
                          aspect='auto', interpolation='none',
                          norm=norm)
    frames[2].set(title='After correction')
    # -------------------------------------------------------------------------
    # add footer text with normalization info
    add_footer_text(fig, 'before: ' + ntext, fontsize=8, pad=0.01)
    # -------------------------------------------------------------------------
    # force a tight layout leaving space at bottom for footer
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, outname, title, description)


def plot_pixels(inst: Any, pixel_dict: Dict[int, np.ndarray],
                frame_num: int = 0):
    # set title and description
    title = 'Bad pixel correction stamps'
    description = (f'Example of the bad pixel correction stamps for '
                   f'pixel {frame_num}. Each stamp is a '
                   f'{inst.params["WLC.GENERAL.PATCH_IBADS_SSIZE"]}x'
                   f'{inst.params["WLC.GENERAL.PATCH_IBADS_SSIZE"]} '
                   f'stamp of the bad pixel and its surrounding pixels.')
    # -------------------------------------------------------------------------
    # set up figure
    fig, frame = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    # plot this frame
    im = plt.imshow(pixel_dict[frame_num], origin='lower', cmap='inferno',
                    aspect='auto', interpolation='none')
    # add a color bar
    plt.colorbar(im, ax=frame, orientation='vertical', label='Flux')
    # add title
    targs = [frame_num, inst.params['WLC.GENERAL.PATCH_IBADS_SSIZE']]
    frame.set(title=title.format(*targs))
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'isolated_pixel_corr_stamps',
                   title, description)

def plot_stability(inst: Any, table: Table):
    # set title and description
    title = 'Stability of white light curve parameters'
    description = ('Stability of the white light curve parameters as a '
                   'function of integration number. Each panel is a different '
                   'parameter with error bars. If baseline domain is '
                   'defined these are highlighted in green.')
    # set function name
    func_name = f'{__NAME__}.plot_stability()'
    # validate out-of-transit domain
    inst.get_baseline_params()
    has_baseline = inst.get_variable('HAS_BASELINE', func_name)
    baseline_domain = inst.get_variable('BASELINE_DOMAIN', func_name)
    # -------------------------------------------------------------------------
    # get the output names, units and factors
    output_names = inst.get_variable('OUTPUT_NAMES', func_name)
    output_units = inst.get_variable('OUTPUT_UNITS', func_name)
    output_factor = inst.get_variable('OUTPUT_FACTOR', func_name)
    # get object name and suffix
    objname = inst.params['INPUTS.OBJECTNAME']
    suffix = inst.params['INPUTS.SUFFIX']
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
    if inst.params['GENERAL.WLC_DOMAIN'] is not None:
        dargs = [inst.params['GENERAL.WLC_DOMAIN'][0],
                 inst.params['GENERAL.WLC_DOMAIN'][1],
                 inst.params['INPUTS.SID']]
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
        # deal with having baseline points
        if has_baseline:
            # plot the out of transit points
            frames[it].errorbar(index[baseline_domain],
                                value[baseline_domain],
                                yerr=errvalue[baseline_domain],
                                fmt='.', color='green', alpha=alpha,
                                label='baseline integrations')
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
            title = f'{objname} -- {suffix}\n'
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
    save_show_plot(inst.params, 'stability', title, description)


# def plot_transit(inst: Any, table: Table):
#     # set function name
#     func_name = f'{__NAME__}.plot_transit()'
#     # validate out-of-transit domain
#     inst.get_baseline_params()
#     has_oot = inst.get_variable('HAS_OUT_TRANSIT', func_name)
#     has_int = inst.get_variable('HAS_IN_TRANSIT', func_name)
#     out_transit_domain = inst.get_variable('OOT_DOMAIN', func_name)
#     in_transit_domain = inst.get_variable('INT_DOMAIN', func_name)
#     baseline_ints = inst.get_variable('BASELINE_INTS', func_name)
#     transit_ints = inst.get_variable('TRANSIT_INTS', func_name)
#     # get wlc_params
#     wlc_params = inst.params.get('WLC')
#     # get object name and suffix
#     objname = inst.params['INPUTS.OBJECTNAME']
#     suffix = inst.params['INPUTS.SUFFIX']
#     # get the polynomial degree for the transit baseline
#     poly_order = wlc_params['GENERAL.TRANSIT_BASELINE_POLYORD']
#     # -------------------------------------------------------------------------
#     # get the number of points
#     npoints = len(table['amplitude'])
#     # -------------------------------------------------------------------------
#     # get the amplitude and error values
#     value = table['amplitude']
#     errvalue = table['amplitude_error']
#     # get the index of the pixels
#     index = np.arange(npoints)
#     # -------------------------------------------------------------------------
#     # deal with no out-of-transit defined
#     if not has_oot:
#         out_transit_domain = np.ones_like(index, dtype=bool)
#     # -------------------------------------------------------------------------
#     # 5-sigma robust poly fit of the continuum
#     ampfit, _ = mp.robust_polyfit(index[out_transit_domain],
#                                   value[out_transit_domain],
#                                   degree=poly_order, nsigcut=5)
#     # remove this fit from the amplitude
#     value = value / np.polyval(ampfit, index)
#     # -------------------------------------------------------------------------
#     # deal with no transit
#     if has_int:
#         # storage for mid transits/eclipses
#         mid_transits, fit_mids, mid_transit_depths = [], [], []
#         # loop around transits/eclipses
#         for cframe in transit_ints:
#             # calculate the mid-transit frames
#             norm_index = index - (cframe[0] + cframe[3]) / 2
#             mid_transit = np.abs(norm_index) < 0.3 * (cframe[3] - cframe[0])
#             # -----------------------------------------------------------------
#             # fit the mid transit frames
#             fit_mid, _ = mp.robust_polyfit(index[mid_transit],
#                                            value[mid_transit],
#                                            degree=2, nsigcut=5)
#             # -----------------------------------------------------------------
#             # calculate the mid-transit point and depth
#             mid_transit_point = -0.5 * fit_mid[1] / fit_mid[0]
#             mid_transit_depth = np.polyval(fit_mid, mid_transit_point)
#             # -----------------------------------------------------------------
#             # append to lists
#             mid_transits.append(mid_transit)
#             fit_mids.append(fit_mid)
#             mid_transit_depths.append(mid_transit_depth)
#
#     else:
#         mid_transits, fit_mids, mid_transit_depths = [], [], []
#
#
#     # -------------------------------------------------------------------------
#     # setup the plot
#     fig, frame = plt.subplots(nrows=1, ncols=1, figsize=[8, 4])
#     # -------------------------------------------------------------------------
#     # plot the out-of-transit
#     if has_oot:
#         frame.errorbar(index[out_transit_domain],
#                        value[out_transit_domain],
#                        yerr=errvalue[out_transit_domain],
#                        fmt='.', color='green', alpha=0.4, label='oot', zorder=3)
#     # otherwise just plot the transit
#     else:
#         frame.errorbar(index, value, yerr=errvalue,
#                        fmt='.', color='green', alpha=0.4, label='oot', zorder=3)
#     # -------------------------------------------------------------------------
#     # plot the in transit (if we have it)
#     if has_int:
#         frame.errorbar(index[in_transit_domain],
#                        value[in_transit_domain],
#                        yerr=errvalue[in_transit_domain],
#                        fmt='.', color='red', alpha=0.4, label='it', zorder=2)
#         # plot the transit fit
#         for it in range(len(mid_transits)):
#             frame.plot(index[mid_transits[it]],
#                        np.polyval(fit_mids[it], index[mid_transits[it]]),
#                        'k--', zorder=10)
#         # add a legend
#         frame.legend()
#
#
#     # ---------------------------------------------------------------------
#     # axis labels, title and grid
#     # ---------------------------------------------------------------------
#     # get the y limits
#     ylim = cal_y_limits(value, errvalue)
#     # get the title for the plot
#     title = f'{objname} -- {suffix}\n'
#     sub_strs = []
#     for m_it, mid_transit_depth in enumerate(mid_transit_depths):
#         sub_strs.append(f'Transit-{m_it+1}: {(1-mid_transit_depth)*1e6:.0f} ppm')
#     title += '\n'.join(sub_strs)
#
#     # set the axis
#     frame.set(xlabel='Nth frame', ylabel='Baseline-corrected flux', ylim=ylim,
#               title=title)
#     # set up the grid
#     frame.grid(linestyle='--', color='grey', zorder=-99)
#     # force a tight layout
#     plt.tight_layout()
#     # -------------------------------------------------------------------------
#     # standard save/show plot for SOSSISSE
#     save_show_plot(inst.params, 'transit')

def plot_spectral_timeseries(inst: Any, spec2: np.ndarray, trace_order: int):
    # set function name
    func_name = f'{__NAME__}.plot_spectral_timeseries()'
    # set title and description
    title = 'Spectral time series'
    description = (f'Spectral time series for trace order {trace_order}. '
                   'Each row is a different integration, each column is a '
                   'different pixel in the spectral direction.')
    # -------------------------------------------------------------------------
    # get object name and suffix
    objname = inst.params['INPUTS.OBJECTNAME']
    suffix = inst.params['INPUTS.SUFFIX']
    # set up the plot
    fig, frame = plt.subplots(figsize=(10, 5))
    # plot the spectral time series
    vmin = np.nanpercentile(spec2, 2.5)
    vmax= np.nanpercentile(spec2, 97.5)
    frame.imshow(spec2, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax,
                 aspect='auto', interpolation='none')
    frame.set_xlabel('x pixel')
    frame.set_ylabel('Integration number')
    # plot colorbar
    plt.colorbar(frame.images[0], ax=frame, orientation='vertical',
                 label='Rel. flux')
    # construct title
    title = f'{objname} -- {suffix} spectral time series order {trace_order}'
    frame.set(title=title)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, f'spectral_timeseries_ord{trace_order}',
                   title, description)

def plot_sed(inst: Any, wavegrid: np.ndarray, sed: np.ndarray,
             trace_order: int):
    # set function name
    # func_name = f'{__NAME__}.plot_sed()'
    # set title and description
    title = 'Spectral energy distribution'
    description = (f'Spectral energy distribution for trace order '
                   f'{trace_order}. This is the flux as a function of '
                   'wavelength, corrected for the instrument throughput.')
    # -------------------------------------------------------------------------
    # get object name and suffix
    objname = inst.params['INPUTS.OBJECTNAME']
    suffix = inst.params['INPUTS.SUFFIX']
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
    save_show_plot(inst.params, 'sed_{0}_ord{1}'.format(objname, trace_order),
                   title, description)


def plot_full_sed(inst: Any, plot_storage: Dict[int, Dict[str, Any]]):
    # set title and description
    title = 'Full Spectral energy distribution'
    description = ('Full spectral energy distribution for all trace orders. '
                   'This is the flux as a function of wavelength, corrected '
                   'for the instrument throughput.')
    # -------------------------------------------------------------------------
    # set up the plot
    fig, frame = plt.subplots(nrows=1, ncols=1)
    # get object name and suffix
    objname = inst.params['INPUTS.OBJECTNAME']
    suffix = inst.params['INPUTS.SUFFIX']
    # loop around tarce orders
    for trace_order in plot_storage.keys():

        # deal with trace order
        if trace_order == 0:
            pkwargs1 = dict(color='b', marker='.', ls='None')
            pkwargs2 = dict(color='orange', marker='o', ls='--')
        elif trace_order == 1:
            pkwargs1 = dict(color='g', marker='.', ls='None')
            pkwargs2 = dict(color='r', marker='o', ls='--')
        elif trace_order == 2:
            pkwargs1 = dict(color='c', marker='.', ls='None')
            pkwargs2 = dict(color='m', marker='o', ls='--')
        else:
            continue
        # get this trace orders parameters
        wavegrid = plot_storage[trace_order]['wavegrid']
        sed_spec = plot_storage[trace_order]['sp_sed']
        throughtput = plot_storage[trace_order]['throughput']
        spec_in = plot_storage[trace_order]['spec']
        spec_err_in = plot_storage[trace_order]['spec_err']
        #transit_depth = plot_storage[trace_order]['transit_depth']
        #wave_bin = plot_storage[trace_order]['wave_bin']
        #flux_bin = plot_storage[trace_order]['flux_bin']
        #flux_bin_err = plot_storage[trace_order]['flux_bin_err']
        # plot the SED
        frame.plot(wavegrid, sed_spec / throughtput, color='k',
                   label='Flux, throughput-corrected, '
                         'order {0}'.format(trace_order))
    # -------------------------------------------------------------------------
    # construct title
    title = f'{objname} -- {suffix}'

    plt.legend(loc=0)
    # set the axis labels
    frame.set(xlabel=r'Wavelength [$\mu$m]', ylabel='ppm',
              title=title)
    # force a tight layout
    plt.tight_layout()
    # -------------------------------------------------------------------------
    # standard save/show plot for SOSSISSE
    save_show_plot(inst.params, 'sed_{0}'.format(objname), title, description)


# =============================================================================
# Define the interactive transit plot functions
# =============================================================================
# class InteractiveTransitPlot:
#     def __init__(self, **kwargs):
#         # get values out of kwargs
#         self.x = np.arange(len(kwargs['amps']))
#         self.y = kwargs['amps']
#         self.yerr = kwargs['eamps']
#         self.mask = kwargs['baseline_domain']
#         # Set title
#         self.title = ('Pick groups of 4 transit integrations'
#                       '\n1: First Contact, 2: Second Contact,'
#                       '3: Third Contact, 4: Fourth Contact'
#                       '\n\nObject name = {0}'.format(kwargs['OBJECTNAME']))
#
#         # Store selected points
#         self.selected_points = []
#         self.lines = []
#         self.fig = None
#         self.frame = None
#         self.frame_reset = None
#         self.frame_accept = None
#         self.btn_reset = None
#         self.btn_accept = None
#         # store outputs
#         self.success = False
#         self.transit_ints = []
#
#     def plot(self):
#         # try to do the plot
#         try:
#             # close any previously open plots
#             plt.close()
#             # Create figure and plot
#             self.fig, self.frame = plt.subplots()
#             plt.subplots_adjust(bottom=0.2)
#
#             # plot out of transit domain in blue
#             self.frame.errorbar(self.x[self.mask], self.y[self.mask],
#                                 yerr=self.yerr[self.mask],
#                                 linestyle='None', marker='o', color='b',
#                                 label='BASELINE_INTS')
#             # plot rejected points in black
#             self.frame.errorbar(self.x[~self.mask], self.y[~self.mask],
#                                 yerr=self.yerr[~self.mask],
#                                 linestyle='None', marker='o', color='b',
#                                 label='Rest of domain')
#             # set title
#             self.frame.set(xlabel='Integration number',
#                            ylabel='Flux',
#                            title=self.title)
#             # Create buttons
#             self.frame_reset = plt.axes([0.3, 0.05, 0.2, 0.075])
#             self.frame_accept = plt.axes([0.55, 0.05, 0.2, 0.075])
#             self.btn_reset = Button(self.frame_reset, 'Reset')
#             self.btn_accept = Button(self.frame_accept, 'Accept')
#
#             self.btn_reset.on_clicked(self.reset)
#             self.btn_accept.on_clicked(self.accept)
#
#             self.fig.canvas.mpl_connect('button_press_event', self.on_click)
#             plt.show(block=True)
#         except Exception as e:
#             misc.printc(str(e), 'error')
#             self.success = False
#
#     def on_click(self, event):
#         """Handles mouse clicks to select points."""
#         if event.inaxes != self.frame:
#             return
#
#         x_selected = event.xdata
#         self.selected_points.append(x_selected)
#
#         line = self.frame.axvline(x_selected, color='r', linestyle='--')
#         self.lines.append(line)
#         self.fig.canvas.draw()
#
#     def reset(self, event):
#         """Clears selected points and removes lines."""
#         _ = event
#         self.selected_points.clear()
#         for line in self.lines:
#             line.remove()
#         self.lines.clear()
#         self.fig.canvas.draw()
#
#     def accept(self, event):
#         """Accepts selections and closes the plot."""
#         _ = event
#         # ask user whether they want to continue
#         if self.try_again():
#             return
#         # close the
#         plt.close(self.fig)
#         # set success to True
#         self.success = True
#         # sort selected points
#         selected_points = list(self.selected_points)
#         selected_points.sort()
#         # storage for transit groups
#         transit_group = []
#         # loop through point and make them integers
#         for point in selected_points:
#             if len(transit_group) < 4:
#                 transit_group.append(int(point))
#             else:
#                 self.transit_ints.append(transit_group)
#                 transit_group = [int(point)]
#         # sort in ascending order
#         self.transit_ints.sort()
#
#     def try_again(self) -> bool:
#         """
#         Ask user if they want to continue
#         :return:
#         """
#         # deal with having 4 points (continue)
#         if len(self.selected_points) % 4 == 0:
#             return False
#         # try to create a warning message box
#         try:
#             import tkinter as tk
#             from tkinter import messagebox
#             root = tk.Tk()
#             root.withdraw()
#             title = 'Selection Error'
#             msg = ('Please select groups of exactly 4 points. '
#                    '\nDo you want to continue selecting?')
#             uinput = messagebox.askquestion(title, msg, icon='warning')
#             if uinput == 'no':
#                 self.success = False
#                 self.transit_ints = [[]]
#                 return False
#         except Exception as e:
#             misc.printc(str(e), 'error')
#             self.success = False
#             return False
#         # if we get here return True
#         return True
