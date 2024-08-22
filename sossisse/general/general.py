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
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from sossisse.core import math
from sossisse.core import misc
from sossisse.general import plots, science, soss_io

from sossisse.instruments import Instrument


def white_light_curve(inst: Instrument):
    # pass either the param file or the params themselves

    # check the sample parameter files for guidance on proper keywords
    # set force = True to force a re-writing of the temporary files
    # meant to speed to code
    misc.sossart()

    # get parameters from instrumental parameters
    objname = inst.params['OBJECTNAME']
    # print the white light curve splash
    print(misc.art('White light curve ' + objname, 'blue', 'CYAN'))

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

    # TODO: **************************************************************
    # TODO: Got to here
    # TODO: **************************************************************

    if params['pixel_level_detrending']:
        cube = science.pixeldetrending(cube, params)

    # if you want to subtract a higher order polynomial to the 1/f noise, change
    # the value of fit_order
    cube, med, med_diff, diff_in_out, params = science.clean_1f(cube, err, params)

    if params['recenter_trace_position']:
        misc.printc('Scan to optimize position of trace', 'info')

        width_current = np.array(params['trace_width_masking'])
        params['trace_width_masking'] = 20
        dys = np.arange(-params['DATA_Y_SIZE'] // 10, params['DATA_Y_SIZE'] // 10 + 1)
        dxs = np.arange(-params['DATA_Y_SIZE'] // 10, params['DATA_Y_SIZE'] // 10 + 1)
        sums = np.zeros([len(dxs), len(dys)], dtype=float)

        best_dx = 0
        best_dy = 0
        best_sum = 0
        for ix in tqdm(range(len(dxs)), leave=False):
            for iy in tqdm(range(len(dys)), leave=False):
                params['x_trace_offset'] = dxs[ix]
                params['y_trace_offset'] = dys[iy]
                params = science.get_trace_map(params, silent=True)
                sums[ix, iy] = np.nansum(params['TRACEMAP'] * med)
                if sums[ix, iy] > best_sum:
                    best_sum = sums[ix, iy]
                    best_dx = dxs[ix]
                    best_dy = dys[iy]

        params['x_trace_offset'] = best_dx
        params['y_trace_offset'] = best_dy
        params['trace_width_masking'] = width_current

        params = science.get_trace_map(params)  # refresh trace map
        misc.printc('Best dx : {} pix'.format(best_dx), 'number')
        misc.printc('Best dy : {} pix'.format(best_dy), 'number')

        """
        loss_ppt = (1-sums/np.nanmax(sums))*1e3
        nmax = 8
        if len(loss_ppt) < nmax:
            nmax = len(loss_ppt)
        ylim = loss_ppt[np.argsort(loss_ppt)][nmax]
        if ylim < loss_ppt[dys == 0]:
            ylim = loss_ppt[dys == 0]*1.1

        for i in range(len(dys)):
            if loss_ppt[i]<ylim:
                printc('\toffset dy = {:3}, err = {:.3f} ppt'.format(dys[i], loss_ppt[i]),'number')
        printc('We scanned the y position of trace, optimum at dy = {}'.format(params['y_trace_offset']),'number')

        fig,ax = plt.subplots(nrows = 2, ncols =1, figsize = [8,8])
        ax[0].plot(dys,loss_ppt)
        ax[0].set(xlabel = 'offset of trace',ylabel = 'flux loss in ppt',ylim =[0,ylim*1.1])
        mask = np.array(params['TRACEMAP'],dtype = float)
        mask[mask == 0] = np.nan
        ax[1].imshow(med*mask,aspect = 'auto')
        plt.tight_layout()
        for figtype in params['figure_types']:
            outname = '{}/trace_flux_loss_{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype)
            plt.savefig(outname)
        if params['show_plots']:
            plt.show()
        plt.close()
        """

    ################################################################################
    # Part of the code that does rotation/shift/amplitude
    ################################################################################
    dx, dy, rotxy, ddy, med_clean = science.get_gradients(med, params)

    mask_trace_pos = np.ones_like(med, dtype=int)

    if params['trace_width_masking'] != 0:
        mask_trace_pos[~params['TRACEMAP']] = 0
        y_trace_pos, x_trace_pos = np.where(binary_dilation(mask_trace_pos, [[0, 1, 0], [1, 1, 1], [0, 1,
                                                                                                    0]]) != mask_trace_pos)

    if params['mask_order_0']:
        # adding the masking of order 0
        mask_order0, x_order0, y_order0 = science.get_mask_order0(params)
        mask_trace_pos[mask_order0] = 0
    else:
        # if we don't have a mask, we set dummy values for the plot later in the code
        x_order0 = [np.nan]
        y_order0 = [np.nan]

    # vectors for the linear reconstruction
    v = med.ravel()

    if params['fit_dx']:
        v = np.append(v, dx.ravel())
        params['output_names'] = np.append(params['output_names'], 'dx')
        params['output_units'] = np.append(params['output_units'], 'mpix')
        params['output_factor'] = np.append(params['output_factor'], 1e3)

    if params['fit_dy']:
        v = np.append(v, dy.ravel())
        params['output_names'] = np.append(params['output_names'], 'dy')
        params['output_units'] = np.append(params['output_units'], 'mpix')
        params['output_factor'] = np.append(params['output_factor'], 1e3)

    if params['fit_rotation']:
        v = np.append(v, rotxy.ravel())
        params['output_names'] = np.append(params['output_names'], 'theta')
        params['output_units'] = np.append(params['output_units'], 'arcsec')
        params['output_factor'] = np.append(params['output_factor'], 1296000 / (np.pi * 2))

    if params['zero_point_offset']:
        v = np.append(v, np.ones_like(dx.ravel()))
        params['output_names'] = np.append(params['output_names'], 'zero point')
        params['output_units'] = np.append(params['output_units'], 'flux')
        params['output_factor'] = np.append(params['output_factor'], 1)

    if params['ddy']:
        v = np.append(v, ddy.ravel())
        params['output_names'] = np.append(params['output_names'], 'ddy')
        params['output_units'] = np.append(params['output_units'], 'mpix$^2$')
        params['output_factor'] = np.append(params['output_factor'], 1e6)

    if params['before_after']:
        v = np.append(v, med_diff.ravel())
        params['output_names'] = np.append(params['output_names'], 'before-after')
        params['output_units'] = np.append(params['output_units'], 'ppm')
        params['output_factor'] = np.append(params['output_factor'], 1e6)

    if params['fit_pca']:
        for ipca in range(params['n_pca']):
            v = np.append(v, params['PCA_components'][ipca].ravel())
            params['output_names'] = np.append(params['output_names'], 'PCA{}'.format(ipca + 1))
            params['output_units'] = np.append(params['output_units'], 'ppm')
            params['output_factor'] = np.append(params['output_factor'], '1.0')

    if params['quadratic_term']:
        v = np.append(v, med_diff.ravel() ** 2)
        params['output_names'] = np.append(params['output_names'], 'flux^2')
        params['output_units'] = np.append(params['output_units'], 'flux^2')
        params['output_factor'] = np.append(params['output_factor'], 1.0)

    v = v.reshape([len(v) // len(med.ravel()), len(med.ravel())])

    # vectors to keep track of the rotation/amplitudes/dx/dy
    all_recon = np.zeros_like(cube)

    # adding place-holders for the output table
    tbl = Table()
    for i in range(len(params['output_names'])):
        tbl[params['output_names'][i]] = np.zeros(cube.shape[0])
        tbl[params['output_names'][i] + '_error'] = np.zeros(cube.shape[0])

    #
    tbl['rms_cube_recon'] = np.zeros(cube.shape[0])
    tbl['sum_trace'] = np.zeros(cube.shape[0])
    tbl['sum_trace_error'] = np.zeros(cube.shape[0])

    tbl['amplitude_no_model'] = np.zeros(cube.shape[0])
    tbl['amplitude_no_model error'] = np.zeros(cube.shape[0])

    # i = 6976

    trace_corr = np.zeros(params['DATA_Z_SIZE'], dtype=float)
    for i in tqdm(range(cube.shape[0]), leave=False):
        # find the best combination of scale/dx/dy/rotation
        # amps is a vector with the amplitude of all 4 fitted terms
        # amps[0] -> amplitude of trace
        # amps[1] -> dx normalized on reference trace
        # amps[2] -> dy normalized on reference trace
        # amps[3] -> rotation (in radians) normalized on reference trace
        # amps[4] -> 2nd derivative in y [if option activated]

        mask = np.array(np.isfinite(cube[i]), dtype=float)
        # mask = np.array(cube_mask[i] , dtype=float)

        mask[mask != 1] = np.nan
        mask[mask_trace_pos == 0] = np.nan

        with warnings.catch_warnings(record=True) as _:
            tbl['sum_trace'][i] = np.nansum(cube[i] * mask)
            tbl['sum_trace_error'][i] = np.sqrt(np.nansum(err[i] ** 2 * mask))

        with warnings.catch_warnings(record=True) as _:
            amp0 = math.odd_ratio_mean(cube[i].ravel() / med.ravel() * mask.ravel(), err[i].ravel() / med.ravel())
            tbl['amplitude_no_model'][i] = amp0[0]
            tbl['amplitude_no_model error'][i] = amp0[1]

        # recon = recon.reshape(med.shape)
        # >5 sigma clipping of linear system
        bad = np.abs((cube[i] - med * amp0[0]) / err[i]) > 5
        mask[bad] = np.nan
        amp_model, err_model, recon = math.lin_mini_errors(cube[i] * mask, err[i], v)

        if params['zero_point_offset']:
            recon -= amp_model[params['output_names'] == 'zero point']

        trace_corr[i] = np.nansum(recon * mask / err[i] ** 2) / np.nansum(med * amp_model[0] * mask / err[i] ** 2)

        if i == 0:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[12, 12])
            ax[0].imshow(cube[i], vmin=np.nanpercentile(cube[i], 1), vmax=np.nanpercentile(cube[i], 95), aspect='auto',
                         origin='lower')
            ax[0].set(title='Sample Image')
            tmp = cube[i] - recon
            ax[1].imshow(tmp, vmin=np.nanpercentile(tmp, 5), vmax=np.nanpercentile(tmp, 95),
                         aspect='auto',
                         origin='lower')

            ax[0].plot(x_trace_pos, y_trace_pos, '.', color='orange', alpha=0.2)
            ax[1].plot(x_trace_pos, y_trace_pos, '.', color='orange', alpha=0.2, label='trace mask')
            if len(x_order0) > 2:
                ax[0].plot(x_order0, y_order0, 'r.', alpha=0.1)
                ax[1].plot(x_order0, y_order0, 'r.', alpha=0.1, label='order 0')
            ax[1].legend()
            ax[1].set(title='Residual')
            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)

            plt.tight_layout()
            for figtype in params['figure_types']:
                # TODO: you shouldn't use '/'  as it is OS dependent
                # TODO: use os.path.join(1, 2, 3)
                outname = '{}/sample_{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype)
                plt.savefig(outname)
            if params['show_plots']:
                plt.show()
            plt.close()

        for j in range(len(amp_model)):
            tbl[params['output_names'][j]][i] = amp_model[j]
            tbl[params['output_names'][j] + '_error'][i] = err_model[j]

        cube[i] -= recon
        all_recon[i] = recon / amp_model[0]

        tbl['rms_cube_recon'][i] = math.estimate_sigma(np.array(cube[i], dtype=float))
    trace_corr /= np.nanmedian(trace_corr)

    tbl['amplitude_uncorrected'] = np.array(tbl['amplitude'])
    tbl['aperture_correction'] = trace_corr
    tbl['amplitude'] = tbl['amplitude'] * trace_corr
    yerr = tbl['amplitude_error']
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=[10, 10])
    ax[0].errorbar(np.arange(len(tbl)), tbl['amplitude_uncorrected'], yerr=yerr, fmt='r.', alpha=0.3,
                   label='uncorrected')
    ax[0].set(title='amplitude')
    ax[0].errorbar(np.arange(len(tbl)) + 0.5, tbl['amplitude'], yerr=yerr, fmt='g.', alpha=0.3, label='corrected')
    ax[0].legend()
    tmp = 1e3 * (trace_corr - 1)
    ax[1].plot(tmp, 'r.', alpha=0.3)
    p12 = np.nanpercentile(tmp, [1, 99])
    ylim = [p12[0] - 0.3 * (p12[1] - p12[0]), p12[1] + 0.3 * (p12[1] - p12[0])]
    ax[1].set(title='Apperture correction', ylabel='corr [ppt]', ylim=ylim)
    plt.tight_layout()
    for figtype in params['figure_types']:
        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        outname = '{}/apperture_correction{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype)
        plt.savefig(outname)
    if params['show_plots']:
        plt.show()
    plt.close()

    # normalizing the sub of trace
    if 'oot_domain' not in params.keys():
        params = science.get_valid_oot(params)

    with warnings.catch_warnings(record=True) as _:
        norm_factor = np.nanmedian(tbl[params['oot_domain']]['sum_trace'])
    tbl['sum_trace'] /= norm_factor
    tbl['sum_trace_error'] /= norm_factor

    if ['per_pixel_baseline_correction']:
        misc.printc('Performing per-pixel baseline subtraction', 'info')
        cube = science.per_pixel_baseline(cube, mask, params)

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    outname = '{}/errormap{}.fits'.format(params['TEMP_PATH'], params['tag'])
    if not os.path.isfile(outname):
        fits.writeto(outname, err, overwrite=True)

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    outname = '{}/residual{}.fits'.format(params['TEMP_PATH'], params['tag'])
    if not os.path.isfile(outname):
        fits.writeto(outname, cube, overwrite=True)

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    outname = '{}/recon{}.fits'.format(params['TEMP_PATH'], params['tag'])
    if not os.path.isfile(outname):
        fits.writeto(outname, all_recon, overwrite=True)

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    tbl.write('{}/soss_stability{}.csv'.format(params['CSV_PATH'], params['tag']), overwrite=True)
    tbl.write('{}/soss_stability{}.csv'.format(params['PLOT_PATH'], params['tag']), overwrite=True)

    # gives the list of methods if you don't know the one to use
    for method in science.get_rms_baseline():
        rms1 = science.get_rms_baseline(tbl['amplitude'], method=method)
        misc.printc('{0}, rms = {1:.1f}ppm'.format(method, rms1 * 1e6), 'number')

    plots.plot_sossice(tbl, params)
    plots.plot_transit(tbl, params)

    science.get_effective_wavelength(params)

    soss_io.yaml_to_html(params)

    return params, tbl


def wrapper(param_file, force=False):
    params = soss_io.load_yaml_params(param_file)

    # do we need to perform sossice at all? We check if the final file
    # exists

    outfile = '{}/soss_stability{}.csv'.format(params['CSV_PATH'], params['tag'])

    todo = not os.path.isfile(outfile)
    if force:
        todo = True

    if todo:
        _ = white_light_curve(params, force=force)
    else:
        misc.printc('\t\n\nFile {} exists we skip to sossisson \n\n'.format(outfile), 'bad3')

    spectral_extraction(params, force=force)


def summary(obj, search_string='*'):

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    yaml_files = np.array(glob.glob('*/{}/*/*/*{}*yaml'.format(obj, search_string)))
    yaml_files = yaml_files[[not os.path.islink('/'.join(file.split('/')[0:3])) for file in yaml_files]]

    if len(yaml_files) == 0:
        raise ValueError("We did not find yaml files ...")

    tbl = Table()
    tbl['yaml files'] = yaml_files
    tbl['DATE'] = 'YYYY/MM/DD HH:mm:SS'
    tbl['RMS point to point [ppm]'] = -1.0

    changed_keys = []

    all_keys = []
    for i in range(len(tbl)):
        params = soss_io.load_yaml_params(tbl['yaml files'][i], do_time_link=False, silent=True)
        for key in params.keys():
            all_keys = np.append(all_keys, key)
    all_keys = np.unique(all_keys)

    for i in range(len(tbl)):
        params = soss_io.load_yaml_params(tbl['yaml files'][i], do_time_link=False, silent=True)
        if i == 0:
            ref_params = soss_io.load_yaml_params(tbl['yaml files'][i], do_time_link=False, silent=True)
            for key in all_keys:
                if key not in ref_params.keys():
                    ref_params[key] = ''

        for key in np.unique(all_keys):
            if '_PATH' in key:
                continue

            if "file_temporary_in_vs_out" in key:
                continue

            if key.startswith('tag'):
                continue

            if key == 'files':
                continue

            if key not in params.keys():
                params[key] = ''

            if not np.min(params[key] == ref_params[key]):
                changed_keys = np.append(changed_keys, key)

    changed_keys = np.unique(changed_keys)
    for key in changed_keys:
        tmp = params[key]
        if type(tmp) == list:
            tmp = ' - '.join(np.array(tmp, dtype='U99'))

        tbl[key] = tmp
        if type(tbl[key][0]) == np.str_:
            tbl[key] = np.array(tbl[key], dtype='U99')

    for i in range(len(tbl)):
        params = soss_io.load_yaml_params(tbl['yaml files'][i], silent=True, do_time_link=False)
        dt = datetime.fromtimestamp(os.path.getmtime(tbl['yaml files'][i])).strftime("%Y/%m/%d %H:%M:%S")
        tbl['DATE'][i] = dt
        for key in changed_keys:
            if key in params.keys():
                tmp = params[key]
            else:
                tmp = ''
            if type(tmp) == list:
                tmp = ' - '.join(np.array(tmp, dtype='U99'))

            tbl[key][i] = tmp

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.basename()
        tbl['yaml files'][i] = tbl['yaml files'][i].split('/')[-1]

        tmp = np.array(glob.glob(params['CSV_PATH'] + '/*.csv'))
        if len(tmp) != 0:
            file_phot = tmp[0]
            misc.printc('File with photometry info: {}'.format(file_phot), 'info')
            tbl_phot = Table.read(file_phot)
            amp = tbl_phot['amplitude']
            rms = math.estimate_sigma(amp - np.roll(amp, 1)) / np.sqrt(2) * 1e6
            tbl['RMS point to point [ppm]'][i] = '{:.3f}'.format(rms)

    return tbl


def spectral_extraction(param_file_or_params, force=False):
    if type(param_file_or_params) == str:
        # load the params from the param files
        params = soss_io.load_yaml_params(param_file_or_params, force=force, do_time_link=False)
    if type(param_file_or_params) == dict:
        params = param_file_or_params

    print(misc.art('Spectral timeseries ' + params['object'], 'blue', 'CYAN'))

    if type(param_file_or_params) == str:
        # load the params from the param files
        params = soss_io.load_yaml_params(param_file_or_params, force=force, do_time_link=False)
    if type(param_file_or_params) == dict:
        params = param_file_or_params

    if force:
        misc.printc('We have force = True as an input, we re-create temporary files if they exist.', 'bad3')
        params['allow_temporary'] = False

    # get the tag for unique ID
    params = soss_io.mk_tag(params)

    for trace_order in params['trace_orders']:
        # Used for scaling residuals

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        tbl = Table.read('{}/soss_stability{}.csv'.format(params['CSV_PATH'], params['tag']))

        # median trace after normalization

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        med = fits.getdata('{}/median{}.fits'.format(params['TEMP_PATH'], params['tag']))

        # residual cube

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        residual = fits.getdata('{}/residual{}.fits'.format(params['TEMP_PATH'], params['tag']))
        params['DATA_X_SIZE'] = residual.shape[2]
        params['DATA_Y_SIZE'] = residual.shape[1]
        params['DATA_Z_SIZE'] = residual.shape[0]

        # needs the data size
        posmax, throughput = science.get_trace_pos(params, order=trace_order)

        wavegrid = science.get_wavegrid(params, order=trace_order)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        # get clean median trace for spectrum
        dx, dy, rotxy, ddy, med_clean = science.get_gradients(med, params)

        sp_sed = np.zeros(med.shape[1])
        for spectral_bin in range(med.shape[1]):
            # get a ribbon on the trace that extends over the input width
            y1 = posmax[spectral_bin] - params['trace_width_extraction'] // 2
            y2 = posmax[spectral_bin] + params['trace_width_extraction'] // 2
            sp_sed[spectral_bin] = np.nansum(med_clean[y1:y2, spectral_bin])

        ax.plot(wavegrid, sp_sed / throughput)
        ax.set(xlabel='Wavelength [nm]', ylabel='Flux\nthroughput-corrected', title='{}\norder {}'.format(params[
                                                                                                              'object'],
                                                                                                          trace_order))
        plt.tight_layout()
        for fig_type in params['figure_types']:
            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            plt.savefig('{}/sed_{}_ord{}.{}'.format(params['PLOT_PATH'], params['object'], trace_order, fig_type))
        if params['show_plots']:
            plt.show()
        plt.close()

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        outname = '{}/sed_{}_ord{}.csv'.format(params['CSV_PATH'], params['object'], trace_order)
        tbl_sed = Table()
        tbl_sed['wavelength'] = wavegrid
        tbl_sed['flux'] = sp_sed / throughput
        tbl_sed['raw flux'] = sp_sed
        tbl_sed['throughput'] = throughput
        tbl_sed.write(outname, overwrite=True)

        # error cube

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        err = fits.getdata('{}/errormap{}.fits'.format(params['TEMP_PATH'], params['tag']))

        # model trace to be compared

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        model = fits.getdata('{}/recon{}.fits'.format(params['TEMP_PATH'], params['tag']))

        if params['mask_order_0']:
            misc.printc('masking order 0', 'info')
            mask_order0, x, y = science.get_mask_order0(params)
            for nth_obs in tqdm(range(residual.shape[0]), leave=False):
                tmp = np.array(model[nth_obs])
                tmp[mask_order0] = np.nan
                model[nth_obs] = tmp

        # placeholder for the N*2048 spectra
        sp = np.zeros([residual.shape[0], residual.shape[2]]) + np.nan

        # placeholder for the N*2048 corresponding errors
        sp_err = np.zeros([residual.shape[0], residual.shape[2]]) + np.nan

        # loop through observations and spectral bins
        for nth_obs in tqdm(range(residual.shape[0]), leave=False):
            for spectral_bin in range(med.shape[1]):
                # get a ribbon on the trace that extends over the input width
                y1 = posmax[spectral_bin] - params['trace_width_extraction'] // 2
                y2 = posmax[spectral_bin] + params['trace_width_extraction'] // 2

                # model of the trace for that observation
                v0 = model[nth_obs, y1:y2, spectral_bin]
                # residual of the trace
                v1 = residual[nth_obs, y1:y2, spectral_bin]
                # corresponding error
                v2 = err[nth_obs, y1:y2, spectral_bin]

                """
                we find the ratio of residual to trace. The logic here is that
                we want to know the mean ratio of the residual to the model trace
                to do this, we divide the residual by the trace and take the weighted
                mean. The thing is that the trace has a very structured shape, so this
                needs to include a propagation of errors. There error of pixels along
                the trace profile are the input errors divided by the model trace.
                Pixels with a very low trace value have correspondingly larger errors.
                """
                with warnings.catch_warnings(record=True) as _:
                    try:
                        ratio = math.odd_ratio_mean(v1 / v0, v2 / v0)
                    except:
                        ratio = np.nan, 0
                # the above code does the equivalent of a sigma-clipped mean and
                # retuns an uncertainty
                if ratio[1] != 0:
                    # only return if a valid error has been found
                    sp[nth_obs, spectral_bin] = ratio[0]
                    sp_err[nth_obs, spectral_bin] = ratio[1]

        # binary indexes of when the transit happens
        in_transit_reject = np.zeros(sp.shape[0], dtype=bool)
        in_transit_reject[params['it'][0]:params['it'][3]] = True

        if 'reject_domain' in params:
            for i_reject in range(len(params['reject_domain']) // 2):
                cut1 = params['reject_domain'][i_reject * 2]
                cut2 = params['reject_domain'][i_reject * 2 + 1]
                in_transit_reject[cut1:cut2] = True

        in_transit_integrate = np.zeros(sp.shape[0], dtype=bool)
        in_transit_integrate[params['it'][1]:params['it'][2]] = True

        if params['remove_trend']:
            for i in range(sp.shape[1]):
                v1 = sp[~in_transit_reject, i]
                g = np.isfinite(v1)
                if np.sum(g) < 2:
                    continue
                fit = np.polyfit(np.arange(sp.shape[0])[~in_transit_reject][g], v1[g],
                                 params['transit_baseline_polyord'])
                sp[:, i] -= np.polyval(fit, np.arange(sp.shape[0]))

        if params['saveresults']:
            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            outname = '{}/residual_no_grey_ord{}{}.fits'.format(params['FITS_PATH'], trace_order, params['tag2'])
            misc.printc('We write {}'.format(outname), 'info')
            fits.writeto(outname, sp, overwrite=True)
            sp2 = sp + np.reshape(np.repeat(np.array(tbl['amplitude']), sp.shape[1]), sp.shape)

            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            outname = '{}/residual_grey_ord{}{}.fits'.format(params['FITS_PATH'], trace_order, params['tag2'])
            misc.printc('We write {}'.format(outname), 'info')
            fits.writeto(outname, sp2, overwrite=True)

            # Save the extracted spectra in format that is readable by jwst/github/jwst-mtl/SOSS/example/sossisson_to_dms.py
            wavegrid_2d = np.tile(wavegrid, (sp.shape[0], 1))  # shape = (nints, npix)
            hdu_wave = fits.ImageHDU(wavegrid_2d, name="WAVELENGTH")
            hdu_extsp = fits.ImageHDU(sp2, name="RELFLUX")
            hdu_extsperr = fits.ImageHDU(sp_err, name="RELFLUX_ERROR")
            hdul = fits.HDUList(hdus=[fits.PrimaryHDU(), hdu_wave, hdu_extsp, hdu_extsperr])

            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            hdul.writeto("{}/spectra_ord{}{}.fits".format(params['FITS_PATH'], trace_order, params['tag2']),
                         overwrite=True)

        # do the same on the photometric time series
        v1 = np.array(tbl['amplitude'][~in_transit_reject])
        fit = np.polyfit(np.arange(sp.shape[0])[~in_transit_reject], v1, 1)
        tbl['amplitude'] /= np.polyval(fit, np.arange(sp.shape[0]))
        if params['tdepth'] == "compute":
            with warnings.catch_warnings(record=True) as _:
                transit_depth = np.nanmedian(tbl["amplitude"][~in_transit_reject]) \
                                - np.mean(tbl['amplitude'][in_transit_integrate])
        else:
            transit_depth = params['tdepth']

        # weights of each point from uncertainties
        weight = 1 / sp_err ** 2

        # in-transit spectrum and error.
        with warnings.catch_warnings(record=True) as _:
            sp_in = np.nansum(sp[in_transit_integrate] * weight[in_transit_integrate], axis=0) / \
                    np.nansum(weight[in_transit_integrate], axis=0)
            err_in = 1 / np.sqrt(np.nansum(1 / sp_err[in_transit_integrate] ** 2, axis=0))

        with warnings.catch_warnings(record=True) as _:
            # out-of-transit spectrum and error.
            err_out = 1 / np.sqrt(np.nansum(1 / sp_err[~in_transit_reject] ** 2, axis=0))

        if params['remove_trend']:
            # if we have removed a trend, we need to add in quadrature out-of-transit
            # errors to in-transit
            err_in = np.sqrt(err_in ** 2 + err_out ** 2)

        # change 'infinite' for 'nan' for plotting and medians
        err_in[~np.isfinite(err_in)] = np.nan
        err_out[~np.isfinite(err_out)] = np.nan

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        outname = '{}/wavelength_ord{}{}.fits'.format(params['FITS_PATH'], trace_order, params['tag2'])
        misc.printc('We write {}'.format(outname), 'info')
        fits.writeto(outname, wavegrid, overwrite=True)

        if trace_order == 1:
            fmt1 = 'g.'
            fmt2 = 'ro--'
        if trace_order == 2:
            fmt1 = 'c.'
            fmt2 = 'mo--'

        ax.errorbar(wavegrid, (sp_in + transit_depth) * 1e6, yerr=err_in * 1e6, fmt=fmt1,
                    alpha=0.25, label='in-transit, order {}'.format(trace_order))

        if params["saveresults"]:
            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            np.savetxt("{}/tspec_ord{}{}.csv".format(params['CSV_PATH'], trace_order, params["tag2"]),
                       np.array([wavegrid, (sp_in + transit_depth) * 1e6, err_in * 1e6]).T)

        if np.nanmin(wavegrid) != 0:
            with warnings.catch_warnings(record=True) as _:
                wbin = np.floor(np.log(wavegrid / np.nanmin(wavegrid)) * params['resolution_bin'])

            wavebin = []
            flux_bin = []
            err_bin = []
            for ibin in np.unique(wbin):
                g = wbin == ibin
                moy, yerr = math.odd_ratio_mean(sp_in[g],
                                                err_in[g])
                wavebin = np.append(wavebin, np.mean(wavegrid[g]))
                flux_bin = np.append(flux_bin, moy)
                err_bin = np.append(err_bin, yerr)

            ax.errorbar(wavebin, (flux_bin + transit_depth) * 1e6, yerr=err_bin * 1e6, fmt=fmt2,
                        label='Resolution {}, order {}'.format(params['resolution_bin'], trace_order))

    # we are done with loop on trace_order, we can convert all (if more than 1) orders
    # into the eureka outputs
    soss_io.sossisson_to_eureka(params)

    ax.set(ylabel='ppm', xlabel='wavelength [$\mu$m]')
    ax.set(ylim=params['spectrum_ylim_ppm'])
    ax.legend()
    plt.savefig('{}/sossisson_output{}.pdf'.format(params['PLOT_PATH'], params['tag2']))
    if params['show_plots']:
        plt.show()
    plt.close()
