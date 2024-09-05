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

# TODO: remove
from sossisse.general import plots, science, soss_io

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
    l_table, lrecon = inst.apply_amp_recon(cube, err, med, mask_trace_pos,
                                           lvector, x_trace_pos, y_trace_pos,
                                           x_order0, y_order0)
    # -------------------------------------------------------------------------
    # normalize the trace but a normalization factor
    l_table = inst.normalize_sum_trace(l_table)

    # -------------------------------------------------------------------------
    # per pixel baseline
    if inst.params['PER_PIXEL_BASELINE_CORRECTION']:
        misc.printc('Performing per-pixel baseline subtraction', 'info')
        cube = inst.per_pixel_baseline(cube, mask)

    # -------------------------------------------------------------------------
    # write files
    # -------------------------------------------------------------------------
    # get the meta data
    meta_data = inst.get_variable('META', func_name)
    # -------------------------------------------------------------------------
    # write the error map
    errfile = os.path.join(inst.params['TEMP_PATH'],
                           'errormap{}.fits'.format(inst.params['tag']))
    io.save_fitsimage(errfile, err, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the residual map
    resfile = os.path.join(inst.params['TEMP_PATH'],
                           'residual{}.fits'.format(inst.params['tag']))
    io.save_fitsimage(resfile, cube, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the recon
    reconfile = os.path.join(inst.params['TEMP_PATH'],
                                'recon{}.fits'.format(inst.params['tag']))
    io.save_fitsimage(reconfile, lrecon, meta=meta_data)
    # -------------------------------------------------------------------------
    # write the table to the csv path
    ltbl_file = os.path.join(inst.params['CSV_PATH'],
                             'stability{}.csv'.format(inst.params['tag']))
    io.save_table(ltbl_file, l_table, fmt='csv')
    # -------------------------------------------------------------------------
    # print the rms baseline for all methods
    for method in inst.get_rms_baseline():
        # calculate the rms for this method
        rms_method = inst.get_rms_baseline(l_table['amplitude'], method=method)
        # print this
        msg = '{0}, rms = {1:.1f}ppm'.format(method, rms_method * 1e6)
        misc.printc(msg, 'number')
    # -------------------------------------------------------------------------
    # plot the stability plot
    plots.plot_stability(inst.params, l_table)
    # -------------------------------------------------------------------------


    # TODO: -------------------------------------------------------------------
    # TODO: Got to here
    # TODO: -------------------------------------------------------------------
    # plot the transit plot
    plots.plot_transit(inst.params, l_table)

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
