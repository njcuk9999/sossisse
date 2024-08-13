import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.ndimage import binary_dilation
from scipy.ndimage import shift
from scipy.signal import medfilt2d
from skimage import measure
from tqdm import tqdm
from wpca import EMPCA

from sossisse.core import math, misc


# sosssisse stuff


def per_pixel_baseline(cube, mask, params):
    cube = np.array(cube, dtype=float)
    index = np.arange(params['DATA_Z_SIZE'], dtype=float)
    oot = params['oot_domain']
    poly_order = params['transit_baseline_polyord']

    mid_transit = int(np.nanmean(params['it']))
    rms0 = math.estimate_sigma(cube[mid_transit] * mask)

    for ix in tqdm(range(params['DATA_X_SIZE']), leave=False):
        cube_slice = np.array(cube[:, :, ix])

        if True not in np.isfinite(cube_slice):
            continue

        for iy in range(params['DATA_Y_SIZE']):
            if np.isfinite(mask[iy, ix]):
                sample = cube_slice[:, iy]
                sample2 = sample[oot]
                index2 = index[oot]
                if np.sum(np.isfinite(sample2)) > poly_order:
                    if False in np.isfinite(sample2):
                        g = np.isfinite(sample2)
                        sample2 = sample2[g]
                        index2 = index2[g]
                    fit = math.robust_polyfit(index2, sample2, poly_order, 5)[0]
                    cube_slice[:, iy] -= np.polyval(fit, index)

        cube[:, :, ix] = np.array(cube_slice)

    rms1 = math.estimate_sigma(cube[mid_transit] * mask)
    misc.printc('-- For sample mid-transit frame -- ', '')
    misc.printc('\t rms[before] : {:.3f}'.format(rms0), 'number')
    misc.printc('\t rms[after] : {:.3f}'.format(rms1), 'number')

    return cube


from scipy.signal import convolve2d


def get_mask_order0(params):
    if params['mode'] != 'SOSS':
        misc.printc('This is *not* a SOSS dataset, we do not mask order 0', 'bad3')
        misc.printc('Don''t worry, we will just skip that step and set "mask order 0" = False', 'bad2')
        params['mask_order_0'] = False
        return 1

    diff = fits.getdata(params['file_temporary_in_vs_out'])

    if params['mode'] == 'SOSS':
        posmax, throughput = get_trace_pos(params, order=1, round_pos=False)
        posmax -= np.nanmean(posmax)

        diff2 = np.array(diff)
        for i in range(diff.shape[1]):
            valid = np.isfinite(diff[:, i])
            if np.mean(valid) < 0.5:
                continue
            spl = ius(np.arange(diff.shape[0])[valid] - posmax[i], diff[:, i][valid], k=3, ext=1)
            diff2[:, i] = spl(np.arange(diff.shape[0]))

        for i in range(diff.shape[0]):
            diff2[i] = math.lowpassfilter(diff2[i])

        for i in range(diff.shape[1]):
            spl = ius(np.arange(diff2.shape[0]) + posmax[i], diff2[:, i], k=3, ext=1)
            diff2[:, i] = spl(np.arange(diff2.shape[0]))
    else:
        diff2 = np.array(diff)
        for i in range(diff.shape[0]):
            diff2[i] = math.lowpassfilter(diff2[i])

    diff -= diff2

    for i in range(params['DATA_X_SIZE']):
        diff[:, i] -= np.nanmedian(diff[:, i])

    nsig = np.array(diff)

    for i in tqdm(range(diff.shape[0]), leave=False):
        nsig[i] /= math.lowpassfilter(np.abs(diff[i]))

    mask = nsig > 1  # we look for a consistent set of >2 sigma pixels

    mask2 = np.zeros_like(mask)
    all_labels1 = measure.label(mask, connectivity=2)
    for u in tqdm(np.unique(all_labels1), leave=False):

        g = (u == all_labels1)
        if not mask[g][0]:
            continue

        if np.sum(g) > 100:
            mask2[g] = True

    rad = np.sqrt(np.nansum((np.indices([7, 7]) - 3.0) ** 2, axis=0))
    mask = binary_dilation(mask2, rad < 3.5)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].imshow(diff, vmin=np.nanpercentile(diff, 2), vmax=np.nanpercentile(diff, 80), aspect='auto', origin='lower',
                 interpolation='none')
    ax[1].imshow(mask, aspect='auto', origin='lower', interpolation='none')
    ax[0].set(title='median residual in-out')
    ax[1].set(title='mask')
    plt.tight_layout()

    for figtype in params['figure_types']:
        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        plt.savefig('{}/masking_order0_{}.pdf'.format(params['PLOT_PATH'], params['tag'], figtype))

    if params['show_plots']:
        plt.show()
    plt.close()

    y, x = np.where(binary_dilation(mask, [[0, 1, 0], [1, 1, 1], [0, 1, 0]]) != mask)

    return mask, x, y


def pixeldetrending(cube, params):
    tracemap = params['tracemap']

    tbl = Table.read(params["pixel_level_detrending_file"])
    keys = params['pixel_level_detrending_keywords']
    sample = np.ones([cube.shape[0], len(keys)])

    for ikey in range(len(keys)):
        sample[:, ikey] = tbl[keys[ikey]]
        sample[:, ikey] -= math.lowpassfilter(sample[:, ikey], 15)

    misc.printc('We perform pixel-level decorrelation', 'info')
    for iy in tqdm(range(cube.shape[1]), leave=False):
        tmp = np.array(cube[:, iy, :])

        for ix in tqdm(range(cube.shape[2]), leave=False):
            v = np.array(tmp[:, ix], dtype=float)
            with warnings.catch_warnings(record=True) as _:
                mean_finite = np.nanmean(np.isfinite(v))

            if not np.isfinite(tracemap[ix, iy]):
                continue

            if mean_finite > 0.5:
                dv = v - math.lowpassfilter(v, 15)

                v2 = np.array(v)

                amps = math.linear_minimization(dv, sample)[0]
                for ikey in range(len(keys) - 1):
                    v2 -= sample[:, ikey] * amps[ikey]

                tmp[:, ix] = v2
        cube[:, iy, :] = tmp

    return cube


def get_trace_map(params, silent=False):
    if params['trace_width_masking'] < 1:
        params['TRACEMAP'] = np.ones([params['DATA_Y_SIZE'], params['DATA_X_SIZE']], dtype=bool)
        return params

    # avoid ambiguity with existence of tracemap
    tracemap = None
    if params['mode'] == 'SOSS':
        if params['DATA_Y_SIZE'] == 256:
            if 'wlc_domain' not in params.keys():
                # that's for the SOSS mode
                tracemap1, throughput = get_trace_pos(params, map2d=True, order=1, silent=silent)
                tracemap2, throughput = get_trace_pos(params, map2d=True, order=2, silent=silent)

                tracemap = tracemap1 | tracemap2
            else:  # only get order 1 if wlc domain is defined
                tracemap, throughput = get_trace_pos(params, map2d=True, order=1, silent=silent)

        if params['DATA_Y_SIZE'] == 96:
            tracemap, throughput = get_trace_pos(params, map2d=True, order=1, silent=silent)

    if params['mode'] == 'PRISM':
        tracemap, throughput = get_trace_pos(params, map2d=True, order=1, silent=silent)

    if 'wlc_domain' in params.keys():
        if not silent:
            message = 'We cut the domain of the WLC to {0:.2f} to {1:.2f}µm'
            misc.printc(message.format(params['wlc_domain'][0], params['wlc_domain'][1]), 'number')
        wavegrid = get_wavegrid(params, order=1)
        tracemap[:, wavegrid < params['wlc_domain'][0]] = False
        tracemap[:, wavegrid > params['wlc_domain'][1]] = False
        tracemap[:, ~np.isfinite(wavegrid)] = False

    params['TRACEMAP'] = tracemap
    return params


def get_effective_wavelength(params):

    # TODO: you shouldn't use '/'  as it is OS dependent
    # TODO: use os.path.join(1, 2, 3)
    med = fits.getdata('{}/median{}.fits'.format(params['TEMP_PATH'], params['tag']))
    params = get_trace_map(params)
    wavegrid = get_wavegrid(params, order=1)

    with warnings.catch_warnings(record=True) as _:
        sp_domain = np.nanmean(params['TRACEMAP'] * med, axis=0)

    mean_photon_weighted = np.nansum(sp_domain * wavegrid) / np.nansum(sp_domain * np.isfinite(wavegrid))
    sp_energy = sp_domain / wavegrid
    mean_energy_weighted = np.nansum(sp_energy * wavegrid) / np.nansum(sp_energy * np.isfinite(wavegrid))

    if 'wlc domain' in params.keys():
        misc.printc('Domain :\t {0:.3f} -- {1:.3f} µm'.format(params['wlc_domain'][0], params['wlc_domain'][1]),
                    'number')
    else:
        misc.printc("Full domain included, parameter params['wlc_domain'] not defined", 'bad2')
    misc.printc('energy-weighted mean :\t{0:.3f} µm'.format(mean_energy_weighted), 'number')
    misc.printc('photon-weighted mean :\t{0:.3f} µm'.format(mean_photon_weighted), 'number')

    params['photon_weighted_wavelength'] = mean_photon_weighted
    params['energy_weighted_wavelength'] = mean_energy_weighted

    return params


def get_rms_baseline(v=None, method='linear_sigma'):
    if type(v) == type(None):
        return ['naive_sigma', 'linear_sigma', 'lowpass_sigma', 'quadratic_sigma']

    if method == 'naive_sigma':
        # we don't know that there's a transit
        return (np.nanpercentile(v, 84) - np.nanpercentile(v, 16)) / 2.0

    if method == 'linear_sigma':
        # difference to immediate neighbours
        return math.estimate_sigma(v[1:-1] - (v[2:] + v[0:-2]) / 2) / np.sqrt(1 + 0.5)
    if method == 'lowpass_sigma':
        return math.estimate_sigma(v - math.lowpassfilter(v, width=15))

    if method == 'quadratic_sigma':
        v2 = -np.roll(v, -2) / 3 + np.roll(v, -1) + np.roll(v, 1) / 3
        return math.estimate_sigma(v - v2) / np.sqrt(20 / 9.0)


def patch_isolated_bads(cube, params):
    # replace isolated bad pixels by valid ones to increase the fraction of pixels
    # that can be used in derivative

    if params['allow_temporary']:

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        params['file_temporary_clean_NaN'] = params['TEMP_PATH'] + '/temporary_cleaned_isolated.fits'
        # params['file_temporary_clean_cube_mask'] = params['TEMP_PATH'] + '/temporary_cleaned_isolated_err.fits'

        files_exist = os.path.isfile(params['file_temporary_clean_NaN'])  # and \
        # os.path.isfile(params['file_temporary_clean_cube_mask'])

        if params['force']:
            files_exist = False

        if files_exist:
            misc.printc('patch_isolated_bads\twe read temporary files to speed things', 'info')
            misc.printc('Reading {}'.format(params['file_temporary_clean_NaN']), 'info')
            # printc('Reading {}'.format(params['file_temporary_clean_cube_mask']),'info')

            return fits.getdata(params['file_temporary_clean_NaN'])  # , \
            # np.array(fits.getdata(params['file_temporary_clean_cube_mask']), dtype=bool)

    misc.printc('Removing isolated NaNs', 'info')
    for islice in tqdm(range(cube.shape[0]), leave=False):
        cube_slice = np.array(cube[islice, :, :])
        mask_slice = np.isfinite(cube_slice)
        ker = np.zeros([3, 3], dtype=float)
        ker[1, :] = 1.0
        ker[:, 1] = 1.0
        isolated_bad = (convolve2d(np.array(mask_slice, dtype=float), ker, mode='same') ==
                        4) * (mask_slice == 0)
        ybad, xbad = np.where(isolated_bad)
        cube_slice[ybad, xbad] = (cube_slice[ybad + 1, xbad] + cube_slice[ybad - 1, xbad] +
                                  cube_slice[ybad, xbad + 1] + cube_slice[ybad, xbad - 1]) / 4.0
        mask_slice[ybad, xbad] = True

        cube[islice, :, :] = cube_slice
        # cube_mask[islice, :, :] = mask_slice

    if params['allow_temporary']:
        misc.printc('We write intermediate files, they will be read to speed things next time', 'info')
        misc.printc(params['file_temporary_clean_NaN'], 'info')
        # printc(params['file_temporary_clean_cube_mask'],'info')

        fits.writeto(params['file_temporary_clean_NaN'], cube, overwrite=True)
        # fits.writeto(params['file_temporary_clean_cube_mask'], np.array(cube_mask, dtype=int), overwrite=True)

    return cube  # , cube_mask


def remove_background(cube, params):
    """
    Removes the background with a 3 DOF model. It's the background image
    times a slope + a DC offset (1 amp, 1 slope, 1 DC)

    Inputs : 
    cube = the transit data
    err = the associated uncertainties, used to flag illuminated pixels
    mask = True where we are *ON THE TRACE* (to be removed)
    nite = number of iterations of sigma clipping

    """
    cube = np.array(cube, dtype=float)

    if params['bkgfile'] != '':
        misc.printc('applying the chorisoss background model correction', 'info')

        bgnd = np.array(params['bgnd'], dtype=float)
        with warnings.catch_warnings(record=True) as _:
            moy = np.nanmean(cube, axis=0)

        box = params['soss_background_glitch_box']

        bgnd_shifts = np.arange(-5.0, 5.0, .2)
        rms = np.zeros_like(bgnd_shifts)
        misc.printc('Tweaking the position of the background', 'info')
        for ishift in tqdm(range(len(bgnd_shifts)), leave=False):
            bgnd_shift = bgnd_shifts[ishift]
            bgnd2 = shift(bgnd, (0, bgnd_shift))
            xvalues = bgnd2[box[2]:box[3], box[0]:box[1]].ravel()
            yvalues = moy[box[2]:box[3], box[0]:box[1]].ravel()
            fit = \
                math.robust_polyfit(xvalues, yvalues, 1, 5)[0]

            diff = moy - np.polyval(fit, bgnd2)
            med = np.nanmedian(diff[box[2]:box[3], box[0]:box[1]], axis=0)
            rms[ishift] = np.std(med)
        imin = np.argmin(rms)
        fit = np.polyfit(bgnd_shifts[imin - 1:imin + 2], rms[imin - 1:imin + 2], 2)
        optimal_offset = -.5 * fit[1] / fit[0]
        misc.printc('Optimal backgorund offset {:.3f} pix'.format(optimal_offset), 'number')

        bgnd2 = shift(bgnd, (0, optimal_offset))
        fit = \
            math.robust_polyfit(bgnd2[box[2]:box[3], box[0]:box[1]].ravel(), moy[box[2]:box[3], box[0]:box[1]].ravel(),
                                1,
                                5)[0]
        bgnd = np.polyval(fit, bgnd2)
        for i in tqdm(range(cube.shape[0]), leave=False):
            cube[i] -= bgnd

        with warnings.catch_warnings(record=True) as _:
            moy = np.nanmean(cube, axis=0)

        mask = np.zeros_like(moy, dtype=float)
        for i in tqdm(range(2048), leave=False):
            with warnings.catch_warnings(record=True) as _:
                mask[:, i] = moy[:, i] < np.nanpercentile(moy[:, i], 50)
        mask[mask != 1.0] = np.nan

        # ref_bgnd_level = np.nanmedian(mask * bgnd)
        # moy_bgnd_level = np.nanmedian(mask * moy)
        # ratio = moy_bgnd_level / ref_bgnd_level

        # with warnings.catch_warnings(record=True) as _:
        #    lowp = math.lowpassfilter(np.nanmedian(mask * bgnd, axis=0), 25)
        # bgnd -= np.tile(lowp, 256).reshape(moy.shape)

        # printc('Background ratio : {:.3f}'.format(ratio), 'number')
        # for i in tqdm(range(cube.shape[0]), leave=False):
        #    cube[i] -= (ratio * bgnd)

        for i in tqdm(range(cube.shape[0]), leave=False):
            with warnings.catch_warnings(record=True) as _:
                lowp = math.lowpassfilter(np.nanmedian(mask * cube[i], axis=0), 25)
            cube[i] -= np.tile(lowp, 256).reshape(moy.shape)

    else:
        misc.printc('we do not clean background, do_bkg == False', 'bad1')

    return cube


def get_gradients(med, params, doplot=False):
    misc.printc('We find the gradients', 'info')

    med2 = np.array(med)

    for _ in tqdm(range(4), leave=False):
        med_filter = medfilt2d(med2, kernel_size=[1, 5])
        bad = ~np.isfinite(med2)
        med2[bad] = med_filter[bad]

    # find gradient along the x and y dimension
    dy, dx = np.gradient(med2)
    ddy = np.gradient(dy, axis=0)

    # find the rotation pattern as per the coupling between the two axes
    y, x = np.indices(med2.shape, dtype=float)

    # we assume a pivot relative to the center of the array
    x -= med2.shape[1] / 2.0
    y -= med2.shape[0] / 2.0

    # infinitesinal motion in rotation scaled to 1 radian
    rotxy = x * dy - y * dx

    if doplot:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all')
        rms = np.nanpercentile(dx, [5, 95])
        rms = rms[1] - rms[0]
        ax[0].imshow(dx, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
        rms = np.nanpercentile(dy, [5, 95])
        rms = rms[1] - rms[0]
        ax[1].imshow(dy, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
        rms = np.nanpercentile(rotxy, [5, 95])
        rms = rms[1] - rms[0]
        ax[2].imshow(rotxy, aspect='auto', vmin=-2 * rms, vmax=2 * rms)
        for figtype in params['figure_types']:
            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            plt.savefig('{}/derivatives{}.{}'.format(params['PLOT_PATH'], params['tag'], figtype))

    dx = np.array(dx, dtype=float)
    dy = np.array(dy, dtype=float)
    rotxy = np.array(rotxy, dtype=float)
    ddy = np.array(ddy, dtype=float)
    return dx, dy, rotxy, ddy, med2


def get_valid_oot(params):
    if 'oot_domain' in params.keys():
        return params

    valid_oot = np.ones(params['DATA_Z_SIZE'], dtype=bool)
    valid_oot[params['it'][0]:params['it'][3]] = False
    if 'reject domain' in params.keys():
        for i_reject in range(len(params['reject_domain']) // 2):
            valid_oot[params['reject_domain'][2 * i_reject]:params['reject_domain'][2 * i_reject + 1]] = False
    params['oot_domain'] = valid_oot

    # find before/after bits of the domain
    params['oot_domain_before'] = valid_oot * (np.arange(len(valid_oot)) < params['it'][0])
    params['oot_domain_after'] = valid_oot * (np.arange(len(valid_oot)) > params['it'][3])

    return params


def clean_1f(cube, err, params):
    # we must update the param file prior to checking if 1/f-cleaned files exist

    if params['allow_temporary']:

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        params['median_image_file'] = '{}/median{}.fits'.format(params['TEMP_PATH'], params['tag'])
        params['clean_cube_file'] = '{}/cube{}.fits'.format(params['TEMP_PATH'], params['tag'])
        params['file_temporary_before_after_clean1f'] = params['TEMP_PATH'] + '/temporary_before_after_clean1f.fits'
        params['file_temporary_pcas'] = params['TEMP_PATH'] + '/temporary_pcas.fits'

        files_exist = os.path.isfile(params['median_image_file']) and \
                      os.path.isfile(params['clean_cube_file']) and \
                      os.path.isfile(params['file_temporary_before_after_clean1f']) and \
                      os.path.isfile(params['file_temporary_in_vs_out'])

        if params['fit_pca']:
            files_exist *= os.path.isfile(params['file_temporary_pcas'])

        if params['force']:
            files_exist = False

        if files_exist:
            misc.printc('clean_1f\tWe read temporary files to speed things', 'info')

            if params['fit_pca']:
                params['PCA components'] = fits.getdata(params['file_temporary_pcas'])

            # cube, med, med_diff, diff_in_out, params
            return fits.getdata(params['clean_cube_file']), \
                fits.getdata(params['median_image_file']), \
                fits.getdata(params['file_temporary_before_after_clean1f']), \
                fits.getdata(params['file_temporary_in_vs_out']), params

    ############################################################################
    # Part of the code that does the 1/f filtering
    ############################################################################
    # create a copy of the cube, we will normalize the amplitude of each trace
    cube2 = np.array(cube, dtype=float)
    # first estimate of the trace amplitude
    misc.printc('first median of cube to create trace estimate', 'info')

    params = get_valid_oot(params)
    with warnings.catch_warnings(record=True) as _:
        if params['ootmed']:
            med = np.nanmedian(cube2[params['oot_domain']], axis=0)
        else:
            med = np.nanmedian(cube2, axis=0)

    # do a dot product of each trace to the median and adjust amplitude so that
    # they all match
    amps = np.zeros(cube2.shape[0])
    for i in tqdm(range(cube2.shape[0]), leave=False):
        mask = np.isfinite(cube[i]) * np.isfinite(med)
        amps[i] = np.nansum(cube2[i][mask] * med[mask]) / np.nansum(med[mask] ** 2)
        cube2[i] /= amps[i]

    # median of the normalized cube
    misc.printc('2nd median of cube with proper normalization', 'info')
    with warnings.catch_warnings(record=True) as _:
        if params['ootmed']:
            med = np.nanmedian(cube2[params['oot_domain']], axis=0)

            before = np.nanmedian(cube2[params['oot_domain_before']], axis=0)
            after = np.nanmedian(cube2[params['oot_domain_after']], axis=0)

            med_diff = (before - after)

            for i in range(med_diff.shape[0]):
                med_diff[i] = math.lowpassfilter(med_diff[i], 15)

            ratio = np.sqrt(np.nansum(med ** 2) / np.nansum(med_diff ** 2))
            med_diff = med_diff * ratio

        else:
            med = np.nanmedian(cube2, axis=0)

        # we also keep track of the in vs out-of-transit 2D image. This is useful to find order-0
        # psfs
        med_out = np.nanmedian(cube2[params['oot_domain']], axis=0)
        med_in = np.nanmedian(cube2[params['it'][0]:params['it'][3]], axis=0)
        diff_in_out = med_in - med_out

    residuals = np.zeros_like(cube)
    for i in tqdm(range(params['DATA_Z_SIZE']), leave=False):
        mask = np.isfinite(cube[i]) * np.isfinite(med) * params['TRACEMAP']
        amps[i] = np.nansum(cube[i][mask] * med[mask]) / np.nansum(med[mask] ** 2)

        # we get the appropriate slice of the error cube
        tmp = err[i]
        err[i] = tmp

        tmp = cube[i] - med * amps[i]

        residuals[i] = tmp

    if params['degree_1f_corr'] == 0:
        noise_1f = np.nanmedian(residuals, axis=1)
        for i in tqdm(range(params['DATA_Z_SIZE']), leave=False):
            for col in range(params['DATA_X_SIZE']):
                cube[i, :, col] -= noise_1f[i, col]
    else:

        tracemap = np.array(params['TRACEMAP'])
        for i in tqdm(range(cube2.shape[0]), leave=False):
            residual = residuals[i]
            err2 = err[i]
            for col in range(err2.shape[1]):
                # subtract only 0th term of the odd_ratio_mean, the next one is the
                # uncertainty in the mean
                if np.nansum(np.isfinite(residual[:, col])) < (params['degree_1f_corr'] + 3):
                    continue

                try:
                    v1, err1, index = residual[:, col], err2[:, col], np.arange(residual.shape[0], dtype=float)
                    g = np.isfinite(v1 + err1) * ~tracemap[:, col] * (np.abs(v1 / err1) < 5)
                    v1, err1, index = v1[g], err1[g], index[g]

                    fit = np.polyfit(index, v1, params['degree_1f_corr'], w=1 / err1)

                    cube[i, :, col] -= np.polyval(fit, np.arange(residual.shape[0]))

                except:
                    # if the fit fails, we just set the column to NaN
                    cube[i, :, col] = np.nan

    if params['fit_pca']:
        nanmask = np.ones_like(params['TRACEMAP'], dtype=float)
        nanmask[~params['TRACEMAP']] = np.nan

        cube3ini = cube2[params['oot_domain']]
        for iframe in tqdm(range(cube3ini.shape[0]), leave=False):
            cube3ini[iframe] -= med

        cube3 = np.array(cube3ini)

        err3 = err[params['oot_domain']]

        for i in tqdm(range(len(cube3)), leave=False):
            cube3[i] *= nanmask

        cube3 = cube3.reshape([cube3.shape[0], params['DATA_Y_SIZE'] * params['DATA_X_SIZE']])
        err3 = err3.reshape([err3.shape[0], params['DATA_Y_SIZE'] * params['DATA_X_SIZE']])
        bad = ~np.isfinite(cube3 + err3)
        weights = 1 / err3
        weights[bad] = 0.0
        cube3[bad] = 0.0

        with warnings.catch_warnings(record=True) as _:
            valid = np.where(np.nanmean(weights != 0, axis=0) > 0.95)[0]
        misc.printc('Computing principal components', 'info')
        with warnings.catch_warnings(record=True) as _:
            pca = EMPCA(n_components=params['n_pca']).fit(cube3[:, valid],
                                                          weights=weights[:, valid])
            fit_pca = pca.fit(cube3[:, valid], weights=weights[:, valid])
            variance_ratio = np.array(pca.explained_variance_ratio_)
        variance_ratio /= variance_ratio[0]

        amps = np.zeros([params['n_pca'], cube3.shape[0]])
        for iframe in tqdm(range(cube3.shape[0]), leave=False):
            for ipca in range(params['n_pca']):
                amps[ipca, iframe] = np.nansum(fit_pca.components_[ipca] * cube3[iframe, valid]
                                               / err3[iframe, valid] ** 2) / np.nansum(1 / err3[iframe, valid] ** 2)

        pcas = np.zeros([params['n_pca'], params['DATA_Y_SIZE'], params['DATA_X_SIZE']], dtype=float)

        cube3 = np.array(cube3ini)
        mask = ~np.isfinite(cube3ini)
        cube3[mask] = 0.0
        # weights = np.zeros_like(pcas)

        fig, ax = plt.subplots(nrows=params['n_pca'], ncols=1, sharex='all',
                               sharey='all', figsize=[8, 4 * params['n_pca']])
        for ipca in tqdm(range(params['n_pca']), leave=False):

            for iframe in range(cube3.shape[0]):
                pcas[ipca] += (amps[ipca, iframe] * cube3[iframe])

            # trick to avoid getting slices of a 3d cube
            tmp = np.array(pcas[ipca, :, :])
            for icol in range(pcas.shape[2]):
                tmp[:, icol] -= np.nanmedian(tmp[:, icol])

            # for irow in range(pcas.shape[1]):
            #    tmp[irow,:] = math.lowpassfilter(tmp[irow,:],15)
            tmp /= np.sqrt(np.nansum(tmp ** 2 * nanmask) / np.nansum(med ** 2 * nanmask))
            pcas[ipca, :, :] = tmp

            if params['n_pca'] != 1:
                ax_tmp = ax[ipca]
            else:
                ax_tmp = ax

            ax_tmp.imshow(pcas[ipca], aspect='auto', vmin=np.nanpercentile(pcas[ipca], 0.5),
                          vmax=np.nanpercentile(pcas[ipca], 99.5), origin='lower')
            ax_tmp.set(title='PCA {}, variance {:.4f}'.format(ipca + 1, variance_ratio[ipca]))
        plt.tight_layout()

        for fig_type in params['figure_types']:
            # TODO: you shouldn't use '/'  as it is OS dependent
            # TODO: use os.path.join(1, 2, 3)
            outname = params['PLOT_PATH'] + '/pcas.' + fig_type
            plt.savefig(outname)
        if params['whoami'] in params['user_show_plot']:
            plt.show()
        plt.close()

        params['PCA_components'] = pcas
        params['PCA_amps'] = amps
        params['PCA_variance_decay'] = variance_ratio

        fits.writeto(params['file_temporary_pcas'], pcas)

    # always write the in-vs-out file, it is required for masking order 0 in SOSS
    fits.writeto(params['file_temporary_in_vs_out'], diff_in_out, overwrite=True)

    if params['allow_temporary']:
        misc.printc('We write temporary files to speed things next time', 'info')
        misc.printc(params['median_image_file'], 'info')
        misc.printc(params['clean_cube_file'], 'info')

        fits.writeto(params['clean_cube_file'], cube, overwrite=True)
        fits.writeto(params['median_image_file'], med, overwrite=True)
        fits.writeto(params['file_temporary_before_after_clean1f'], med_diff, overwrite=True)

    return cube, med, med_diff, diff_in_out, params


def get_trace_pos(params, map2d=False, order=1, round_pos=True, silent=False):
    if not os.path.isfile(params['pos_file']) and params['mode'] == 'PRISM':

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        hf = h5py.File(params['CALIBPATH'] + '/' + params['wave_file_prism'], 'r')
        xpix = hf['x']
        wave = hf['wave_1d']

        if not silent:
            misc.printc('This is an PRISM dataset without a pos file', 'bad1')
        med = np.nanmedian(fits.getdata(params['file_temporary_clean_NaN']), axis=0)
        iy, ix = np.indices(med.shape)
        s1 = np.nansum(iy * med, axis=0)
        s2 = np.nansum(med, axis=0)
        index = np.arange(med.shape[1])
        is_flux = s2 > (np.nanpercentile(s2, 95) / 5)
        fit, mask = math.robust_polyfit(index[is_flux], (s1 / s2)[is_flux], 2, 4)

        tbl = Table()
        tbl['X'] = index
        tbl['Y'] = np.polyval(fit + params['y_trace_offset'], index + params['x_trace_offset'])
        # set to NaN if outside of domain
        tbl['WAVELENGTH'] = np.nan
        # put values from table if inside
        tbl['WAVELENGTH'][np.array(xpix, dtype=int)] = wave

        misc.printc('We write {}'.format(tbl.write(params['pos_file'])), 'info')
        tbl.write(params['pos_file'], overwrite=True)

    # reference TRACE table
    tbl_ref = Table.read(params['pos_file'], order)
    # if we don't have a throughput, we assume == 1
    if 'THROUGHPUT' not in tbl_ref.keys():
        if not silent:
            misc.printc('\tThe trace table does not have a "THROUGHPUT" column', 'bad3')
            misc.printc('\tWe set THROUGHPUT = 1', 'bad3')
        tbl_ref['THROUGHPUT'] = 1.0

    valid = (tbl_ref['X'] > 0) & (tbl_ref['X'] < params['DATA_X_SIZE'] - 1)
    tbl_ref = tbl_ref[valid]
    tbl_ref = tbl_ref[np.argsort(tbl_ref['X'])]

    spl_y = ius(tbl_ref['X'] + params['x_trace_offset'], tbl_ref['Y'] + params['y_trace_offset'], ext=0, k=1)
    spl_throughput = ius(tbl_ref['X'] + params['x_trace_offset'], tbl_ref['THROUGHPUT'], ext=0, k=1)

    if round_pos:
        dtype = int
    else:
        dtype = float
    posmax = np.array(spl_y(np.arange(params['DATA_X_SIZE'])) - .5, dtype=dtype)  # round to integer
    throughput = np.array(spl_throughput(np.arange(params['DATA_X_SIZE'])), dtype=float)

    if map2d:
        posmap = np.zeros([params['DATA_Y_SIZE'], params['DATA_X_SIZE']], dtype=bool)
        for i in range(params['DATA_X_SIZE']):
            bottom = posmax[i] - params['trace_width_masking'] // 2
            top = posmax[i] + params['trace_width_masking'] // 2
            if bottom < 0:
                bottom = 0
            if top > (params['DATA_Y_SIZE'] - 1):
                top = (params['DATA_Y_SIZE'] - 1)
            posmap[bottom:top, i] = True

        return posmap, throughput
    else:
        return posmax, throughput


def bin_cube(cube, params, bin_type='Flux'):
    """

    :param cube:
    :param params:
    :param bin_type: "Flux" (dumb sum), "Error" (sum of variance) or "DQ"
    :return:
    """
    if not params['time_bin']:
        return cube
    if params['n_time_bin'] == 1:
        return cube

    if bin_type not in ["Flux", "Error", "DQ"]:
        err_string = 'bin_type in function "bin_cube" must be "Flux", "Error" or "DQ" \n type {}'.format(bin_type)
        raise ValueError(err_string)

    dims_bin = np.array(cube.shape, dtype=int)
    dims_bin[0] = dims_bin[0] // params['n_time_bin']
    cube2 = np.zeros(dims_bin)
    misc.printc('We bin data by a factor {}'.format(params['n_time_bin']), 'number')
    for i in tqdm(range(dims_bin[0]), leave=False):
        if bin_type == "Flux":
            cube2[i] = np.nansum(cube[i * params['n_time_bin']:(i + 1) * params['n_time_bin']], axis=0)
        if bin_type == "Error":
            cube2[i] = np.sqrt(np.nansum(cube[i * params['n_time_bin']:(i + 1) * params['n_time_bin']] ** 2, axis=0))
        if bin_type == "DQ":
            cube2[i] = np.nanmax(cube[i * params['n_time_bin']:(i + 1) * params['n_time_bin']], axis=0)

    return cube2


def load_data_with_dq(params):
    if params['allow_temporary']:

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        params['file_temporary_initial_cube'] = params['TEMP_PATH'] + '/temporary_initial_cube.fits'
        params['file_temporary_initial_err'] = params['TEMP_PATH'] + '/temporary_initial_err.fits'
        # params['file_temporary_initial_dq'] = params['TEMP_PATH'] + '/temporary_initial_dq.fits'
        # params['file_temporary_initial_cube_mask'] = params['TEMP_PATH'] + '/temporary_initial_cube_mask.fits'

        files_exist = os.path.isfile(params['file_temporary_initial_cube']) and \
                      os.path.isfile(params['file_temporary_initial_err'])  # and \
        # os.path.isfile(params['file_temporary_initial_dq']) and \
        # os.path.isfile(params['file_temporary_initial_cube_mask'])

        if params['force']:
            files_exist = False

        if files_exist:
            misc.printc('load_data\twe read temporary files to speed things', 'info')
            cube = fits.getdata(params['file_temporary_initial_cube'])
            # for future reference in the code, we keep track of data size
            params['DATA_X_SIZE'] = cube.shape[2]
            params['DATA_Y_SIZE'] = cube.shape[1]
            params['DATA_Z_SIZE'] = cube.shape[0]

            return cube, fits.getdata(params['file_temporary_initial_err'])  # , \
            # fits.getdata(params['file_temporary_initial_dq']), \
            # fits.getdata(params['file_temporary_initial_cube_mask'])

    n = 0  # number of slices in the final cube
    # handling the files with their varying sizes. We read and count slices
    for ifile in range(len(params['files'])):
        c = bin_cube(fits.getdata(params['files'][ifile]), params, "Flux")

        n += c.shape[0]

    if params['flatfile'] != '':
        flat = fits.getdata(params['flatfile'])

        if flat.shape == (2048, 2048) and params['mode'] == 'SOSS':
            flat = flat[-256:]
        # some sanity checks in flat
        flat[flat == 0] = np.nan
        flat[flat <= 0.5 * np.nanmedian(flat)] = np.nan
        flat[flat >= 1.5 * np.nanmedian(flat)] = np.nan
    else:
        flat = 1.0

    # container for cube of science data, error cube and data quality
    cube = np.zeros([n, c.shape[len(c.shape) - 2], c.shape[len(c.shape) - 1]])
    err = np.zeros([n, c.shape[len(c.shape) - 2], c.shape[len(c.shape) - 1]])
    dq = np.zeros([n, c.shape[len(c.shape) - 2], c.shape[len(c.shape) - 1]])

    n = 0
    for ifile in range(len(params['files'])):

        # TODO: you shouldn't use '/'  as it is OS dependent
        # TODO: use os.path.join(1, 2, 3)
        misc.printc('Nth file being read :  {}/{}'.format(ifile + 1, len(params['files'])), 'number')
        c = bin_cube(fits.getdata(params['files'][ifile]), params, "Flux")
        if type(c.ravel()[0]) != np.float64:
            misc.printc('\twe change datatype to float64', 'bad2')
            c = np.array(c, dtype=float)

        flag_cds = False

        # if this is a 4-D matrix, we need to perform a CDS
        if len(c.shape) == 4:
            flag_cds = True
            c = c[:, params['cds_id'][0], :, :] - c[:, params['cds_id'][1], :, :]

        cube[n:n + c.shape[0], :, :] = c

        if not flag_cds:
            # if *not* a fgs image, we assume a SOSS image and get the proper extensions
            if 'fg' not in params['files'][ifile]:
                c = bin_cube(fits.getdata(params['files'][ifile], ext=2), params, "Error")
                err[n:n + c.shape[0], :, :] = c
                c = bin_cube(fits.getdata(params['files'][ifile], ext=3), params, "DQ")
                dq[n:n + c.shape[0], :, :] = c
            else:  # a fgs image, we use sqrt(N) as a noise proxy
                err[n:n + c.shape[0], :, :] = np.sqrt(c)
        else:
            err[n:n + c.shape[0], :, :] = np.sqrt(np.abs(c) + params['RON'])
            dq[n:n + c.shape[0], :, :] = ~np.isfinite(c)

        n += c.shape[0]

    if 'flatfile' in params.keys():
        misc.printc('We apply the flat field', 'info')
        for iframe in tqdm(range(cube.shape[0]), leave=False):
            cube[iframe] /= flat
            err[iframe] /= flat

    # patch to avoid annoying zeros in error map
    with warnings.catch_warnings(record=True) as _:
        err[err == 0] = np.nanmin(err[err != 0])

    #  Trick to get a mask where True is valid
    cube_mask = np.zeros_like(cube, dtype=bool)
    for valid_dq in params['valid_dq']:
        misc.printc('Accepting DQ = {}'.format(valid_dq), 'number')
        cube_mask[dq == valid_dq] = True

    cube = remove_background(cube, params)

    cube[~cube_mask] = np.nan
    err[~cube_mask] = np.inf

    if params['allow_temporary']:
        misc.printc('We write intermediate files, they will be read to speed things next time', 'info')
        misc.printc(params['file_temporary_initial_cube'], 'info')
        misc.printc(params['file_temporary_initial_err'], 'info')
        # printc(params['file_temporary_initial_dq'],'info')
        # printc(params['file_temporary_initial_cube_mask'],'info')

        if type(cube[0, 0, 0]) != np.float64:
            misc.printc('\twe change datatype to float64', 'bad2')
            cube = np.array(cube, dtype=float)
        if type(cube[0, 0, 0]) != np.float64:
            misc.printc('\twe change datatype to float64', 'bad2')
            err = np.array(err, dtype=float)
        # if type(cube[0,0,0]) != np.float64:
        #    cube = np.array(cube, dtype = float)

        fits.writeto(params['file_temporary_initial_cube'], cube, overwrite=True)
        fits.writeto(params['file_temporary_initial_err'], err, overwrite=True)
        # fits.writeto(params['file_temporary_initial_dq'], dq, overwrite=True)
        # fits.writeto(params['file_temporary_initial_cube_mask'], np.array(cube_mask, dtype=int), overwrite=True)

    # for future reference in the code, we keep track of data size
    params['DATA_X_SIZE'] = cube.shape[2]
    params['DATA_Y_SIZE'] = cube.shape[1]
    params['DATA_Z_SIZE'] = cube.shape[0]

    return cube, err  # , dq, cube_mask


def get_wavegrid(params, order=1):
    tbl_ref = Table.read(params['pos_file'], order)

    valid = (tbl_ref['X'] > 0) & (tbl_ref['X'] < (params['DATA_X_SIZE'] - 1)) & np.isfinite(
        np.array(tbl_ref['WAVELENGTH']))
    tbl_ref = tbl_ref[valid]
    tbl_ref = tbl_ref[np.argsort(tbl_ref['X'])]

    spl_wave = ius(tbl_ref['X'], tbl_ref['WAVELENGTH'], ext=1, k=1)
    wave = spl_wave(np.arange(params['DATA_X_SIZE']))
    wave[wave == 0] = np.nan

    if (params['mode'] == 'SOSS') and (order == 2):
        tracemap1, throughput = get_trace_pos(params, map2d=True, order=1)
        tracemap2, throughput = get_trace_pos(params, map2d=True, order=2)
        overlap = np.nansum(tracemap1 * tracemap2, axis=0) != 0
        wave[overlap] = np.nan

    return wave
