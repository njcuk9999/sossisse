from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def odd_ratio_linfit(x, y, yerr):
    """
    Fit a linear model to the data using an iterative weighted least squares method.

    :param x: Abscissa
    :param y: Ordinate
    :param yerr: Error on the ordinate
    :return: Linear fit and error on the fit
    """
    # Remove NaN values
    g = np.isfinite(y + yerr + x)
    x = x[g]
    y = y[g]
    yerr = yerr[g]
    # Initialize weights
    w = np.ones(len(x))

    # Iterate until weights converge
    sum = 1.0
    sum0 = 0.0
    while np.abs(sum0 - sum) > 1e-6:
        sum0 = np.sum(w)
        # Fit the data with current weights
        fit, sig = np.polyfit(x, y, 1, w=w / yerr, cov=True)
        errfit = np.sqrt(np.diag(sig))
        # Compute residuals and update weights
        res = (y - np.polyval(fit, x)) / yerr
        p1 = np.exp(-0.5 * res ** 2)
        p2 = 1e-6
        w = p1 / (p1 + p2)
        sum = np.sum(w)

    return fit, errfit
key1 = '88d0cd07' # 10th percentile
key2 = '5eb939b5' # 20th percentile
key3 = 'a808f66c' # 30th percentile
key4 = '1ffe0985' # box =30
key5 = 'c7495c4f' # box = 20

labels = [ '20th percentile', '10th percentile', '30th percentile', 'box = 30', 'box = 20']

keys = [key1, key2, key3, key4, key5]

fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)
path_to_sed = '/Volumes/ariane/sossisse/SOSS/trappist-1f/{}/csvs/sed_trappist-1f_ord1.csv'
for ikey, key in enumerate(keys):


    tbl = Table.read(path_to_sed.format(key))
    tbl['flux']/=np.nanpercentile(tbl['flux'],80)
    ax[0].plot(tbl['wavelength'], tbl['flux'], label=labels[ikey])

    if ikey == 0:
        ref_flux = tbl['flux']
    else:
        ax[1].plot(tbl['wavelength'], tbl['flux']-ref_flux, label=labels[ikey])

ax[0].legend()
ax[0].set(xlabel='Wavelength (microns)', ylabel='Flux')
ax[1].legend()
ax[1].set(xlabel='Wavelength (microns)', ylabel='Flux difference to 20th percentile')
plt.show()

it = [69, 77, 100, 106]
it_start = 1
it_end = 2


fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)
for ikey, key in enumerate(keys):

    file = f'/Volumes/ariane/sossisse/SOSS/trappist-1f/{key}/fits/spectra_ord1_trappist' \
           f'-1f_visit2_{key}_oddratio.fits'

    wavelength = fits.getdata(file, 'WAVELENGTH')
    relflux = fits.getdata(file, 'RELFLUX')
    relflux_error = fits.getdata(file, 'RELFLUX_ERROR')
    sp = np.nanmean(relflux[it[it_start]:it[it_end], :], axis=0)
    err = 1/np.sqrt(np.nansum(1/relflux_error[it[it_start]:it[it_end], :]**2, axis=0))

    fit, sig_fit = odd_ratio_linfit(wavelength[0]-1.6, sp-1, err)

    print(f'Fit for {labels[ikey]}: zero point = {fit[1]*1e6:.1f} +/- {sig_fit[1]*1e6:.1f}, slope = {fit[0]*1e6:.1f} +/- '
          f'{sig_fit[0]*1e6:.1f}')

    #ax[0].plot(wavelength[0], sp, label=labels[ikey])
    ax[0].fill_between(wavelength[0], sp-err, sp+err, alpha=0.5, label=labels[ikey])

    fit[1]+=1
    ax[0].fill_between(wavelength[0], np.polyval(fit, wavelength[0]-1.6)-np.polyval(sig_fit, wavelength[0]-1.6),
                          np.polyval(fit, wavelength[0]-1.6)+np.polyval(sig_fit, wavelength[0]-1.6), alpha=0.5)

    if ikey == 0:
        ref_sp = sp
        err_ref = err
    else:
        ax[1].plot(wavelength[0], sp-ref_sp, label=labels[ikey], alpha=0.3)
        ax[1].fill_between(wavelength[0], sp-ref_sp-err, sp-ref_sp+err, alpha=0.5, label=labels[ikey])

        fit, sig_fit = odd_ratio_linfit(wavelength[0] - 1.6, sp - ref_sp, np.sqrt(err**2+err_ref**2))

        print(
            f'\tDifference for {labels[ikey]}: zero point = {fit[1] * 1e6:.1f} +/- {sig_fit[1] * 1e6:.1f}, slope ='
            f' {fit[0] * 1e6:.1f} +/- '
            f'{sig_fit[0] * 1e6:.1f}')

    ax[0].legend()
ax[0].set(xlabel='Wavelength (microns)', ylabel='Relative flux')
ax[0].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax[1].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax[1].legend()
ax[1].set(xlabel='Wavelength (microns)', ylabel='Relative flux difference to 20th percentile')
plt.show()