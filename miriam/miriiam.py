import numpy as np
from astropy.io import fits
import os
from scipy.optimize import curve_fit, minimize

from numba import njit, prange
from tqdm import tqdm

from astropy.table import Table

import matplotlib.pyplot as plt

def whoami():
    # return the name of the user in the terminal
    return os.environ['USER']

# import the Time to get a the 'now'
from astropy.time import Time



def prefix_now():
    return Time.now().iso.split(' ')[-1]+' | '

def robust_sigma(x):
    return (np.percentile(x, 84) - np.percentile(x, 16)) / 2


def least_square(all_cds2, all_cds_err2, psf, gradx, grady, psf2=None):
    """
    Perform a weighted linear least squares fit to model all_cds2 as a linear combination of
    psf, gradx, grady, and psf2. Errors are propagated using all_cds_err2.

    Parameters
    ----------
    all_cds2 : ndarray
        Flattened data array to fit.
    all_cds_err2 : ndarray
        Flattened error array for each pixel.
    psf, gradx, grady, psf2 : ndarray
        2D arrays (same shape as image) for the basis functions.
        psf2 is optional

    Returns
    -------
    fit : ndarray
        Best-fit amplitudes for [psf, gradx, grady, psf2].
    sig : ndarray
        1-sigma uncertainties on the amplitudes.
    """
    # Flatten basis arrays to match all_cds2
    if psf2 is not None:
        A = np.vstack([
            psf.flatten(),
            gradx.flatten(),
            grady.flatten(),
            psf2.flatten()
        ]).T  # shape (Npix, 4)
    else:
        A = np.vstack([
            psf.flatten(),
            gradx.flatten(),
            grady.flatten()
        ]).T  # shape (Npix, 3)

    # Only use finite data and errors
    mask = np.isfinite(all_cds2) & np.isfinite(all_cds_err2)
    y = all_cds2[mask]
    sigma = all_cds_err2[mask]
    A = A[mask]

    # Weighted least squares: minimize sum(((A @ x - y)/sigma)**2)
    # Solution: x = inv(A^T W A) @ (A^T W y), W = diag(1/sigma^2)
    W = 1.0 / sigma**2
    Aw = A * W[:, None]
    cov = np.linalg.inv(Aw.T @ A)
    fit = cov @ (Aw.T @ y)
    sig = np.sqrt(np.diag(cov))

    return fit, sig



def printc(string, color = 'black'):
    """
    This function prints a string in the specified color.

    :param string: The string to be printed
    :param color: The color of the string
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[30m'
    }
    # get terminal width
    columns = os.get_terminal_size().columns
    now = prefix_now()

    print(colors[color] + now + string + '\033[0m')

def plot_correlations(tbl, filename = None, plot_pedestal = True, segment_gaps = True):
    # We plot the correlations between the different parameters

    ufiles = np.unique(tbl['FILENAME'])
    tbl['NTH_FILE'] = 0
    for i, uf in enumerate(ufiles):
        tbl['NTH_FILE'][tbl['FILENAME'] == uf] = i

    if plot_pedestal:
        fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    else:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    ax[0].errorbar(tbl['int_mid_BJD_TDB'], tbl['amps'], yerr = tbl['amps_err'], alpha = 0.3, fmt = '.')
    ax[0].set_ylabel('Normalized flux')
    ax[0].grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    if segment_gaps:
        for i in range(1,len(tbl)):
            if tbl['NTH_FILE'][i] != tbl['NTH_FILE'][i-1]:
                ax[0].axvline(tbl['int_mid_BJD_TDB'][i], color='pink', linestyle='-', ymin = 0, ymax= 1.0)

    ax[1].errorbar(tbl['int_mid_BJD_TDB'], tbl['dxs'], yerr = tbl['dxs_err'], alpha = 0.3 , label =r"$b_x$", fmt = '.')
    ax[1].errorbar(tbl['int_mid_BJD_TDB'], tbl['dys'], yerr = tbl['dys_err'], alpha = 0.3 , label =r"$b_y$", fmt = '.')
    ax[1].legend()
    ax[1].set_ylabel('Position offset \n[pixels]')
    ax[1].grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    ax[2].errorbar(tbl['int_mid_BJD_TDB'], tbl['d2s'], yerr = tbl['d2s_err'], alpha = 0.3, fmt = '.')
    if not plot_pedestal:
        ax[2].set_xlabel('Time [BJD]')
    ax[2].set_ylabel(r"Shape parameter ($c_\perp$)")
    ax[2].grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    if plot_pedestal:
        ax[3].plot(tbl['int_mid_BJD_TDB'], tbl['pedestal_mean'],'.', alpha = 0.3, label = 'Pedestal Mean')
        ax[3].plot(tbl['int_mid_BJD_TDB'], tbl['pedestal_psf'],'.', alpha = 0.3, label = 'Pedestal PSF')
        ax[3].set_xlabel('Time [BJD]')
        ax[3].set_ylabel('Pedestal')
        ax[3].legend()
        ax[3].grid(color = 'gray', linestyle = '--', linewidth = 0.5)


    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def remove_background2d(image):
    """
    
    Remove the background from a 2D image.
    
    This is done by fitting a sinc2d with a pedestal. We only
    subtract the pedestal and return the image. The sinc2d 
    function is just there to remove the bias from the PSF and
    not subtracted in the end.
    
    """

    imax = np.argmax(image)
    xmax = imax % image.shape[1]
    ymax = imax // image.shape[1]

    # we construct the 2d pixel values
    ypix, xpix = np.indices(image.shape)
    # we find valid pixels
    valid = np.isfinite(image)
    # we ravel this into a 1d array for posterity
    xy = np.array([xpix[valid], ypix[valid]])

    def sinc2d(xy,amp,per,pedestal_psf,xcen,ycen,bwidth=0.1):

        x,y = xy
        x0 = x-xcen
        y0 = y-ycen

        rad = np.sqrt(x0**2 + y0**2)

        # we create 5 sinc to simulate bandpass
        dband = 1+(np.arange(5)-2)/4.0*bwidth

        outprofile = np.zeros_like(rad)
        for dd in dband:
            per2 = per*dd
            outprofile+=(amp/len(dband))*np.sinc(rad / np.pi/per2)**2

        return outprofile+pedestal_psf

    # best guess as the psf center. this is the brightest pixel
    xmax = np.nanargmax(image) % image.shape[1]
    ymax = np.nanargmax(image) // image.shape[1]

    # amplitude best guess is image peak
    # sinc with =3 in pixels
    # pedestal = median
    # xmax and y max are best guess of psf center
    # 0.2 is the psf bandwidth guess
    p0 = np.array([np.nanmax(image), 3.0, np.nanmedian(image), xmax, ymax,.2])

    popt, _ = curve_fit(sinc2d,xy,image[valid], p0=p0)

    # item [2] is the pedestal that we subtract
    return image-popt[2], popt[2]


def remove_backbground_cube(hyper_cube):
    """
    This function removes the background from the CDS data.

    :param cube: The input cube data
    :return: The background-subtracted cube data
    """

    printc('Removing background from the cube', 'green')

    def remove_from_3d(cube3d):
        pedestal = np.zeros(cube3d.shape[0])
        # Loop through each slice of the cube. We do this as a 3D cube to avoid
        # but this may go into a 4D hypercube. 
        for i in tqdm(range(cube3d.shape[0]), leave=False):
            sl = cube3d[i, :, :]
            med0 = np.nanmedian(sl)

            # Subtract the median of each quadrant
            sl[::2, ::2] -= np.nanmedian(sl[::2, ::2])
            sl[1::2, ::2] -= np.nanmedian(sl[1::2, ::2])
            sl[::2, 1::2] -= np.nanmedian(sl[::2, 1::2])
            sl[1::2, 1::2] -= np.nanmedian(sl[1::2, 1::2])
            sl2 = np.array(sl+med0)

            cube3d[i, :, :], pedestal[i] = remove_background2d(sl2)    
        return cube3d, pedestal

    # if a 4D hypercube is given, we loop through each cube.
    if len(hyper_cube.shape) == 4:
        for i in tqdm(range(hyper_cube.shape[0]), leave=False):
            hyper_cube[i], pedestal[i] = remove_from_3d(hyper_cube[i])
    else:
        hyper_cube, pedestal = remove_from_3d(hyper_cube)

    return hyper_cube, pedestal

def asqrt(v):
    # This function returns the square root of the input value
    # if the input value is negative, it returns zero
    v2 = np.sqrt(np.abs(v))
    v2*= np.sign(v)
    return v2

def neighbourbad(image):
    # for each NaN pixel, find the 3x3 box (less if at the edge) and fit a 2d 2nd order
    # polynomial for find the best estimate of the missing pixel

    ybad, xbad = np.where(~np.isfinite(image))


    ypixgrid, xpixgrid =  np.meshgrid([-1,0,1],[-1,0,1])

    for ibad in range(len(xbad)):
        xpix = xbad[ibad]+xpixgrid
        ypix = ybad[ibad]+ypixgrid

        # remove xpix and ypix that would be off the edge of the image.shape
        rem_pix = (xpix < 0) | (xpix >= image.shape[1]) | (ypix < 0) | (ypix >= image.shape[0])
        xpix = xpix[~rem_pix]
        ypix = ypix[~rem_pix]

        # Fit a 2D polynomial to the surrounding pixels
        z = image[ypix, xpix]
        """

        g = np.isfinite(z)

        X = xpix[g].astype(float)
        Y = ypix[g].astype(float)
        Z = z[g].astype(float)

        A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
        B = Z.flatten()

        coeff, r, rank, s = np.linalg.lstsq(A, B, rcond = None)

        v = np.polynomial.polynomial.polyval2d(ypix[ibad],xpix[ibad],coeff)
        """
        image[ybad[ibad], xbad[ibad]] = np.nanmean(z)
    
    return image

def noise_cube(cube):
    # Calculate the 16th, 50th (median), and 84th percentiles along the first axis
    n1, med, p1 = np.nanpercentile(cube, [16, 50, 84], axis=0)
    # Define negative and positive cut thresholds
    cut_neg = med - 25 * (med - n1)
    cut_pos = med + 25 * (p1 - med)
    # Initialize an array for cube errors
    cube_err = np.zeros_like(cube)
    nanbad = ~np.isfinite(cube)

    # Loop through each slice of the cube
    for i in range(cube.shape[0]):
        # Identify bad pixels that are outside the cut thresholds
        bad = np.logical_or(cube[i] < cut_neg, cube[i] > cut_pos)
        # Calculate the error for each pixel
        cube_err[i] = (p1 - n1) / 2
        # Set the error of bad pixels to NaN
        cube_err[i][bad] = np.inf

        cube[i] = neighbourbad(cube[i])
    cube_err[nanbad] = np.inf

    return cube, cube_err

def hyper_cube_errors(hcube):
    # We find the errors in the cube by calculating the 16th and 84th percentiles
    # for each slice. It may be a 3d or 4d


    if len(hcube.shape) == 4:
        hcube_err = np.zeros_like(hcube)
        printc('Calculating the errors in the hypercube', 'green')        
        for i in tqdm(range(hcube.shape[0]), leave=False):
            tmp1, tmp2 = noise_cube(hcube[i])
            hcube[i] = tmp1
            hcube_err[i] = tmp2
    else:
        hcube, hcube_err = noise_cube(hcube)

    return hcube, hcube_err


def read_miri_cube(filename):
    """
    This function reads the MIRI cube data from the given file. It also r
    emoves the zero point of the uncal file.

    :param filename: The path to the file
    :return: The MIRI cube data
    """
    if 'uncal' in filename:
        printc('We remove the zero point of the uncal file')
        # Read the cube data and header
        cube = fits.getdata(filename).astype(float)

        h = fits.getheader(filename)

        # Calculate the zero point
        zp = cube[:, 0, :, :] * 2 - cube[:, 1, :, :]

        # Remove the zero point from each slice
        for islice in tqdm(range(cube.shape[1]), leave=False):
            cube[:, islice, :, :] -= zp

        # Correct for odd-even differences
        #for islice in range(cube.shape[1]):
        #    odd_even = np.nanmedian(cube[:, islice, ::2, :] - cube[:, islice, 1::2, :].astype(float))
        #    printc('Odd-even difference: {}, islice: {}'.format(odd_even, islice))
        #    cube[:, islice, ::2, :] -= odd_even / 2
        #    cube[:, islice, 1::2, :] += odd_even / 2

        # Write the zero-point subtracted cube to a new file
        zp_name = filename.replace('uncal.fits', 'zpsub.fits')
        printc('Writing the zero-point subtracted cube to {}'.format(zp_name))
        fits.writeto(zp_name, cube, h, overwrite=True)

    # Read the zero-point subtracted cube
    printc('Reading the zero-point subtracted cube from {}'.format(zp_name))
    cube = fits.getdata(zp_name)


    return cube


def cube2phot(cube, error=None, tbl=None,  d2flag = False,ww=1.0, file_plot_psf = None):
    # This function takes as an input a cube and returns the photometry
    # derived from the cube. 

    # TODO be consistent with rateints
    if error is None:
        # We assume that the error is the square root of the cube
        error = np.sqrt(np.abs(cube))
    else:
        error = np.array(error)
 
    # Normalize the CDS data. We first calculate the median of the cube
    # and then do a linear fit to the median. We then divide the cube by
    # the linear fit to normalize the data.

    # TODO this could be trimmed, for ex the first readouts that have a different
    # morphology.
    cds_normalize = np.array(cube)
    psf_guess = np.nanmedian(cds_normalize, axis=0)
    for i in range(cds_normalize.shape[0]):
        # We make a copy of the median and set the invalid values to NaN
        # this is to avoid the using a NaN value in the linear fit.
        # Also, this is done at each slice since the NaNs are different
        # for each slice.
        psf_guess0 = np.array(psf_guess)
        valid = np.isfinite(cube[i])
        psf_guess0[~valid] = np.nan
        amp = np.nansum(cube[i] * psf_guess0 * ww**2) / np.nansum((psf_guess0*ww)**2)
        # We normalize the data by the linear fit
        cds_normalize[i] = cds_normalize[i] / amp

    # Calculate the PSF and its gradients
    psf = np.nanmedian(cds_normalize, axis=0)

    if len(psf.shape) == 2:    
        grady, gradx = np.gradient(psf)
    else:
        grady, gradx,_ = np.gradient(psf)


    psf2 = psf**2
    ## TODO we could minimise sum of squares also
    #def decorr(amp):
    #    psf_residual = psf2 - amp*psf
    #    dot =  np.corrcoef(psf.ravel(),psf_residual.ravel())[0,1]
    #    return  np.abs(dot)

    #mini = minimize(decorr,  np.sum(psf*psf2)/np.sum(psf**2),method = 'Nelder-Mead')
    # psf2 is the square of the psf minus the psf times an amplitude that makes it 
    # orthogonal to psf



    bad = ~np.isfinite(psf)
    ww_masked = np.array(ww)
    ww_masked[bad] = 0
    K = np.nansum(psf*psf2*ww_masked**2)/np.nansum(psf**2*ww_masked**2)
    psf2ortho = (psf2 - K * psf)
    
    if file_plot_psf is not None:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))

        # List of images and titles
        images = [psf, psf2ortho, gradx, grady]
        titles = [
            r'$M_{i,j}$',
            r'$D_{i,j}$',
            r'$\frac{\partial M_{i,j}}{\partial x}$',
            r'$\frac{\partial M_{i,j}}{\partial y}$'
        ]

        for a, img, title in zip(ax.flat, images, titles):
            a.imshow((img/np.max(np.abs(img))), origin='lower', cmap='viridis', vmin=-1, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            # Put the label *inside* the axes, upper left corner
            a.text(
                0.04, 0.96, title,
                transform=a.transAxes,
                fontsize=24,
                va='top', ha='left',
                bbox=dict(facecolor='wheat', edgecolor='none',
                        boxstyle='round,pad=0.3'))

        # Make spacing much tighter
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.tight_layout()
        plt.savefig(file_plot_psf, dpi=300)
        plt.show()

    # Initial guess for the PSF model parameters. We assume a scale of 1
    # and no gradients. This is a good starting point for the fit.
    p0 = [0.95, 0.1, 0.1, 1e-5]
    p0 = np.array(p0, dtype=np.float32) 

    # Initialize arrays to store the fit results and errors
    amps = np.zeros(cube.shape[0])
    amps_err = np.zeros(cube.shape[0])
    dxs = np.zeros(cube.shape[0])
    dxs_err = np.zeros(cube.shape[0])
    dys = np.zeros(cube.shape[0])
    dys_err = np.zeros(cube.shape[0])
    d2s = np.zeros(cube.shape[0])
    d2s_err = np.zeros(cube.shape[0])

    corr = np.zeros(cube.shape[0])
    # Loop through each slice of the CDS data to fit the PSF model
    for i in tqdm(range(cube.shape[0]), leave=False, desc='Fitting PSF model to CDS data'):
        all_cds_updated = np.array(cube[i], dtype=np.float32)
        mask_nan = ~np.isfinite(cube[i]+error[i])
        all_cds_updated[mask_nan] = 0
        # We keep a mask here to avoid dividing by zero
        weights = np.ones_like(all_cds_updated, dtype = np.float32)
        weights[mask_nan] = 0
        all_cds_err_updated = np.array(error[i])
        # bad pixels are set to errors of infinity
        all_cds_err_updated[mask_nan] = np.inf

        if d2flag:
            fit, sig = least_square(all_cds_updated.ravel(), all_cds_err_updated.ravel(), psf.ravel(), gradx.ravel(), grady.ravel(), psf2ortho.ravel())
            psf_reconstructed = psf*fit[0]+fit[1]*gradx+fit[2]*grady+fit[3]*psf2ortho
        else:
            fit, sig = least_square(all_cds_updated.ravel(), all_cds_err_updated.ravel(), psf.ravel(), gradx.ravel(), grady.ravel())
            psf_reconstructed = psf*fit[0]+fit[1]*gradx+fit[2]*grady

        corr[i] = np.nansum(psf_reconstructed*ww)/np.nansum(psf*ww)


        # Store the fit results and errors
        amps[i] = corr[i] # amplitude of the PSF
        amps_err[i] = sig[0] # error in the amplitude
        dxs[i] = fit[1] # offset in x
        dxs_err[i] = sig[1] # error in the offset in x
        dys[i] = fit[2] # offset in y
        dys_err[i] = sig[2] # error in the offset in y
        if d2flag:
            d2s[i] = fit[3] # 2nd order gradient, representative of the FWHM
            d2s_err[i] = sig[3] # error in the 2nd order gradient

    if tbl is None:
        tbl = Table()
    else:
        tbl = Table(tbl)

    norm = np.nanmedian(amps)
    tbl['amps'] = amps/norm
    tbl['amps_err'] = amps_err/norm
    tbl['dxs'] = dxs
    tbl['dxs_err'] = dxs_err
    tbl['dys'] = dys
    tbl['dys_err'] = dys_err
    tbl['d2s'] = d2s
    tbl['d2s_err'] = d2s_err

    return tbl

def get_stats(x):
    # Calculate the 16th and 84th percentiles of the input array
    n1, p1 = np.percentile(x, [16, 84])
    # Print the robust standard deviation
    printc('Robust STD : {:.3e}'.format(0.5 * (p1 - n1)))

    # Calculate the rolling robust standard deviation
    n1, p1 = np.percentile(x - np.roll(x, 1), [16, 84]) / np.sqrt(2)
    printc('Rolling robust STD : {:.3e}'.format(0.5 * (p1 - n1)))

@njit(parallel=True)
def mask_outliers(cube):
    # Calculate the 16th, 50th (median), and 84th percentiles along the first axis
    n1, med, p1 = np.percentile(cube, [16, 50, 84], axis=0)
    # Define negative and positive cut thresholds
    cut_neg = med - 3 * (med - n1)
    cut_pos = med + 3 * (p1 - med)
    # Initialize an array for cube errors
    cube_err = np.zeros_like(cube)

    # Loop through each slice of the cube
    for i in prange(cube.shape[0]):
        # Identify bad pixels that are outside the cut thresholds
        bad = np.logical_or(cube[i] < cut_neg, cube[i] > cut_pos)
        # Set bad pixels to NaN
        cube[i][bad] = np.nan
        # Calculate the error for each pixel
        cube_err[i] = (cut_pos - cut_neg) / 2
        # Set the error of bad pixels to NaN
        cube_err[i][bad] = np.nan

    return cube, cube_err

def smart_errors(val, err):
    """
    This function calculates the number of digits in the error and formats the value and error accordingly.

    :param val: The value to be formatted
    :param err: The error associated with the value
    :return: A string with the formatted value and error
    """
    # Calculate the number of digits in the error
    ndigits = int(np.log10(err)) - 1

    if ndigits < 0:
        nn = str(-ndigits)
        v = '{:.' + nn + 'f}'
    else:
        v = '{:.0f}'
        val = np.round(val, -ndigits)
        err = np.round(err, -ndigits)

    # Format the value and error
    out = v.format(val) + ' +/- ' + v.format(err)

    return out


def double_decay(x, a1, a2, t1, t2, zp):
    """
    This function calculates the double decay model. This is a sum of two exponential decays with different time constants
    and a zero point.

    :param x: The input array
    :param a1: Amplitude of the first decay
    :param a2: Amplitude of the second decay
    :param t1: Time constant of the first decay
    :param t2: Time constant of the second decay
    :param zp: Zero point
    :return: The double decay model
    """
    return a1 * np.exp(-x / t1) + a2 * np.exp(-x / t2) + zp



@njit(parallel=True)
def subract_zp(cube, zp):
    """
    This function subtracts the zero point from the input cube.

    :param cube: The input cube data
    :param zp: The zero point data
    :return: The zero-point subtracted cube data
    """
    # Loop through each slice of the cube
    for islice in prange(cube.shape[1]):
        cube[:, islice, :, :] -= zp

    return cube

def subtract_odd_even(cube):
    """
    The subtract_odd_even function is designed to correct for
    odd-even differences in the input data cube. This is typically
    necessary in imaging data where there might be systematic
    differences between odd and even indexed elements, which
    can introduce artifacts into the data. The function calculates
    the median difference between odd and even indexed elements
    along a specific axis and then adjusts the data to remove
    this difference. This helps in normalizing the data and
    reducing systematic errors.

    :param cube: The input cube data
    :return: The cube data with odd-even differences removed
    """
    # Loop through each slice of the cube
    for islice in range(cube.shape[1]):
        # Calculate the odd-even difference
        odd_even = np.nanmedian(cube[:, islice, ::2, :] - cube[:, islice, 1::2, :])
        printc('Odd-even difference: {}, islice: {}'.format(odd_even, islice))
        # Remove the odd-even difference
        cube[:, islice, ::2, :] -= odd_even / 2
        cube[:, islice, 1::2, :] += odd_even / 2

    return cube

def read_miri_cube(filename):
    """
    This function reads the MIRI cube data from the given file. It also removes the zero point of the uncal file.

    :param filename: The path to the file
    :return: The MIRI cube data
    """
    if 'uncal' in filename:
        # Write the zero-point subtracted cube to a new file
        zp_name = filename.replace('uncal.fits', 'zpsub.fits')
        if not os.path.isfile(zp_name):

            printc('We remove the zero point of the uncal file')
            # Read the cube data and header
            cube = fits.getdata(filename).astype(float)
            h = fits.getheader(filename)

            # Compute the zero point
            zp = cube[:, 0, :, :] * 2 - cube[:, 1, :, :]

            # Remove the zero point from each slice
            printc('Subtracting zero point...')
            cube = subract_zp(cube, zp)
            printc('Subtracting odd-even differences...')
            cube = subtract_odd_even(cube)

            fits.writeto(zp_name, cube, h, overwrite=True)
        else:
            cube = fits.getdata(zp_name)
    else:
        # Read the zero-point subtracted cube
        cube = fits.getdata(filename)
    return cube

def remove_backbground_cds(cube):
    """
    This function removes the background from the CDS data.

    :param cube: The input cube data
    :return: The background-subtracted cube data
    """
    # Loop through each slice of the cube
    for i in range(cube.shape[0]):
        printc('Removing background from slice {}/{}'.format(i + 1, cube.shape[0]))
        sl = cube[i, :, :]
        # Subtract the median of each quadrant
        sl[::2, ::2] -= np.nanmedian(sl[::2, ::2])
        sl[1::2, ::2] -= np.nanmedian(sl[1::2, ::2])
        sl[::2, 1::2] -= np.nanmedian(sl[::2, 1::2])
        sl[1::2, 1::2] -= np.nanmedian(sl[1::2, 1::2])
        sl2 = np.array(sl)
        # Set outliers to NaN
        sl2[sl2 > 2 * np.nanpercentile(sl2, 95)] = np.nan
        # Subtract the median along the first axis
        med = np.nanmedian(sl2, axis=0)
        for ii in range(sl.shape[0]):
            sl[ii, :] -= med
            sl2[ii, :] -= med
        # Subtract the median along the second axis
        med = np.nanmedian(sl2, axis=1)
        for ii in range(sl.shape[1]):
            sl[:, ii] -= med
            sl2[:, ii] -= med
        cube[i, :, :] = sl

    return cube