import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.special import erf

import miriiam as mi

def get_stats(x):
    # Calculate the 16th and 84th percentiles of the input array
    n1, p1 = np.percentile(x, [16, 84])
    # Print the robust standard deviation
    mi.printc('Robust STD : {:.3e}'.format(0.5 * (p1 - n1)))

    # Calculate the rolling robust standard deviation
    n1, p1 = np.percentile(x - np.roll(x, 1), [16, 84]) / np.sqrt(2)
    mi.printc('Rolling robust STD : {:.3e}'.format(0.5 * (p1 - n1)))

def sigma(x):
    n1, p1 = np.nanpercentile(x, [16, 84])
    return (p1 - n1) / 2

def smart_errors(val, err):
    """
    This function calculates the number of digits in the error and formats the value and error accordingly.

    :param val: The value to be formatted
    :param err: The error associated with the value
    :return: A string with the formatted value and error
    """
    # Calculate the number of digits in the error
    ndigits = int(np.log10(err)) - 2

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
    This function calculates the double decay model. This is a sum of two 
    exponential decays with different time constants
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

# same as above with a single decay
def single_decay(x, a1, t1, zp):
    """
    This function calculates the single decay model. This is an exponential 
    decay with a time constant and a zero point.

    :param x: The input array
    :param a1: Amplitude of the decay
    :param t1: Time constant of the decay
    :param zp: Zero point
    :return: The single decay model
    """
    return a1 * np.exp(-x / t1) + zp


#### MAIN ####
### INPUTS ###

whoami = mi.whoami()


#if whoami == 'jwst':
#    path0 = f'/genesis/jwst/miri/MIRI_lkurve/{batch}'
#else:
#    path0 = f'/Users/eartigau/mnt/miri/MIRI_lkurve/{batch}'

#path0 = '/Users/eartigau/mnt/BELUGA/fortune/{}'.format(batch)

where_is_data = 'beluga' # or 'beluga'
if where_is_data == 'miri':
    batch = 'eclipse3'
    path0 = '/Users/eartigau/mnt/miri/MIRI_lkurve/{}'.format(batch)
    file_type = 'rateints'
    force = True
elif where_is_data == 'beluga':
    batch = 'Eclipse3'
    path0 = '/Users/eartigau/mnt/BELUGA/fortune/{}'.format(batch)
    file_type = 'calints'
    force = True

else:
    raise ValueError('where_is_data should be "miri" or "beluga"')

#
#file_type = 'uncal'
#file_type = 'calints'

outname = f'{path0}/photometry_{batch}.rdb'

if False:
    if os.path.isfile(outname):
        print(f'File {outname} already exists')
        print('Exiting')

        tbl = Table.read(outname, format='rdb')
        mi.plot_correlations(tbl)
        exit()

# Define the path to the merged CDS file
merged_cds_name = f'{path0}/cds.fits'

# Define the template for the cube file paths
cube_files = f'{path0}/*_{file_type}.fits'


cube_files = np.array(glob.glob(cube_files))
cube_files = cube_files[np.argsort(cube_files)]

if len(cube_files) == 0:
    raise ValueError('No cube files found')

# Define the list of exposures to skip
skip_exposure = [0, 1]  # skip these exposures
# Define the width of the box for cropping
wbox = 32  # in pixels

# Define the timestep in seconds
timestep = 11.082503758370876  # in seconds
# Define the number of frames during the eclipse
nframes_eclipse = 367
# Define the start and end frames of the eclipse
t1 = 600
t2 = t1 + nframes_eclipse
# Define the number of Monte Carlo simulations
N_MC = 1000

# Define whether to add curvature to the light curve model
add_curvature = True

### END INPUTS ###

# Check if the merged CDS file does not exist
if not os.path.isfile(merged_cds_name) or force:
    # Check if the merged CDS file does not exist
    for ncube in range(len(cube_files)):
        cube_file = cube_files[ncube]
        # Format the cube file path using the current file ID
        cds_name = cube_file.replace(f'{file_type}.fits', 'cds.fits')
        # Generate the CDS file name by replacing 'uncal.fits' with 'cds.fits'
        cds_table = cube_file.replace(f'{file_type}.fits', 'cds_table.rdb')
        # Generate the CDS table name by replacing 'uncal.fits' with 'cds_table.rdb'

        if not os.path.isfile(cds_table) or force:
            # Check if the CDS table file does not exist
            mi.printc('We read the INT_TIMES from the cube file extension INT_TIMES')
            tbl0 = Table(fits.getdata(cube_file, 'INT_TIMES'))
            tbl0['FILENAME'] = cube_files[ncube]

            # Read the INT_TIMES data from the cube file into a table
            tbl0.write(cds_table, format='rdb',overwrite=True)
            # Write the table to the CDS table file in 'rdb' format
        else:
            mi.printc('We read the INT_TIMES from the existing CDS table file')
            tbl0 = Table.read(cds_table, format='rdb')
            # Read the existing CDS table file into a table

        if not os.path.isfile(cds_name) or force:
            # Check if the CDS file does not exist

            if file_type == 'uncal':
                mi.printc('We read the cube file')
                cube = mi.read_miri_cube(cube_file)
                # Read the MIRI cube data from the cube file
                mi.printc('We perform a CDS')
                # Perform a CDS (Correlated Double Sampling) by subtracting the first
                # readout from the second-to-last readout
                cds = cube[:, -2, :, :] - cube[:, 0, :, :]
                mi.printc('We remove the background from the CDS')
                cds = mi.remove_backbground_cds(cds)
                cds, cds_err = mi.hyper_cube_errors(cds)

            else:
                cds = fits.getdata(cube_file,'SCI')
                meds = np.nanmedian(cds, axis=(1,2))

                for islice in range(cds.shape[0]):
                    frame = cds[islice,:, :]
                    pix = np.arange(frame.shape[1])
                    for ite in range(2):
                        for icol in range(4):
                            frame[:, icol::4] -= np.nanmedian(frame[:, icol::4])
                        med = np.nanmedian(frame, axis=0)
                        fit = np.polyfit(pix,med,1)
                        for i in range(frame.shape[0]):
                            frame[i, :] -= fit[0] * pix + fit[1]

                        for irow in range(4):
                            frame[irow::4, :] -= np.nanmedian(frame[irow::4, :])


                    cds[islice,:, :] = frame

                """
                psf_med = np.nanmedian(cds, axis=0)
                for islice in range(cds.shape[0]):
                    res = cds[islice] - psf_med
                    model = np.zeros(res.shape)
                    medx = np.nanmedian(res, axis=1)
                    medy = np.nanmedian(res, axis=0)
                    for i in range(res.shape[0]):
                        model[i, :] = medx[i]
                    for i in range(res.shape[1]):
                        model[:, i] = medy[i]
                    print(sigma(res)/sigma(res-model))
                """


                # TODO use the DQ for the errors. Set to inf for values that are flagged as invalid
                cds_err = fits.getdata(cube_file,'ERR')

                # Read the MIRI cube data from the cube file


            # Remove the background from the CDS data
            mi.printc(f'We write the CDS data to {cds_name}')
            fits.writeto(cds_name, cds, overwrite=True)
            # Write the CDS data to the CDS file, overwriting if it already exists
        else:
            mi.printc('We read the existing CDS file')
            cds = fits.getdata(cds_name)
            #cds, cds_err = mi.hyper_cube_errors(cds)
            # Read the existing CDS file data

        # TODO revisit centering
        # Crop the all_cds array to a smaller region centered around the middle of the array
        cds = cds[:, 128 - wbox // 2:128 + wbox // 2, 128 - wbox // 2:128 + wbox // 2]
        # Crop the all_cds_err array to the same region
        cds_err = cds_err[:, 128 - wbox // 2:128 + wbox // 2, 128 - wbox // 2:128 + wbox // 2]

        if ncube == 0:
            # If this is the first cube file being processed
            tbl = tbl0
            # Set the table to the current table
            all_cds = cds
            # same for cds_err
            all_cds_err = cds_err
            # Set the all_cds array to the current CDS data
        else:
            tbl = vstack([tbl, tbl0])
            # Vertically stack the current table with the previous tables
            all_cds = np.append(all_cds, cds, axis=0)
            all_cds_err = np.append(all_cds_err, cds_err, axis=0)
            # Append the current CDS data to the all_cds array along the first axis


    # Write the cropped all_cds array to the merged CDS file, overwriting if it already exists
    fits.writeto(merged_cds_name, all_cds, overwrite=True)
    # Write the cropped all_cds_err array to a separate file with '_err.fits' suffix, overwriting if it already exists
    fits.writeto(merged_cds_name.replace('.fits', '_err.fits'), all_cds_err, overwrite=True)
    # Write the table to a file with '_table.rdb' suffix in 'rdb' format
    tbl.write(merged_cds_name.replace('.fits', '_table.rdb'), format='rdb',overwrite=True)
else:
    # If the merged CDS file already exists, read the all_cds array from the file
    all_cds = fits.getdata(merged_cds_name)
    # Read the all_cds_err array from the file with '_err.fits' suffix
    all_cds_err = fits.getdata(merged_cds_name.replace('.fits', '_err.fits'))
    # Read the table from the file with '_table.rdb' suffix in 'rdb' format
    tbl = Table.read(merged_cds_name.replace('.fits', '_table.rdb'), format='rdb')

# We need to remove the exposures that are not used as defined in the skip_exposure list
index = np.ones(len(tbl), dtype=bool)
for i in skip_exposure:
    index[i] = False
tbl = tbl[index]
all_cds = all_cds[index]
all_cds_err = all_cds_err[index]

radius = 16
iy, ix = np.indices(all_cds[0].shape)
ix -= all_cds[0].shape[1] // 2
iy -= all_cds[0].shape[0] // 2

# this is a soft-edged mask of ones and zeros.
# TODO us an erf at the edge to go from 1 to zero
rad = np.sqrt(ix ** 2 + iy ** 2) - radius
ww = 1-(erf(rad*2)/2+0.5)
ww_invert = 1-ww


print(all_cds.shape)
pedestal_mean = np.zeros(all_cds.shape[0])
for i in range(all_cds.shape[0]):
    valid = np.isfinite(all_cds[i])
    pedestal_mean[i] = np.sum((all_cds[i]*ww_invert)[valid])/np.sum(ww_invert[valid])

tbl['pedestal_mean'] = pedestal_mean

# We remove the background from all the 2d images of the cube or hypercube
cube2, pedestal = mi.remove_backbground_cube(all_cds)

# TODO decide if we want to compute errors or keep errors from the original cube files
# We calculate the errors of the cube
cube2, cube2_err = mi.hyper_cube_errors(cube2)

tbl['pedestal_psf'] = pedestal

tbl0 = mi.cube2phot(cube2, cube2_err, tbl, d2flag = True, ww=ww,file_plot_psf = f'{path0}/psf_d2flag_{batch}.pdf')
tbl1 = mi.cube2phot(cube2, cube2_err, tbl, d2flag = False,ww=ww)

s3 = tbl0['amps']
print('RMS global {:.1f}ppm / rms p2p {:.1f}ppm --> d2 corrigé'.format(sigma(s3)*1e6,sigma(s3-np.roll(s3,1))/np.sqrt(2)*1e6))

s4 = tbl1['amps']
print('RMS global {:.1f}ppm / rms p2p {:.1f}ppm --> d2 ignoré'.format(sigma(s4)*1e6,sigma(s4-np.roll(s4,1))/np.sqrt(2)*1e6))

# We plot the photometric measurements
mi.plot_correlations(tbl0, filename=f'{path0}/correlations_d2flag_{batch}.pdf',plot_pedestal = False, segment_gaps = False)
plt.show()
plt.close()
# We plot the photometric measurements
mi.plot_correlations(tbl1, filename=f'{path0}/correlations_d2flag_false_{batch}.pdf',plot_pedestal = False, segment_gaps = False)
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
ax[0].plot(tbl0['int_mid_BJD_TDB'], tbl0['amps'], 'g.', label='D2 Flag True')
ax[0].plot(tbl1['int_mid_BJD_TDB'], tbl1['amps'], 'r.', label='D2 Flag False')
ax[0].set_ylabel('Flux Amplitude')
ax[0].set_title(f'{batch} - Photometric Measurements')
ax[1].set_xlabel('Time [BJD]')
ax[1].set_ylabel('Flux difference')
ax[1].plot(tbl0['int_mid_BJD_TDB'], tbl0['amps'] - tbl1['amps'], 'k.', label='Flux Difference')
ax[0].legend()
plt.savefig(f'{path0}/photometry_{batch}.png', dpi=300)
plt.show()

mi.printc('We write the photometry table to ' + outname)
tbl0.write(outname, format='rdb',overwrite=True)
