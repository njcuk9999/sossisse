import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
# import median filtering
from scipy.signal import medfilt2d
# scipy optimization
from etienne_tools import lin_mini

from scipy.ndimage import affine_transform

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from astropy.table import Table


tbl_wavesol = Table.read('SOSS_ref_trace_table_lhs1140b_visit1.fits')

spl = ius(tbl_wavesol['X'],tbl_wavesol['WAVELENGTH'],k=1,ext=1)

def rot_shift(image, dx, dy, theta):
    # transform the image with a rotation/shift in dx and dy
    cd_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    return affine_transform(image, cd_matrix, offset=[dx, dy], order=2, mode='constant', cval=0)


def correl(a, b):
    # gives the correlation coefficient between a and b maps
    a = np.array(a)
    b = np.array(b)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    # a and b are 2D arrays
    # a is the reference
    # b is the target
    # return the correlation coefficient between a and b
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)

    return np.nansum(a * b) / np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))


def reg2pix(reg_file):
    # read a ds9 region file and return a mask of the region
    # read file content within a 'with' block
    with open(reg_file) as f:
        content = f.readlines()

    # remove lines that do not have a box in the string
    content = [x for x in content if 'box' in x]

    reg_table = Table()
    reg_table['x'] = [float(x.split('(')[1].split(',')[0]) for x in content]
    reg_table['y'] = [float(x.split(',')[1].split(',')[0]) for x in content]
    reg_table['w'] = [float(x.split(',')[2].split(',')[0]) for x in content]
    reg_table['h'] = [float(x.split(',')[3].split(',')[0]) for x in content]

    mask = np.zeros([256, 2048])
    for i in range(len(reg_table)):
        x = reg_table['x'][i]
        y = reg_table['y'][i]
        w = reg_table['w'][i]
        h = reg_table['h'][i]
        mask[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = 1

    return mask

# read the file for which we want to adjust the contamination trace position
mef_file = 'median_ootmed1_bkg1_1fpolyord0_fitrot1_fitzp0_fitddy1_it1-425-it4-770.fits'
med = fits.getdata(mef_file)

# read the proxy of the contamination. It needs to be of similar spectral type
# and not have too many field stars contaminating it.
ref_trace_file = 'med_wasp80.fits'
ref_trace = fits.getdata(ref_trace_file)
ref_trace = np.array(ref_trace)

# median filter the reference trace. You don't want bad pixels in there
# ton contaminate the other image. The filter size is 7 in the spectral dimension
# and 1 (no filter) in the spatial as you would cut the 'horns' of the SOSS
# trace.
for ite in range(2):
    ref_trace2 = medfilt2d(ref_trace, kernel_size=[1, 7])
    bad = ~np.isfinite(ref_trace2)
    ref_trace2[bad] = ref_trace[bad]
    ref_trace = ref_trace2

# remove the bad pixels and set 0
ref_trace[~np.isfinite(ref_trace)] = 0

# read the mask
mask_ref = reg2pix('ds9.reg') == 1
mask_order1 = reg2pix('ds9_order1.reg') == 1

# compute the gradient of the reference trace in the spatial dimension only
gradx = np.gradient(med, axis=0)
grad_ref = np.gradient(ref_trace, axis=0)


fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

vmin1 = np.nanpercentile(gradx, 10)
vmax1 = np.nanpercentile(gradx, 90)
ax[0].imshow(gradx, vmin=vmin1, vmax=vmax1, aspect='auto')
ax[0].set_title('Target')
vmin2 = np.nanpercentile(grad_ref, 1)
vmax2 = np.nanpercentile(grad_ref, 99)
ax[1].imshow(grad_ref, vmin=vmin2, vmax=vmax2, aspect='auto')
ax[1].set_title('Reference')
plt.show()


best_dx = 0
best_dy = 0

v1 = gradx[mask_ref]
for ite in range(5):
    # we iteratively adjust the position of the contamination trace
    # in x and y. We do this by computing the correlation coefficient
    # between the target and the reference trace. We then shift the
    # reference trace by one pixel in x and y and compute the correlation
    # coefficient again. We keep the shift that maximizes the correlation
    # coefficient.
    dd = np.arange(-200, 200)
    c = np.zeros_like(dd, dtype=float)

    for i in range(len(dd)):
        dx = dd[i]
        if np.abs(dx + best_dx) < 20:
            continue
        gradx2 = np.roll(grad_ref, dx + best_dx, axis=0)
        gradx2 = np.roll(gradx2, best_dy, axis=1)
        c[i] = correl(v1, gradx2[mask_ref])

    best_dx = dd[np.nanargmax(c)] + best_dx
    plt.plot(dd, c)

    for i in range(len(dd)):
        dy = dd[i]
        gradx2 = np.roll(grad_ref, best_dx, axis=0)
        gradx2 = np.roll(gradx2, dy + best_dy, axis=1)
        c[i] = correl(v1, gradx2[mask_ref])
    best_dy = dd[np.nanargmax(c)] + best_dy
    print(best_dx, best_dy)
    plt.plot(dd, c)
    plt.show()

gradx2 = np.roll(np.roll(grad_ref, best_dx, axis=0), best_dy, axis=1)
ref_trace2 = np.roll(np.roll(ref_trace, best_dx, axis=0), best_dy, axis=1)

best_dx = ((best_dx + 512) % 256)
if best_dx > 128:
    best_dx = best_dx - 256

xx, yy = np.indices(ref_trace2.shape)
ref_trace2_slope = ref_trace2 * yy

dtrace = np.gradient(ref_trace2, axis=0)

sample = np.array([ref_trace2[mask_ref],
                   ref_trace2_slope[mask_ref],
                   gradx2[mask_ref],
                   np.ones_like(gradx2[mask_ref]),
                   xx[mask_ref],
                   yy[mask_ref]])
sample = np.array(sample)
v = med[mask_ref]

fit, _ = lin_mini(v, sample)

trace3 = ref_trace2 * fit[0] + ref_trace2_slope * fit[1] + gradx2 * fit[2]

if best_dy > 0:
    trace3[:, 0:best_dy] = 0
if best_dy < 0:
    trace3[:, best_dy:] = 0

if best_dx > 0:
    trace3[0:best_dx, :] = 0
if best_dx < 0:
    trace3[best_dx:, :] = 0

# now that we have found the shift, we find the amplitude. We project the trace that is contaminating onto the
# target trace and compute the amplitude of the contamination. This is done pixelwise and we use the median of the
# target amplitudes to set the 'global' amplitude of the contamination.
amp = np.nansum(gradx2 * gradx * mask_ref, axis=0) / np.nansum(gradx2 ** 2 * mask_ref, axis=0)
plt.plot(amp)
tmp = (med - trace3)
gradx = np.gradient(tmp, axis=0)
amp = np.nansum(gradx2 * gradx * mask_ref, axis=0) / np.nansum(gradx2 ** 2 * mask_ref, axis=0)
plt.plot(amp)
plt.show()

fits.writeto('contamination_trace_lhs1140_visit1.fits', trace3, overwrite=True)
fits.writeto('test2.fits', tmp, overwrite=True)


amp1 = np.nansum(med**2,axis=0)
amp2 = np.nansum(trace3*med,axis=0)

x = np.arange(len(amp1))
w = spl(x)
plt.plot(w,amp2/amp1)
plt.show()