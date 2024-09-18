import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
# import median filtering
from scipy.signal import medfilt2d
from tqdm import tqdm
# scipy optimization
from scipy.optimize import minimize

from scipy.ndimage import affine_transform

def rot_shift(image, dx, dy, theta):
    cd_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    return affine_transform(image, cd_matrix, offset=[dx, dy], order=2, mode='constant', cval=0)



def correl(a,b):
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

    return np.nansum(a*b)/np.sqrt(np.nansum(a**2)*np.nansum(b**2))


reg_file = 'ds9.reg'
mef_file = 'median_ootmed1_bkg1_1fpolyord0_fitrot1_fitzp0_fitddy1_it1-425-it4-770.fits'
med = fits.getdata(mef_file)

ref_trace_file = 'med_wasp80.fits'

ref_trace = fits.getdata(ref_trace_file)
for ite in range(2):
    ref_trace2 = medfilt2d(ref_trace, kernel_size=[1,7])
    keep = np.isfinite(ref_trace2) & (np.isnan(ref_trace))
    ref_trace[keep] = ref_trace2[keep]

ref_trace[~np.isfinite(ref_trace)] = 0
ref_trace = medfilt2d(ref_trace, kernel_size=[1, 5])
grad_ref = np.gradient(ref_trace,axis = 0)


gradx = np.gradient(med,axis = 0)

mask = np.zeros_like(med)

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

for i in range(len(reg_table)):
    x = reg_table['x'][i]
    y = reg_table['y'][i]
    w = reg_table['w'][i]
    h = reg_table['h'][i]
    mask[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)] = 1

p1, p99 = np.percentile(gradx[mask==1],[1,99])
mask[gradx<p1] = 0
mask[gradx>p99] = 0
mask = mask==1
v1 = gradx[mask]

w = 200
dxs, dys = np.indices([2*w+1,2*w+1])-w
c = np.zeros_like(dxs,dtype = float)

for j in tqdm(range(len(dys)), leave = False):
    for i in tqdm(range(len(dxs)),leave = False):
        if c[i,j] ==0:
            dx = dxs[i, j]
            gradx2 = np.roll(grad_ref, dx, axis=0)
            dy = dys[i,j]
            gradx2 = np.roll(gradx2,dy,axis = 1)
            c[i,j] = correl(v1,gradx2[mask])

            if c[i,j]<0.2:
                c[i,j-20:j+20] = c[i,j]


plt.imshow(c)
plt.show()

ddx = dxs.ravel()[np.argmax(c.ravel())]
ddy = dys.ravel()[np.argmax(c.ravel())]

grad_ref2 = np.roll(grad_ref,ddx,axis = 0)
grad_ref2 = np.roll(grad_ref2, ddy,axis = 1)


p1, p99 = np.percentile(gradx[mask==1],[1,99])
p1b,p99b = np.percentile(grad_ref2[mask==1],[1,99])

amp = np.nansum(gradx*grad_ref2*mask,axis=0)/np.nansum(grad_ref2**2*mask,axis=0)



ref_trace2 = np.roll(ref_trace,ddx,axis = 0)
ref_trace2 = np.roll(ref_trace2, ddy,axis = 1)

def func_mini(grad_ref2, theta, dx,dy):
    grad_ref3 = rot_shift(grad_ref2,dx,dy,theta)
    c = correl(gradx[mask],grad_ref3[mask])
    print(theta,dx,dy)
    return 1-c

fit = minimize(func_mini, [0,0,0],  method = 'Nelder-Mead')

print(fit)

tmp = (med - ref_trace2*amp)
fits.writeto('test2.fits',tmp, overwrite = True )

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = True)
ax[0].imshow(grad_ref2*mask*amp, origin = 'lower', cmap = 'gray', aspect = 'auto', vmin = p1, vmax = p99)
ax[1].imshow(gradx*mask, origin = 'lower', cmap = 'gray', aspect = 'auto', vmin = p1, vmax = p99)
ax[2].imshow((gradx - grad_ref2*amp)*mask, origin = 'lower', cmap = 'gray', aspect = 'auto', vmin = p1, vmax = p99)
plt.show()