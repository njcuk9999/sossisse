from astropy.io import fits
import numpy as np
from sossisse import soss_io
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from etienne_tools import robust_polyfit, lowpassfilter
# IUS spline
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import minimize, curve_fit
import yaml
import os

force = False
param_file_or_params = '/Volumes/ariane/sossisse/config_lhs1140b_visit1.yaml'

params = soss_io.load_yaml_params(param_file_or_params )

name_sequence = '{}_{}'.format(params['object'],params['suffix'])
outname = params['pos_file'].replace('SUBSTRIP256',name_sequence)


print('Creating {}'.format(outname))

if not os.path.exists('test.fits'):
    im = fits.getdata(params['files'][0])
    im = np.nanmedian(im,axis=0)
    im[im == 0] = np.nan
    fits.writeto('test.fits', im, overwrite=True)
else:
    im = fits.getdata('test.fits')

im2 = median_filter(im,[3,7])
xpix, ypix = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
w = 20
mask = np.zeros_like(im, dtype=bool)
for i in range(im.shape[1]):
    try:
        imax = np.nanargmax(median_filter(im[:, i],7))
        mask[imax-w:imax+w, i] = True
    except:
        pass

tracepos = (np.nansum(ypix*im*mask,axis=0)/np.nansum(im*mask,axis=0))

params = soss_io.load_yaml_params(param_file_or_params, force=force, do_time_link=True)

pos = Table.read(params['pos_file'])
pos = pos[np.argsort(pos['X'])]
spl = IUS(pos['X'], pos['Y'], k=3, ext=1)
pos = pos[pos['X']>0]
pos = pos[pos['X']<im.shape[1]]
pos2 = Table.read(params['pos_file'],2)
pos2 = pos2[np.argsort(pos2['X'])]
spl2 = IUS(pos2['X'], pos2['Y'], k=3, ext=1)
pos2 = pos2[pos2['X']>0]
pos2 = pos2[pos2['X']<im.shape[1]]
w = 10


def pos_trace(xpix,dx,dy):
    return spl(xpix+dx)+dy

nsig_max = 100
while nsig_max > 5:
    g = np.isfinite(tracepos)
    dxdy_trace = curve_fit(pos_trace, np.arange(im.shape[1])[g], tracepos[g], p0=[0,0])[0]
    traceopos_fit = pos_trace(np.arange(im.shape[1]), *dxdy_trace)

    residual = tracepos-traceopos_fit
    sig_neg_pos = np.nanpercentile(residual, [16, 84])
    nsig_res = np.abs(residual)/(0.5*(sig_neg_pos[1]-sig_neg_pos[0]))
    nsig_max = np.nanmax(nsig_res)
    if nsig_max > 5:
        tracepos[nsig_res > 5] = np.nan
        print('nsig_max = ', nsig_max)
        print('nans = ', np.sum(nsig_res > 5))
        print('')

fig, ax = plt.subplots(nrows= 2, ncols=1, sharex=True)
ax[0].plot(np.arange(im.shape[1]),tracepos,  'r-')
ax[0].plot(np.arange(im.shape[1]),traceopos_fit,  'b-')
ax[0].set_ylabel('Trace position [pix]')
ax[1].plot(np.arange(im.shape[1]),tracepos-traceopos_fit,  'r-')
ax[1].set_ylabel('Residuals [pix]')
plt.show()



pos = Table.read(params['pos_file'])
#fit_wavelength = np.polyfit(pos['X']+dxdy_trace[0], pos['WAVELENGTH'],5)
#fit_throughput = np.polyfit(pos['X']+dxdy_trace[0], pos['THROUGHPUT'],5)
pos['X'] = pos['X']-dxdy_trace[0]
#pos['WAVELENGTH'] = np.polyval(fit_wavelength , pos['X'])
#pos['THROUGHPUT'] = np.polyval(fit_throughput , pos['X'])

pos['Y'] = pos['Y']+dxdy_trace[1]
pos = pos[np.argsort(pos['X'])]

pos2 = Table.read(params['pos_file'],2)
#fit_wavelength = np.polyfit(pos2['X']+dxdy_trace[0], pos2['WAVELENGTH'],5)
#fit_throughput = np.polyfit(pos2['X']+dxdy_trace[0], pos2['THROUGHPUT'],5)
#pos2['THROUGHPUT'] = np.polyval(fit_throughput , pos2['X'])
pos2['X'] = pos2['X']-dxdy_trace[0]
#pos2['WAVELENGTH'] = np.polyval(fit_wavelength , pos2['X'])
#pos2['THROUGHPUT'] = np.polyval(fit_throughput , pos2['X'])

pos2['Y'] = pos2['Y']+dxdy_trace[1]
pos2 = pos2[np.argsort(pos2['X'])]


tracepos = pos_trace(np.arange(im.shape[1]), *dxdy_trace)

mask = np.zeros_like(im, dtype=bool)
w = 20
for i in range(im.shape[1]):
    min_y = int(tracepos[i]-w)
    max_y = int(tracepos[i]+w)
    mask[min_y:max_y, i] = True

spl_wave = IUS(pos['X'], pos['WAVELENGTH'], k=3, ext=1)

spectrum = np.nansum(im*mask,axis=0)
wave = spl_wave(np.arange(im.shape[1]))

spectrum/=lowpassfilter(spectrum,100)

sp_ref = Table.read('/Volumes/ariane/sossisse/M4.5V_Gl268AB.txt',format='ascii')
sp_ref['col2'][sp_ref['col2'] <= 0] = np.nan
sp_ref['col2'] = sp_ref['col2']/np.nanmedian(sp_ref['col2'])
sp_ref['col2'] /= lowpassfilter(sp_ref['col2'],100)

keep = (sp_ref['col1'] > np.nanmin(wave)) & (sp_ref['col1'] < np.nanmax(wave))
sp_ref = sp_ref[keep]



plt.plot(wave, spectrum, 'k-')
plt.plot(sp_ref['col1'], sp_ref['col2'], 'r-')
plt.show()


"""

tbl_sp = Table()
tbl_sp['WAVELENGTH'] = wave
tbl_sp['FLUX'] = spectrum
tbl_sp.write('test_sp.csv', overwrite=True)


plt.imshow(np.sqrt(np.abs(im)), origin='lower', cmap='gray',aspect = 'auto', vmin = np.nanpercentile(np.sqrt(np.abs(im)),1),
           vmax = np.nanpercentile(np.sqrt(np.abs(im)), 99))
plt.plot(pos['X'], pos['Y'], 'g-')
plt.plot(pos2['X'], pos2['Y'], 'r-')
plt.xlim(0,im.shape[1])
plt.ylim(0,im.shape[0])
plt.show()


# write a multi-extension fits file with pos and pos2 as two tables
hdu = fits.PrimaryHDU()
hdulist = fits.HDUList([hdu])
hdulist.append(fits.table_to_hdu(pos))
hdulist.append(fits.table_to_hdu(pos2))
hdulist.writeto(outname, overwrite=True)

params['pos_file'] = outname
params['x_trace_offset'] = 0.0
params['y_trace_offset'] = 0.0
params['recenter_trace_position'] = False
"""