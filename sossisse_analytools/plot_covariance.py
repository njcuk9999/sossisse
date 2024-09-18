from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from etienne_tools import sigma


#1.9µm
tbl1_file = '/Volumes/ariane/sossisse/SOSS/wasp-80/5cbe1e13/csvs/soss_stability_wasp-80_visit1_5cbe1e13.csv'
# 1.6µm
tbl2_file = '/Volumes/ariane/sossisse/SOSS/wasp-80/fcfef7e1/csvs/soss_stability_wasp-80_visit1_fcfef7e1.csv'

tbl1 = Table.read(tbl1_file)
tbl1['index'] = np.arange(len(tbl1))
tbl2 = Table.read(tbl2_file)
tbl2['index'] = np.arange(len(tbl2))
tbl1['amplitude']-= np.mean(tbl1['amplitude'])
tbl2['amplitude']-= np.mean(tbl2['amplitude'])

plt.errorbar(tbl1['amplitude'], tbl2['amplitude'], xerr = tbl1['amplitude_error'], yerr = tbl2['amplitude_error'],
             fmt = '.', alpha = 0.7)
plt.xlabel('1.9µm amplitude - 1')
plt.ylabel('1.6µm amplitude - 1')
plt.tight_layout()
plt.savefig('wasp-80_amplitude_correlation.png')
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.errorbar(tbl1['index'], tbl1['amplitude'], yerr = tbl1['amplitude_error'], label='1.9µm', fmt = '.',
               color = 'orange', alpha = 0.7)
ax.errorbar(tbl2['index'], tbl2['amplitude'], yerr = tbl2['amplitude_error'], label='1.6µm', fmt = '.',
               color = 'blue', alpha = 0.7)
ax.legend()
ax.set_ylabel('Amplitude')
ax.set_xlabel('Nth frame')
plt.tight_layout()
ax.set(title = 'WASP-80')
plt.savefig('wasp-80_amplitude.png')
plt.show()


fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
ax[0].errorbar(tbl1['index'], tbl1['amplitude'], yerr = tbl1['amplitude_error'], label='1.9µm', fmt = '.',
               color = 'orange', alpha = 0.7)
ax[0].errorbar(tbl2['index'], tbl2['amplitude'], yerr = tbl2['amplitude_error'], label='1.6µm', fmt = '.',
               color = 'blue', alpha = 0.7)
ax[0].legend()
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Nth frame')
tbl3 = Table(tbl2)
tbl3['amplitude'] = tbl2['amplitude'] - tbl1['amplitude']
tbl3['amplitude_error'] = np.sqrt(tbl1['amplitude_error']**2 + tbl2['amplitude_error']**2)
ax[1].errorbar(tbl2['index'], tbl3['amplitude'], yerr = tbl3['amplitude_error'], label='1.9µm-1.6µm',
               fmt = '.', color =  'green', alpha = 0.7)

ax[1].set_ylabel('Color amplitude')
ax[1].set_xlabel('Nth frame')
plt.tight_layout()
ax[0].set(title = 'WASP-80')
plt.show()


# autocorrelation
tbl1['amplitude'] -= np.mean(tbl1['amplitude'])
ac1 = np.correlate(np.array(tbl1['amplitude']), np.array(tbl1['amplitude']), mode='full')
tbl3['amplitude'] -= np.mean(tbl3['amplitude'])
ac3 = np.correlate( np.array(tbl3['amplitude']),  np.array(tbl3['amplitude']), mode='full')


plt.plot(ac3[:len(ac3)//2][::-1], label='1.9µm-1.6µm', alpha= 0.8 )
plt.plot(ac1[:len(ac1)//2][::-1], label='1.9µm', alpha= 0.8 )
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()