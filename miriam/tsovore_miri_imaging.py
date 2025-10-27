import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits

from jwst import datamodels
# Level 1
from jwst.group_scale import GroupScaleStep
from jwst.dq_init import DQInitStep
from jwst.emicorr import EmiCorrStep
from jwst.saturation import SaturationStep
from jwst.ipc import IPCStep
from jwst.firstframe import FirstFrameStep
from jwst.lastframe import LastFrameStep
from jwst.reset import ResetStep
from jwst.linearity import LinearityStep
from jwst.rscd import RscdStep
from jwst.dark_current import DarkCurrentStep
from jwst.refpix import RefPixStep
from jwst.charge_migration import ChargeMigrationStep
from jwst.jump import JumpStep
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep
from jwst.gain_scale import GainScaleStep
# Level 2
from jwst.background import BackgroundStep
from jwst.assign_wcs import AssignWcsStep
from jwst.flatfield import FlatFieldStep
from jwst.photom import PhotomStep

'''
To know what paramaters a step accepts, at the command line type:
% strun jwst.jump.JumpStep -h
% strun jwst.firstframe.FirstFrameStep -h
etc...
'''

# Example of how one can control the paramaters passed to each step. Here for
# the jump step.
#
# This applies when using the call to Image1Pipeline which itself calls
# the single steps. It does not work in the step by step example of this python
# script:
# cfg = dict()
# cfg['jump'] = {} # set up empty dictionary for multiple parameters to be set per step
# cfg['jump']['rejection_threshold'] = 5 # default is 4.0, min=0
# cfg['jump']['max_shower_amplitude'] = 4 # default is 4
# cfg['jump']['three_group_rejection_threshold'] = 6 # default is 6.0
# cfg['jump']['four_group_rejection_threshold'] = 5 # default is 5.0
# cfg['jump']['only_use_ints'] = True # default True
#
# Here is what works for single steps:
jumpargs = {"rejection_threshold": 5,
            "max_shower_amplitude": 4,
            "three_group_rejection_threshold": 6,
            "four_group_rejection_threshold": 5,
            "only_use_ints": True}

# If set to False, the fits file produced by each step will be kept on disk
# rather than erased (if True).
REMOVE_INTERMEDIATE_FITS = False

datadir = './MIRI_03730_obs009/'
uncalfiles = ['jw03730009001_03101_00001-seg001_mirimage_uncal.fits',
              'jw03730009001_03101_00001-seg002_mirimage_uncal.fits',
              'jw03730009001_03101_00001-seg003_mirimage_uncal.fits',
              'jw03730009001_03101_00001-seg004_mirimage_uncal.fits',
              'jw03730009001_03101_00001-seg005_mirimage_uncal.fits']

# Number of segments in the time-series
nsegments = np.size(uncalfiles)

# Create a list of basenames to properly name the rateints.fits at the
# RampFitStep step
basename = uncalfiles.copy()
for i in range(nsegments):
    tmp = os.path.basename(uncalfiles[i])
    basename[i] = tmp.split('_uncal')[0]

# Add the path to the names
for i in range(nsegments):
    uncalfiles[i] = datadir + uncalfiles[i]
    print(uncalfiles[i])

# Define the list of the last saved files on disk. Initialize with the uncal.
list_lastondisk = uncalfiles.copy()

# Level 1 processing
# Perform the first 4 steps in a loop - they will likely never have to be
# split apart so might as well loop to keep result in memory rather than
# having to write to disk between each step as is the case for later steps.
for i, filename in enumerate(uncalfiles):
    # Read input segments files from disk
    input_data = datamodels.open(filename, output_dir=datadir, save_results=False)
    # Execute the first step
    result = GroupScaleStep.call(input_data)
    # Execute the second step
    result = DQInitStep.call(result, output_dir=datadir, save_results=False)
    # Execute the third step
    # The EmiCorrStep is by default skipped when executed on these data
    # result = EmiCorrStep.call(input_data, save_results=False)
    # Execute the fourth step
    # this time saving files on disk
    result = SaturationStep.call(result, output_dir=datadir, save_results=True)
    list_lastondisk[i] = datadir + result.meta.filename

# Each of the subsequent step is performed one by one rather than grouped.
# That provides a good framework to more easily insert custom steps in between
# which, for example, would require all previous steps to be complete to work.

# Perform the 5th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = IPCStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 6th step - FirstFrameStep
# All this does is set the DQflag of the first frame of every integration
# to DO NOT USE provided that the integrations have more than 3 usable (non-
# saturated group).
# TODO: Replace this step with a copy and simply change that so that more than
# 1 frame is set to DO NOT USE (if that is what Etienne needs)
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = FirstFrameStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 7th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = LastFrameStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 8th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = ResetStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 9th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = LinearityStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 10th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = RscdStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 11th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = DarkCurrentStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 12th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = RefPixStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# The ChargeMigrationStep is by default skipped when executed on these data
## Perform the 13th step
# for i,filename in enumerate(list_lastondisk):
#    # Read input segments files from disk
#    input_data = datamodels.open(filename)
#    result = ChargeMigrationStep.call(input_data, output_dir=datadir, save_results=True)
#    # To limit disk usage, erase the file from the previously saved step
#    os.remove(list_lastondisk[i])
#    # Save the last file saved on disk in our list
#    list_lastondisk[i] = datadir+result.meta.filename

# Perform the 14th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = JumpStep.call(input_data, output_dir=datadir, save_results=True,
                           **jumpargs)
    # Fot the JumpStep, no new outfile file is created, instead, the previous
    # step's file is modified when running the JumpStep. I don't know why.
    ## To limit disk usage, erase the file from the previously saved step
    # if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# The CleanFlickerStep is by derfault skipped when executed on these data
## Perform the 15th step
# for i,filename in enumerate(list_lastondisk):
#    # Read input segments files from disk
#    input_data = datamodels.open(filename)
#    result = CleanFlickerNoiseStep.call(input_data, output_dir=datadir, save_results=True)
#    # To limit disk usage, erase the file from the previously saved step
#    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
#    # Save the last file saved on disk in our list
#    list_lastondisk[i] = datadir+result.meta.filename


# Perform the 16th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    rate, result = RampFitStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform the 17th step
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = GainScaleStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# The background step in order to work needs a list of *** dithered ***
# exposures which we don't have with TSO. Skip that step.
## Perform level 2 - BackgroundStep
# for i,filename in enumerate(list_lastondisk):
#    # Read input segments files from disk
#    input_data = datamodels.open(filename)
#    result = BackgroundStep.call(input_data, output_dir=datadir, save_results=True)
#    # To limit disk usage, erase the file from the previously saved step
#    os.remove(list_lastondisk[i])
#    # Save the last file saved on disk in our list
#    list_lastondisk[i] = datadir+result.meta.filename

# Perform level 2 - AssignWcsStep
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = AssignWcsStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    if REMOVE_INTERMEDIATE_FITS: os.remove(list_lastondisk[i])
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

# Perform level 2 - FlatFieldStep
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = FlatFieldStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

    # Rename the output files to adhere to the MAST convention.
    # _rateints.fits <-- _1_flatfieldstep.fits
    mast_name = datadir + basename[i] + '_rateints.fits'
    os.rename(list_lastondisk[i], mast_name)
    list_lastondisk[i] = mast_name

# Perform level 2 - PhotomStep
for i, filename in enumerate(list_lastondisk):
    # Read input segments files from disk
    input_data = datamodels.open(filename)
    result = PhotomStep.call(input_data, output_dir=datadir, save_results=True)
    # To limit disk usage, erase the file from the previously saved step
    # Save the last file saved on disk in our list
    list_lastondisk[i] = datadir + result.meta.filename

    # Rename the output files to adhere to the MAST convention.
    # _rateints.fits <-- _1_flatfieldstep.fits
    mast_name = datadir + basename[i] + '_calints.fits'
    os.rename(list_lastondisk[i], mast_name)
    list_lastondisk[i] = mast_name
