# SOSSISSE - SOSS Inspired SpectroScopic Extraction

- [Installation](https://github.com/njcuk9999/sossisse/blob/main/README.md#installation)
- [Example data set](https://github.com/njcuk9999/sossisse/blob/main/README.md#example-data-set)
- [Description of code](https://github.com/njcuk9999/sossisse/blob/main/README.md#description-of-code)


## Installation

### Step 1: Download the GitHub repository

```
git clone git@github.com:njcuk9999/sossisse.git
```

### Step 2: Install python 3.8 using conda

Using conda, create a new environment and activate it

```
conda create --name sossisse-env python=3.9
```

```
conda activate sossisse-env
```

### Step 3: Install sossisse

```
cd {SOSSISSE_ROOT}

pip install -U -e .
```

Note one can also use venv (instead of conda)

Note `{SOSSISSE_ROOT}` is the path to the cloned GitHub repository (i.e. `/path/to/sossisse`)


## Example data set

The sample dataset provided as a demonstration of the code is the transit of TRAPPIST-1b observed with the SOSS mode.
The data is saved in two fits files (rateints data product) that sample 150 frames. The transit happens between 
frames 90 and 110 (1st and 4th contact) with a depth of ~0.88%. The point-to-point RMS of the white light curve over 
the SOSS domain is ~160 ppt.

## Description of code

### Before getting started

All parameters for the SOSSISSE code, from the inputs files to the detrending parameters, are passed through a 
single yaml file. One should not edit the python codes themselves, but rather edit the yaml file to change any 
relevant value. As it is likely that users will want to try  a number of parameter combinations, SOSSISSE creates a 
unique hash key for each parameter combination. This key is used to name the output files and folder name. The 
hash key is not meant to be an explicit description of the parameter combination, but rather a unique identifier. 
The hash key is created from a truncated checksum of the yaml file. As the link between the hash key, and the input 
yaml cannot be readily determined, the yaml file is copied within the output folder it created.

To run the codes, one needs to define a path for the subfolders and the yaml. This path should be defined as a 
system variable called `SOSSIOPATH` (SOSS-input-output path). 

#### On Mac

Within the `~/.zshrc` on Mac, this is done with the 
following line added to the startup profile : 

```
export SOSSIOPATH=/YourPath/Your_SOSSISSE_folder/
```

#### On Linux

Within the `~/.bashrc` on Linux, this is done with the
following line added to the startup profile : 

```
export SOSSIOPATH=/YourPath/Your_SOSSISSE_folder/
```

#### On Windows

For windows you can follow the guide [here](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html#GUID-DD6F9982-60D5-48F6-8270-A27EC53807D0)
The environment variable name should be `SOSSIOPATH` and the value should be the path to the sossisse directory e.g. `C:\YourPath\Your_SOSSISSE_folder\`

Note after this point we will refer to all paths using hte unix (Linux/Mac) notation, adjust accordingly for Windows.


### Data structure

Within that folder, one should define a subfolder for the 'mode' used (here 'SOSS') and a per-target folder that 
will hold the ouputs from the analysis and includes a *rawdata* subfolder that contains the raw data.

Create the following folder structure : 

```
/YourPath/Your_SOSSISSE_folder/SOSS/
/YourPath/Your_SOSSISSE_folder/SOSS/t1b/
/YourPath/Your_SOSSISSE_folder/SOSS/t1b/rawdata/
```

where `t1b` is the name of the target (in this case `t1b` for Trappist-1b).

Within the `/YourPath/Your_SOSSISSE_folder/SOSS/t1b/rawdata/` folder, you should place the raw data files.

A demo dataset (Trappist-1b) can be dowloaded from http://www.astro.umontreal.ca/~artigau/soss/t1b_sample.tar. 

You will also need to put reference files, in the relevant subfolders, that can be downloaded from http://www.astro.umontreal.ca/~artigau/soss/ref_files.tar.
Place them for example in the following folder: `/YourPath/Your_SOSSISSE_folder/SOSS/calibrations/*.fits `

All yaml files should be placed in (or symbolically liked to) the `/YourPath/Your_SOSSISSE_folder/` folder.
You will find an example yaml file in the sossisse github directory: `sossisse/data/defaults.yaml`
As well as in the demo dataset for Trappist-1.


### Running the code

The code is run in the python terminal with the following command : 

```
In [1] import sossisse
In [2]: sossisse.wrapper('config_t1b.yaml')
```

### Understanding outputs


#### White light curve
The first step of the code being a determination of the *grey* lightcurve, we first have a photometric curve with an 
obvious transit signature. Note that the time axis is in 'frames', which correspond to the **it**, 'in transit', 
parameters (4 terms; Nth frame for 1st, 2nd, 3rd and 4th contact). In this plot, the values measured between 1st and 4th contact 
are in red while baseline (before 1st and after 4th) are coded in green.

![White light curve for the transit](resources/lightcurve.png)

**Figure 1.** White light curve for the transit with color coding to highlight 'in' and 'out' of transit data. A rough fit 
of the transit depth is also shown.

#### 2D trace and residual image
The logic behind SOSSISSE is to construct a model of the trace and subtract it from the data. The model is 
constructed as a fixed-structure trace onto which we add a perturbation terms on a given number of parameters that 
are constructed as the derivatives of the trace with respect to the perturbation parameters. In the most simplistic 
case, one could recenter the trace by adding onto it the spatial derivative of the trace with respect to the *x* and 
*y* axis of the pixel grid. To understand systematics that are unaccounted for by this model, we construct a 
residual map of the data after the subtraction of the model. This residual map is then used to remove some low-level 
residuals such as  leftover sky residuals, or low-level detector effects. The residual map encodes spectral 
information on the transit signal.

![Residual plot](resources/residuals.png)

** Figure 2.** [top] Sample 2D image of the trace with a highlighted region where the trace is scaled in amplitude. The 
white 
pixels correspond to NaN values (i.e. pixels that are not used in the analysis). [bottom] Same as Figure 2 but following the subtraction of the scaled trace model and residual low-level 
structures.  This image encodes the planet spectroscopic signal. Noise levels vary from pixel to pixel and is taken 
into account 
when extracting the 1D *residual* spectrum.

#### Detrending parameter drifts
As an ancillary output, we provide the amplitude of all terms used in the detrending model. The parameters against 
which we detrend are the *x* and *y* position of the trace, the *x* and *y* position of the trace center, the 
rotation of the trace, the trace amplitude, a pedestal to the trance and the trace width. The amplitude of the linear 
drifts for each of these 
are shown in Figure 3. The red points correspond to in-transit data. One sees that the trace center offsets in *x* 
are affected by periodic variations that are linked to the ~200s periodic error in the primary mirror shape [ref].

![Detrending parameter drifts](resources/drifts.png)

**Figure 3.** Amplitude of linear drifts for each detrending parameter. The red points correspond to in-transit data.

#### Per-order spectral energy distribution
 This is done as a 'convenience' output as the main goal of 
SOSSISSE is to get a very accurate *differential* extraction of the planet spectrum. This can be used to confirm the 
 wavelength calibration, which is not possible directly in residual space. The SED is constructed by summing the 
 flux in the *y* direction for each order. The SED is then normalized to the median flux in the out-of-transit data. 
 The SED accounts for the wavelength-dependent throughput of the instrument.


![SED-order 1](resources/sed_ord1.png)
![SED-order 2](resources/sed_ord2.png)

**Figure 4.** Spectral energy distribution for each order.

### Description of the code

First of all, one neeeds to create a yaml file that contains all the parameters for the analysis. A yaml file is 
provided as an example (see section below). Here are the main steps of the analysis :

-- **Step 1** : Read the yaml file and check that all the parameters are present and have the correct format.

-- **Step 2** : Read the raw data and the calibration files. The raw data are read as a cube of 3D images. The
calibration files are read as a cube of 2D images. The calibration files are used to construct a master flat field
and a master bad pixel mask. The master flat field is used to correct for the pixel-to-pixel variations in the
detector response. We use the per-pixel QC value to flag bad pixels. We correct the raw 
data for the flat field and the bad pixels.

-- **Step 3** : Construct a normalized reference trace. This is done by medianing all the frames in the raw data 
cube. This is done iteratively as one expects the amplitude to change slightly from frame to frame.

-- **Step 4** : Construct derivatives of the trace with respect to the detrending parameters. This assumes that the 
trace only in its morphology by very small amounts and that linear perturbations are sufficient to model it through 
time. Of course this assumption is expected to break at some point, if only because of the chromatic nature of 
transits, but this will be captured by the residual map.

-- **Step 5** : Construct a model of the trace. This is done by adding the perturbation terms to the normalized trace.

-- **Step 6** : Construct a residual map. This is done by subtracting the trace model from the raw data.

-- **Step 7** : Subtract low-level detector noises (column-wise offset or gradients) from the residual map. This is 
done by fitting a low-order polynomial to the residual map and subtracting it (see corresponding parameter in yaml).

-- **Step 8** : Construct a 1D spectrum of the residuals. This is done by summing the flux in the *y* direction and 
accounts for the variations in the noise level from pixel to pixel (1/$\sigma^2$ weighting). Errors are also propagated.

-- **Step 9** : Construct a 1D spectrum of the trace model. This is done by summing the flux in the *y* direction 
and provides an SED estimate.

### Sample yaml file

```
\###############################################################################
\##  Definition of inputs related to the data location and file naming
\###############################################################################
#
# Folder structure :
# -> that's where your input data should be
#  / sossiopath / mode / object / rawdata / files  
# -> that's where you put calibration files relevant for the mode  
#  / sossiopath / mode / calibrations / *.fits     
# -> that's where your temporary *FITS* files will be saved
#  / sossiopath / mode / object / temporary_*unique_id* / 
# -> plots for that run  
#  / sossiopath / mode / object / plots_*unique_id* / 
# -> csv files for that run
#  / sossiopath / mode / object / csv_*unique_id* /
#
# *unique_id* will be a long random string constructed from the checksum of the 
# config file
#
# sosspath *must* be defined as a system constant "sossiopath"
#
object: "trappist-1b"
# Only NIRISS "SOSS" and NIRPSEC "PRISM" modes are supported at the moment
mode: "SOSS"
#
suffix: "visit1"

#
files:
- "jw02589001001_04101_00001-seg001_nis_rateints.fits"
- "jw02589001001_04101_00001-seg002_nis_rateints.fits"

# CALIBRATION FILES -- put in the / general_path / mode / calibrations folder
#
# background file --> leave as '' if the is no backgorund available for the mode
bkgfile: "model_background256.fits"
#
# flat field file --> leave as '' when no flat field available for the mode
flatfile: "jwst_niriss_flat_0275.fits"
#
# trace position file
pos_file: "SOSS_ref_trace_table_SUBSTRIP256.fits"
#
# allow for temporary files to speed the process if you run the code more than 
# once. This saves the temporary files in the temporary folder, which is more 
# efficient but takes more space. If you want to save space, set this to False
allow_temporary: true
#
# Save results at the end
saveresults: true
#
# output(s) type of figure
figure types:
- png
- pdf

################################################################################
##  User-related behavior
################################################################################

# if you want to see the plots at the end of the code, add your login here
# this should be the string as returned by "os.getlogin()"
user show plot:
- "dummy"

################################################################################
## Inputs related to the position within data cube timeseries
################################################################################
# Data quality flags to accept in the analysis. All other values will be 
# set to NaN
valid dq:
- 0

# define the Nth frame for 1st contact [it1].. through 4th contact
it:
- 90
- 97
- 103
- 110

# used to reject bits of domain from the analysis
# you can reject frames 0-600 with the values:
#
# reject domain:
# - 0
# - 600
#
# if you want to reject two bits of domain, 0-600 and 3000-3200
# just use
# reject domain:
# - 0
# - 600
# - 3000
# - 3200
#
# ... of course, you need an even number of values here.

# If PRISM data or saturated, you can perform a CDS between these two readouts
#cds_id:
#- 2
#- 0

################################################################################
## What do we fit in the linear model
################################################################################

# fit the dx -- along the dispersion
fit dx: true

# fit the dy -- along the cross-dispersion
fit dy: true

# fit the before - after morphological change
before-after: false

# fit the rotation in the linear reconstruction of the trace
fit rotation: true

# fit the zero point offset in the linear model for the trace
zero point offset: true

# fit the 2nd derivative in y, good to find glitches!
ddy: true

fit pca: false
n pca: 0

################################################################################
## Inputs related to handling of the data within each frame
################################################################################

pixel-level detrending: false

per pixel baseline correction: false

trace_orders:
- 1
- 2

# wavelength domain for the write light curve
#wlc domain:
#- 1.2
#- 1.6

# median of out-of-transit values for reference trace construction. 
# If set to false, then we have the median of the entire timeseries
ootmed: true

y trace offset: 0

mask order 0: true

recenter trace position: true

# out of transit polynomial level correction
transit_baseline_polyord: 2

# out-of-trace
trace_baseline_polyord: 2

# degree of the polynomial for the 1/f correction
# degree_1f_corr = 0 -> just a constant through the 256 pix spatial
# degree_1f_corr = 1 -> slope ... and so on
degree_1f_corr: 0

# set trace_width_masking to 0 to use the full image
# use in SOSSICE
trace_width_extraction: 40

# used for masking and WLC
trace_width_masking: 40

# do remove trend from out-of-transit
remove_trend: false

# define how the "white" transit depth is computed/assigned
#   "compute": Compute transit depth using median OOT relative flux from WLC, 
#              and mean in-transit relative flux from WLC
#   OR
#   provide a number: if "white" tdepth is known (from fitting the WLC for 
#                      example), one can provide the number he
tdepth: "compute"

resolution_bin: 20

spectrum_ylim_ppm:
- 6000
- 12000
'''


