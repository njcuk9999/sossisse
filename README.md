# SOSSISSE - SOSS Inspired SpectroScopic Extraction

- [Installation](https://github.com/njcuk9999/sossisse/blob/main/README.md#installation)
- [Example data set](https://github.com/njcuk9999/sossisse/blob/main/README.md#example-data-set)
- [Description of code](https://github.com/njcuk9999/sossisse/blob/main/README.md#description-of-code)


## Installation

### Step 1: Download the github repository

```
git clone git@github.com:njcuk9999/lbl.git
```

### Step 2: Install python 3.8 using conda

Using conda, create a new environment and activate it

```
conda create --name lbl-env python=3.9
```

```
conda activate lbl-env
```

### Step 3: Install sossisse

```
cd {SOSSISSE_ROOT}

pip install -U -e .
```

Note one can also use venv (instead of conda)

Note `{SOSSISSE_ROOT}` is the path to the cloned github repository (i.e. `/path/to/sossisse`)


## Example data set

Details on example data set here

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
system variable called *SOSSIOPATH* (SOSS-input-output path). With the *.zshrc* on Mac, this is done with the 
following line added to the startup profile : 

```
export SOSSIOPATH=/YourPath/Your_SOSSISSE_folder/
```
Within that folder, one should define a subfolder for the 'mode' used (here 'SOSS') and a per-target folder that 
will hold the ouputs from the analysis and includes a *rawdata* subfolder that contains the raw data.

Create the following folder structure : 

**/YourPath/Your_SOSSISSE_folder/SOSS/**

**/YourPath/Your_SOSSISSE_folder/SOSS/wasp-96/**

**/YourPath/Your_SOSSISSE_folder/SOSS/wasp-96/rawdata/**

Within the **/YourPath/Your_SOSSISSE_folder/SOSS/wasp-96/rawdata/** folder, you should place the raw data files that 
can be dowloaded from **http://www.astro.umontreal.ca/~artigau/soss/wasp-96_demodataset.tar**.

All yaml files should be placed in (of symbolically liked to) the **/YourPath/Your_SOSSISSE_folder/** folder.

### Running the code

The code is run in the python terminal with the following command : 

```
In [1] import sossisse
In [2]: sossisse.wrapper('config_wasp-96.yaml')
```
### Understanding outputs

The first step of the code being a determination of the *grey* lightcurve, we first have a photometric curve with an 
obvious transit signature. Note that the time axis is in 'frames', which correspond to the **it** parameters (4 
terms; Nth frame for 1st, 2nd, 3rd and 4th contact). In this plot, the values measured between 1st and 4th contact 
are in red while baseline (before 1st and after 4th) are coded in green.

![White light curve for the transit](resources/lightcurve.png)

![Residual plot](resources/residuals.png)


![Detrending parameter drifts](resources/drifts.png)

![Drag Racing](resources/sed_ord1.png)

![Drag Racing](resources/sed_ord2.png)



### Description of the code

