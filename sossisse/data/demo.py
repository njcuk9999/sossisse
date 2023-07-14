#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo code for running sossisse

Requires: SOSSIOPATH to be set (e.g. export SOSSIOPATH=/YourPath/Your_SOSSISSE_folder)

Within that folder, one should define a subfolder for the 'mode' used (here 'SOSS') and a per-target folder that
will hold the ouputs from the analysis and includes a *rawdata* subfolder that contains the raw data.

Create the following folder structure :

```
/YourPath/Your_SOSSISSE_folder/SOSS/
/YourPath/Your_SOSSISSE_folder/SOSS/t1b/
/YourPath/Your_SOSSISSE_folder/SOSS/t1b/rawdata/
```

where `t1b` is the name of the target (in this case `t1b` for Trappist-1b).

Also requires the demo dataset (Trappist-1b) which can be dowloaded from http://www.astro.umontreal.ca/~artigau/soss/t1b_sample.tar.

You will also need to put reference files, in the relevant subfolders, that can be downloaded from http://www.astro.umontreal.ca/~artigau/soss/ref_files.tar.
Place them for example in the following folder: `/YourPath/Your_SOSSISSE_folder/SOSS/calibrations/*.fits `

The demo.yaml should be placed in (or symbolically liked to) the `/YourPath/Your_SOSSISSE_folder/` folder.

Created on 2023-07-14 at 11:45

@author: cook
"""

import sossisse

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    sossisse.wrapper('demo.yaml')

# =============================================================================
# End of code
# =============================================================================
