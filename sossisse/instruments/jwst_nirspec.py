#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2024-08-13 at 11:29

@author: cook
"""
import os
from typing import List, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table

from aperocore import math as mp

from sossisse.core import io
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.jwst_nirspec'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
class JWST_NIRSPEC(default.Instrument):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC'
        # set up the instrument
        self.param_override()

    def param_override(self):
        """
        Override the parameters for this instrument
        :return:
        """
        # run the super version first, then override after
        super().param_override()
        # we do not mask order zero for NIRSPEC PRISM
        self.params['WLC.GENERAL.MASK_ORDER_ZERO'] = False
        # for NIRSPEC PRISM we shouldn't have recenter trace position
        self.params['WLC.GENERAL.RECENTER_TRACE_POSITION'] = False

    def load_data(self, filename: str, ext: int = None, extname: str = None,
                  force_pixel_grid: bool = False):
        """
        Load data, optionally selecting the NIRSpec pixel grid.

        :param filename: str, the filename to load
        :param ext: int, the extension number to load
        :param extname: str, the extension name to load
        :param force_pixel_grid: bool, slice the last axis to the pixel grid

        :return: data, the loaded data
        """
        data = io.load_fits(filename, ext, extname)
        if force_pixel_grid:
            pixel_start, pixel_end = self._force_pixel_grid(filename)
            data = data[..., pixel_start:pixel_end]
        return data

    def load_cube(self, n_slices: int, image_shape: List[int],
                  flag_cds: bool, force_pixel_grid: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load science, error and DQ cubes on the NIRSpec pixel grid.

        When requested, slice the last axis before binning or
        converting CDS data, as in load_data.

        :param n_slices: int, the number of output frames
        :param image_shape: list, the image shape [y, x]; x is determined
                            from the science data after pixel selection
        :param flag_cds: bool, whether the input data are in CDS format
        :param force_pixel_grid: bool, slice the last axis to the pixel grid
        :return: tuple, the science, error and DQ cubes
        """
        input_files = self.params.rget('GENERAL.FILES', required=True)
        if len(input_files) == 0:
            emsg = 'No files defined in yaml. Please set the FILES parameter.'
            raise exceptions.SossisseConstantException(emsg)
        n_slice = 0
        for ifile, filename in enumerate(input_files):
            with fits.open(filename) as hdul:
                tmp_data = hdul[1].data
                if force_pixel_grid:
                    pixel_start, pixel_end = self._force_pixel_grid(filename)
                    tmp_data = tmp_data[..., pixel_start:pixel_end]
                tmp_data = self.bin_cube(tmp_data, bin_type='Flux')
                if flag_cds:
                    tmp_data, tmp_err, tmp_dq = self.from_cds(tmp_data)
                else:
                    tmp_err = hdul[2].data
                    tmp_dq = hdul[3].data
                    if force_pixel_grid:
                        tmp_err = tmp_err[..., pixel_start:pixel_end]
                        tmp_dq = tmp_dq[..., pixel_start:pixel_end]
                    tmp_err = self.bin_cube(tmp_err, bin_type='Error')
                    tmp_dq = self.bin_cube(tmp_dq, bin_type='DQ')
            if ifile == 0:
                cube = np.zeros([n_slices, image_shape[0], tmp_data.shape[-1]])
                err = np.zeros_like(cube)
                dq = np.zeros_like(cube)
            end = n_slice + tmp_data.shape[0]
            cube[n_slice:end, :, :] = tmp_data
            err[n_slice:end, :, :] = tmp_err
            dq[n_slice:end, :, :] = tmp_dq
            n_slice = end
        return cube, err, dq

    def _force_pixel_grid(self, filename: str) -> Tuple[int, int]:
        """
        Force the pixel grid for NIRSpec based on the detector

        For NRS1 we have to cut down the data to the
        last N pixels (based on the wavegrid) - for NRS2 we keep all pixels

        :param filename: str, the filename to load
        :return: Tuple[int, int], the pixel start and end indices
        """
        # deal with NRS1 vs NRS2 - for NRS1 we have to cut down the data to the
        # last N pixels (based on the wavegrid) - for NRS2 we keep all pixels
        hdr = io.load_header(filename, ext=0)
        # two keys are required from the header for this task:
        #     DETECTOR and SUBSIZE1
        detector = hdr.get('DETECTOR', None)
        nbxpix = hdr.get('SUBSIZE1', None)
        # Deal with no detector keyword in header
        if detector is None:
            emsg = ('No DETECTOR keyword found in header.'
                    'For {0} this key is required in input file={1}')
            eargs = [self.params['INPUTS.INSTRUMENTMODE'], filename]
            raise exceptions.SossisseException(emsg.format(*eargs))
        # Deal with no SUBSIZE1 keyword in header
        if nbxpix is None:
            emsg = ('No SUBSIZE1 keyword found in header.'
                    'For {0} this key is required in input file={1}')
            eargs = [self.params['INPUTS.INSTRUMENTMODE'], filename]
            raise exceptions.SossisseException(emsg.format(*eargs))
        # get the wave grid for the instrument mode
        wavegrid = self.get_wavegrid()
        # calculate where we should start and end based on the wavegrid shape
        pixel_start = nbxpix-wavegrid.shape[0]
        pixel_end = nbxpix
        # return the start and end pixels
        return pixel_start, pixel_end


    def optimize_trace_mask(self, log: bool = True):
        """
        Optimize the trace position and save a new pos mask file

        Currently does nothing for NIRSpec

        :return: None
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.optimize_trace_mask()'
        return

    def get_trace_positions(self, med: np.ndarray = None,
                            cube: np.ndarray = None,  log: bool = True
                            ) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        :return: np.ndarray, the trace position map 
        """
        # set function name
        func_name = f'{__NAME__}.{self.name}.get_trace_positions()'

        gen_params = self.params.get('GENERAL')
        wlc_gen_params = self.params.get('WLC.GENERAL')
        # deal with no trace pos file for prism
        if not os.path.exists(gen_params['POS_FILE']):
            xoffset = wlc_gen_params.rget('X_TRACE_OFFSET', required=True,
                                         func=func_name)
            yoffset = wlc_gen_params.rget('Y_TRACE_OFFSET', required=True,
                                         func=func_name)
            # log that we don't have a POS_FILE and are creating one
            if log:
                msg = 'No POS_FILE defined for mode={0} - creating trace map'
                margs = [self.name]
                misc.printc(msg.format(*margs), msg_type='warning')
            # get the wave grid from the parameters
            xpix, wavegrid = self.get_wavegrid(return_xpix=True)
            # -----------------------------------------------------------------
            # deal with having a median supplied (and no temp clean nan)
            if med is not None:
                med = np.array(med)
            # deal with having a cube supplied (and no temp clean nan)
            elif cube is not None:
                med = np.nanmedian(cube, axis=0)
            # else raise exception
            else:
                emsg = 'No med/cube provided'
                emsg += '\n function {0} must set med or cube'
                eargs = [func_name]
                raise exceptions.SossisseException(emsg.format(*eargs))
            # -----------------------------------------------------------------
            # get the indices of the median image
            pixy, pixx = np.indices(med.shape)
            # calculate the sum and running sum of the median image
            s1 = np.nansum(pixy * med, axis=0)
            s2 = np.nansum(med, axis=0)
            # get the indices along the x direction
            index = np.arange(med.shape[1])
            # make a mask for values which will be considered as part of
            #    the trace
            is_flux = s2 > np.nanpercentile(s2, 95) / 5
            # fit the indices robustly (with a quadratic
            tfit, tmask = mp.robust_polyfit(index[is_flux],
                                            s1[is_flux] / s2[is_flux],
                                            degree=2, nsigcut=4)
            # push this into a trace map table
            tracetable = Table()
            tracetable['X'] = index
            tracetable['Y'] = np.polyval(tfit + yoffset, index + xoffset)
            tracetable['WAVELENGTH'] = np.full(len(index), np.nan)
            # push in the wave sol
            valid_wave = np.array(xpix, dtype=int)
            tracetable['WAVELENGTH'][valid_wave] = wavegrid[valid_wave]
            # print that we are writing pos file
            if log:
                msg = 'Writing POS_FILE={0}'
                margs = [gen_params['POS_FILE']]
                misc.printc(msg.format(*margs), msg_type='info')
            # write trace file
            tracetable.write(gen_params['POS_FILE'], overwrite=True)
        # ---------------------------------------------------------------------
        # get the trace positions from the white light curve
        trace_mask, _ = self.get_trace_pos(map2d=True, order_num=1)
        # return the trace positions
        return trace_mask


class JWST_NIRSPEC_PRISM(JWST_NIRSPEC):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_PRISM, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.PRISM'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_GRATING(JWST_NIRSPEC):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_GRATING, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.GRATING'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G395(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G395, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G395'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G395M(JWST_NIRSPEC_G395):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G395M, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G395M'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G395H(JWST_NIRSPEC_G395):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G395H, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G395H'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G235(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G235, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G235'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G235M(JWST_NIRSPEC_G235):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G235M, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G235M'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G235H(JWST_NIRSPEC_G235):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G235H, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G235H'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G140(JWST_NIRSPEC_GRATING):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G140, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G140'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G140M(JWST_NIRSPEC_G140):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G140M, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G140M'
        # set up the instrument
        self.param_override()


class JWST_NIRSPEC_G140H(JWST_NIRSPEC_G140):
    def __init__(self, params):
        """
        Construct the instrument class

        :param params: dict, the parameters for the instrument
        """
        # get the default parameters
        super(JWST_NIRSPEC_G140H, self).__init__(params)
        # set name
        self.name = 'JWST.NIRSPEC.BOTS.G140H'
        # set up the instrument
        self.param_override()



# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
