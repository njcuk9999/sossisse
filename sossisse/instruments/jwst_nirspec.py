#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIRSpec loading, detector pairing and isolated trace extraction support.

Trace labels 1 and 2 identify NRS1 and NRS2; physical extraction uses order 1.

Created on 2024-08-13 at 11:29

@author: cook
"""
import copy
import os
import re
import shutil
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table

from aperocore import math as mp

from sossisse.core import io
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import misc
from sossisse.core.exceptions import SossisseConstantException
from sossisse.general.datacube import DataCube
from sossisse.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.instruments.jwst_nirspec'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define nirspec classes
# =============================================================================
class JWST_NIRSPEC(default.Instrument):
    """Load NIRSpec data and isolate detector-specific extraction contexts."""

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

    def get_detector_children(self):
        """
        Get or create independently configured NIRSpec detector instruments.

        Trace 1 labels NRS1 and trace 2 labels NRS2; both detectors use
        physical extraction order 1. Creation updates the parent's trace
        selection, prefix and output paths, creates directories, copies
        optional calibrations, and caches the child instruments.

        :return: dict or None, detector label to child instrument, or None
                 when this instrument is already a detector child
        :raises SossisseConstantException: if science grouping, selection or
                                          wavelength calibrations are invalid
        :raises OSError: if directory creation or calibration copying fails
        """
        # Keep child creation and its cache in the shared detector worker.
        return detector_children(self)

    def get_traces(self):
        """
        Get the detector labels processed by this instrument.

        These labels identify NRS1/NRS2, not separate diffraction orders.
        A parent validates all supplied detector pairs before applying its
        selection; a child already carries its isolated detector label.

        :return: list, selected detector labels in ascending order
        :raises SossisseConstantException: if parent files or selection are
                                          inconsistent
        """
        # Children retain their assigned detector without regrouping files.
        trace = getattr(self, '_detector_trace', None)
        # The marker is absent on a parent instrument before splitting.
        if trace is not None:
            # Match the list interface used by parent trace selections.
            return [trace]
        # Grouping validates paired coverage even when only one is selected.
        groups = group_files(self.params['GENERAL.FILES'],
                             self.params['GENERAL.TRACES'])
        return list(groups)

    def get_trace_pos(self, map2d=False, order_num=None, round_pos=True):
        """
        Read the physical order-1 trace on this detector's pixel grid.

        Detector labels 1 and 2 must not be passed through as diffraction
        orders: each detector's position file describes physical order 1.

        :param map2d: bool, request the parent's two-dimensional trace map
        :param order_num: int or None, accepted for interface compatibility
                          but ignored in favor of physical order 1
        :param round_pos: bool, request rounded positions from the parent
        :return: tuple, the trace positions or map and associated throughput
             array returned by the parent implementation
        :raises Exception: propagates position-file loading and validation
                           errors from the parent implementation
        """
        # Preserve the parent's output format while fixing the physical order.
        return super().get_trace_pos(map2d=map2d, order_num=1,
                                     round_pos=round_pos)

    def check_detector_cube(self):
        """
        Reject unknown detectors or a cube spanning both NIRSpec detectors.

        Detector-isolated processing is required because the two detectors
        have distinct pixel grids and calibrations. This guard checks header
        labels only; it does not validate exposure pairing or reject an empty
        file list, and does not change instrument parameters.

        :return: None
        :raises SossisseConstantException: if any detector is unknown or both
                                          NRS1 and NRS2 are present
        :raises Exception: propagates required-parameter and header-read errors
        """
        # A set distinguishes repeated segments from mixed detector inputs.
        detectors = set()
        # Inspect every file, including those after the first cube segment.
        for filename in self.params.rget('GENERAL.FILES', required=True):
            # Normalize header spelling before comparing detector identities.
            detector = str(io.load_header(filename, ext=0).get(
                'DETECTOR', '')).strip().upper()
            # Missing labels normalize to an empty string and are invalid too.
            if detector not in ('NRS1', 'NRS2'):
                # Include the offending input so the header can be inspected.
                emsg = f'Unknown DETECTOR in {filename}'
                raise exceptions.SossisseConstantException(emsg)
            # Repeated files from one detector do not create a mixed cube.
            detectors.add(detector)
        # Joining NRS1/NRS2 as temporal frames would mix unrelated pixel grids.
        if len(detectors) > 1:
            # Direct callers to the workflow that constructs isolated children.
                emsg = ('Mixed NIRSpec detector cube: use '
                    'linear_recon/spectral_extraction '
                    'or get_detector_children() for '
                    'detector-isolated processing')
                raise exceptions.SossisseConstantException(emsg)

    def load_data_with_dq(self):
        """
        Load science, error and DQ through the detector-isolated parent path.

        The parent normalizes each raw segment into bounded FITS products,
        reusing only complete matching products when requested. It updates
        data dimensions and header-sorted observation metadata in both cases.

        :return: DataCube, aligned science, error and DQ storage
        :raises SossisseConstantException: if input detector labels are unknown
                                          or span both detectors
        :raises Exception: propagates header, parameter and parent-load errors
        """
        # Reject mixed detector grids before the parent allocates or loads
        # data.
        self.check_detector_cube()
        # Leave loading, caching and any parameter updates to the parent.
        return super().load_data_with_dq()

    def get_datacube_options(self) -> dict:
        """Crop raw DataCube reads to the active detector wavelength grid."""
        datacube_kwargs = super().get_datacube_options()
        datacube_kwargs['pixel_bounds'] = self._force_pixel_grid
        return datacube_kwargs

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
        converting CDS data, as in load_data. Files are concatenated along
        the frame axis in configured order; they must all belong to one
        detector. Error and DQ arrays receive the same pixel selection as
        science data, or are generated by the CDS converter.

        :param n_slices: int, the number of output frames
        :param image_shape: list, the image shape [y, x]; x is determined
                            from the science data after pixel selection
        :param flag_cds: bool, whether the input data are in CDS format
        :param force_pixel_grid: bool, slice the last axis to the pixel grid
        :return: tuple, the science, error and DQ arrays, each shaped
                 (n_slices, image_shape[0], selected and binned width)
        :raises SossisseConstantException: if no files are configured, a
                                          detector is unknown, or inputs mix
                                          NRS1 and NRS2
        :raises Exception: propagates FITS access, pixel-grid, binning, CDS
                           conversion and incompatible array-shape errors
        """
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO(DataCube): Legacy full-array loader. The new initial loader uses
        # get_datacube_options instead; retire this API after caller migration.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # A temporal cube must never concatenate the two detector pixel grids.
        self.check_detector_cube()
        # Preserve configured file ordering when assembling the frame axis.
        input_files = self.params.rget('GENERAL.FILES', required=True)
        # An empty list cannot establish the output width from science data.
        if len(input_files) == 0:
            # Keep the configuration error distinct from FITS loading failures.
            emsg = 'No files defined in yaml. Please set the FILES parameter.'
            # Stop before any output arrays would remain uninitialized.
            raise exceptions.SossisseConstantException(emsg)
        # Track the next output frame across all file segments.
        n_slice = 0
        # Only the first loaded segment determines allocation of the cubes.
        for ifile, filename in enumerate(input_files):
            # Keep science, error and DQ reads within the FITS resource
            # lifetime.
            with fits.open(filename) as hdul:
                # Extension 1 supplies the science array and its spatial width.
                tmp_data = hdul[1].data
                # Apply the wavelength-grid crop before spatial/temporal
                # binning.
                if force_pixel_grid:
                    # Reuse these exact column bounds for error and DQ below.
                    pixel_start, pixel_end = self._force_pixel_grid(filename)
                    # Ellipsis preserves all leading frame and row dimensions.
                    tmp_data = tmp_data[..., pixel_start:pixel_end]
                # Flux binning defines the science shape used for allocation.
                tmp_data = self.bin_cube(tmp_data, bin_type='Flux')
                # CDS inputs derive uncertainty and flags from the science
                # data.
                if flag_cds:
                    # Conversion returns all three aligned products together.
                    tmp_data, tmp_err, tmp_dq = self.from_cds(tmp_data)
                # Non-CDS products carry independent error and DQ extensions.
                else:
                    # Read uncertainty before applying its own binning rule.
                    tmp_err = hdul[2].data
                    # DQ flags must be binned as flags rather than flux values.
                    tmp_dq = hdul[3].data
                    # Keep all three products on the selected detector columns.
                    if force_pixel_grid:
                        # The uncertainty crop must match the science crop.
                        tmp_err = tmp_err[..., pixel_start:pixel_end]
                        # Flag locations follow the same last-axis selection.
                        tmp_dq = tmp_dq[..., pixel_start:pixel_end]
                    # Propagate errors using the dedicated uncertainty rule.
                    tmp_err = self.bin_cube(tmp_err, bin_type='Error')
                    # Combine flags with the dedicated DQ binning rule.
                    tmp_dq = self.bin_cube(tmp_dq, bin_type='DQ')
            loaded_extensions = [1]
            if not flag_cds:
                loaded_extensions.extend([2, 3])
            io.log_fits_load(filename, ext=loaded_extensions)
            # Derive width from actual selected data, not the requested x size.
            if ifile == 0:
                # The caller supplies frame count and row count for allocation.
                cube = np.zeros([n_slices, image_shape[0], tmp_data.shape[-1]])
                # Match science allocation shape and dtype for uncertainties.
                err = np.zeros_like(cube)
                # Retain the existing shared allocation dtype for DQ output.
                dq = np.zeros_like(cube)
            # Each segment contributes its post-binning number of frames.
            end = n_slice + tmp_data.shape[0]
            # Place corresponding science frames in a contiguous output span.
            cube[n_slice:end, :, :] = tmp_data
            # Uncertainties use the same frame interval as science.
            err[n_slice:end, :, :] = tmp_err
            # DQ remains frame-aligned with both numerical products.
            dq[n_slice:end, :, :] = tmp_dq
            # Advance by the actual segment length rather than input metadata.
            n_slice = end
        # Return the aligned products in the established loader order.
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

    def get_trace_positions(self, med: Optional[np.ndarray] = None,
                            cube: Optional[DataCube] = None,
                            log: bool = True) -> np.ndarray:
        """
        Get the trace positions in a combined map
        (True where the trace is, False otherwise)

        Without POS_FILE, fit and write a trace from the median image; if
        only the DataCube is given, its exact per-pixel nanmedian over all
        active integrations is computed with bounded tiled reads.

        :param med: optional 2D median image, preferred over cube
        :param cube: optional active DataCube; never materialized
        :param log: bool, if True print progress
        :return: np.ndarray, the trace position map
        :raises SossisseException: if POS_FILE is absent and neither med nor
                                   cube is supplied
        :raises MemoryError: if the exact median exceeds the math budget
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
                # Exact per-pixel median of all integrations in tiles.
                med = cube.nanmedian('FLUX', axis=0)
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
            trace_fit_args = [index[is_flux], s1[is_flux] / s2[is_flux]]
            trace_fit_kwargs = dict(degree=2, nsigcut=4)
            trace_fit = mp.robust_polyfit(*trace_fit_args, **trace_fit_kwargs)
            tfit, tmask = trace_fit
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
    """Configure NIRSpec BOTS prism processing on isolated detector grids."""

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
    """Configure the common NIRSpec BOTS grating instrument settings."""

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
    """Configure the G395 grating family without changing detector identity."""

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
    """Select the medium-resolution G395M instrument mode."""

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
    """Select the high-resolution G395H instrument mode."""

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
    """Configure the G235 grating family without changing detector identity."""

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
    """Select the medium-resolution G235M instrument mode."""

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
    """Select the high-resolution G235H instrument mode."""

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
    """Configure the G140 grating family without changing detector identity."""

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
    """Select the medium-resolution G140M instrument mode."""

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
    """Select the high-resolution G140H instrument mode."""

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
# Worker functions
# =============================================================================
def group_files(filenames, traces=None):
    """
    Validate NIRSpec exposure coverage and group files by detector label.

    Trace 1 means NRS1 and trace 2 means NRS2, not physical extraction
    orders; each detector is extracted as order 1. Pairing compares exposure
    metadata and integration numbers, not detector-specific timestamps, so
    corresponding detectors may have different integration times.

    When both detectors are supplied, complete pairing is required before
    applying the requested selection. A selection cannot hide an incomplete
    pair. Files are sorted by exposure identity and first integration number.
    Input sequences and FITS files are not modified.

    :param filenames: iterable, science FITS paths to inspect
    :param traces: list or None, distinct integer detector labels selected
                   from 1 and 2; None selects all detectors present
    :return: dict, ascending detector labels mapped to ordered filename lists
    :raises SossisseConstantException: if detectors, exposure identities,
                                      integration coverage, pairing or trace
                                      selection are missing or inconsistent
    :raises Exception: propagates FITS access and missing SCI extension errors
    """
    # Separate detector namespaces allow matching keys across NRS1 and NRS2.
    groups = dict()
    groups[1] = dict()
    groups[2] = dict()
    # Validate every supplied file before narrowing to requested detectors.
    for filename in filenames:
        # Read headers, integration identities and SCI shape from one FITS
        # file.
        with fits.open(filename) as hdus:
            # Primary metadata identifies the exposure independently of timing.
            header = hdus[0].header
            # Normalize detector labels while rejecting absent labels below.
            detector = str(header.get('DETECTOR', '')).strip().upper()
            # Only the two NIRSpec detector identities have trace labels here.
            if detector not in ('NRS1', 'NRS2'):
                # Name the input whose detector cannot be assigned.
                emsg = f'Unknown DETECTOR in {filename}'
                raise SossisseConstantException(emsg)
            # The last identity element is the segment within this exposure.
            identity = tuple(header.get(key) for key in (
                'PROGRAM', 'OBSERVTN', 'VISIT', 'VISITGRP', 'SEQ_ID',
                'ACT_ID', 'EXPOSURE', 'EXSEGNUM'))
            # EXPOSURE is required even when other identity fields are absent.
            if header.get('EXPOSURE') is None:
                # Without an exposure identity, pairing would be ambiguous.
                emsg = f'Missing exposure identity in {filename}'
                raise SossisseConstantException(emsg)
            # Header bounds are inclusive integration numbers, not timestamps.
            start = header.get('INTSTART')
            # Keep an absent bound distinct from a valid integration number.
            stop = header.get('INTEND')
            # Prefer explicit per-frame integration identities when available.
            if 'INT_TIMES' in hdus:
                if 'integration_number' in hdus['INT_TIMES'].columns.names:
                    # Detector timestamps may differ; compare identities only.
                    int_times = hdus['INT_TIMES'].data
                    integrations = tuple(int_times['integration_number'])
            # Older products may provide coverage only through primary headers.
            elif start is not None and stop is not None:
                # Include INTEND when reconstructing the integration sequence.
                integrations = tuple(range(start, stop + 1))
            # Neither timestamps nor array length can replace frame identities.
            else:
                # Require explicit coverage for reliable detector pairing.
                emsg = f'Missing integration identities in {filename}'
                raise SossisseConstantException(emsg)
            # Empty segments and repeated integration IDs cannot be paired.
            invalid_integrations = not integrations
            if not invalid_integrations:
                invalid_integrations = (
                    len(set(integrations)) != len(integrations))
            if invalid_integrations:
                # Report malformed identities before checking contiguity.
                emsg = f'Invalid integrations in {filename}'
                raise SossisseConstantException(emsg)
            # Require ascending, gap-free coverage within each science segment.
            expected_integrations = tuple(range(integrations[0],
                                                integrations[-1] + 1))
            if integrations != expected_integrations:
                # Gaps would otherwise silently omit science frames.
                emsg = f'Incomplete integrations in {filename}'
                raise SossisseConstantException(emsg)
            # Explicit table identities must agree with any complete header
            # span.
            if start is not None and stop is not None:
                # Compare the full sequence, not merely its first/last values.
                if integrations != tuple(range(start, stop + 1)):
                    # Reject disagreement between table and primary metadata.
                    emsg = f'Incomplete integrations in {filename}'
                    raise SossisseConstantException(emsg)
            # Each SCI frame must have exactly one validated integration ID.
            if hdus['SCI'].shape[0] != len(integrations):
                # Metadata-only matches are insufficient for paired data cubes.
                emsg = f'Integration coverage mismatch in {filename}'
                raise SossisseConstantException(emsg)
            # Detector suffixes become processing labels, not physical orders.
            trace = int(detector[-1])
            # Keys contain exposure fields, segment ID, then integration tuple.
            key = identity + (integrations,)
            # An identical key on one detector would duplicate science frames.
            if key in groups[trace]:
                # The same key on the other detector is instead a valid pair.
                emsg = f'Duplicate detector exposure: {filename}'
                raise SossisseConstantException(emsg)
            # Check neighboring segments already registered on this detector.
            for existing in groups[trace]:
                # existing[:-2] drops segment and integrations; identity[:-1]
                # drops segment, leaving the same exposure-level metadata.
                if existing[:-2] != identity[:-1]:
                    # Integrations may repeat in unrelated exposures.
                    continue
                # existing[-2] is EXSEGNUM; a known segment cannot occur twice.
                if identity[-1] is not None and existing[-2] == identity[-1]:
                    # Reject reused segments even if their coverage differs.
                    emsg = f'Duplicate detector segment: {filename}'
                    raise SossisseConstantException(emsg)
                # existing[-1] holds integration IDs, which must not overlap.
                if set(existing[-1]).intersection(integrations):
                    # Prevent double-counting frames across exposure segments.
                    emsg = f'Overlapping integrations: {filename}'
                    raise SossisseConstantException(emsg)
            # Store plain paths after all checks for this segment succeed.
            groups[trace][key] = str(filename)
    # Preserve the ascending detector order established by the group mapping.
    present = [trace for trace in groups if groups[trace]]
    # No detector selection is meaningful without at least one science input.
    if not present:
        # Keep this distinct from a requested detector missing from valid data.
        emsg = 'No NIRSpec files supplied'
        raise SossisseConstantException(emsg)
    # Validate both supplied detectors before applying a one-detector
    # selection.
    if len(present) == 2:
        # Compare identities without requiring matching detector timestamps.
        if groups[1].keys() != groups[2].keys():
            emsg = 'Incomplete NRS1/NRS2 exposure pairs'
            raise SossisseConstantException(emsg)
    # A null selection means all available detectors, not always both
    # detectors.
    selected = present if traces is None else traces
    # Require an explicit nonempty list of unique integers; bool is not a
    # label.
    invalid_selection = not isinstance(selected, list) or not selected
    # Only iterate a nonempty list; bool must not be accepted as int here.
    if not invalid_selection:
        invalid_selection = any(type(trace) is not int or trace not in (1, 2)
                                for trace in selected)
    # Hash only labels already validated as integers, preserving error order.
    if not invalid_selection:
        invalid_selection = len(set(selected)) != len(selected)
    if invalid_selection:
        # Reject invalid selections without coercing their values or container.
        emsg = 'GENERAL.TRACES must be [1], [2], [1, 2] or null'
        raise SossisseConstantException(emsg)
    # Valid detector labels must also be represented by supplied science files.
    if any(trace not in present for trace in selected):
        # A requested but absent detector cannot produce an isolated child.
        emsg = 'Missing selected NIRSpec detector'
        raise SossisseConstantException(emsg)
    # Sort exposure fields via strings to tolerate missing metadata, then sort
    # segments by first integration; item[:-2] excludes segment and coverage.
    selected_groups = dict()
    # Preserve ascending detector labels and the original exposure sort key.
    for trace in sorted(selected):
        exposure_keys = sorted(groups[trace], key=lambda item: (
            tuple(str(value or '') for value in item[:-2]), item[-1][0]))
        selected_groups[trace] = [groups[trace][key] for key in exposure_keys]
    return selected_groups


def calibration_detector(filename, default_trace=None):
    """
    Identify a calibration's detector from its header or filename.

    A recognized primary-header DETECTOR takes precedence over filename
    tokens. If the header is absent or unreadable, isolated nrs1/nrs2
    filename tokens identify the detector. The fallback applies only when
    neither source provides an identity, not when a token is ambiguous.

    :param filename: str, calibration path whose header and basename to inspect
    :param default_trace: int or None, fallback label supplied by the caller
                          for an otherwise unidentified calibration
    :return: int, detector label 1 for NRS1 or 2 for NRS2, or the supplied
                  default_trace when fallback is permitted
    :raises SossisseConstantException: if a header detector is unknown or
                                      filename tokens cannot identify one
                                      detector and no fallback applies
    """
    # A FITS detector declaration is authoritative when it can be read.
    try:
        # Missing DETECTOR and unreadable headers share filename fallback
        # below.
        detector = fits.getheader(filename).get('DETECTOR')
    # Text calibrations need not have FITS headers to be usable.
    except (OSError, ValueError):
        # Retain the distinction between no identity and an invalid
        # declaration.
        detector = None
    # Do not override an explicit detector with a filename or caller fallback.
    if detector is not None:
        # Header case does not change the detector's identity.
        detector = str(detector).upper()
        # Known detector suffixes map to the processing trace labels.
        if detector in ('NRS1', 'NRS2'):
            # Both detector labels still use physical extraction order 1.
            return int(detector[-1])
        # An unknown declaration must not silently select a different detector.
        emsg = f'Unknown calibration DETECTOR: {filename}'
        raise SossisseConstantException(emsg)
    # Token boundaries avoid matching nrs1 inside unrelated alphanumeric names;
    # a set treats repeated references to the same detector as one identity.
    matches = set(re.findall(r'(?<![a-z0-9])nrs([12])(?![a-z0-9])',
                             os.path.basename(filename).lower()))
    # The caller may allow an unlabeled scalar single-detector calibration.
    if not matches and default_trace is not None:
        # Preserve the caller's supplied label without further coercion.
        return default_trace
    # Zero tokens without fallback and two conflicting tokens are both invalid.
    if len(matches) != 1:
        # Never use the default to resolve contradictory detector tokens.
        emsg = f'Ambiguous calibration detector: {filename}'
        raise SossisseConstantException(emsg)
    # The only remaining token identifies the calibration's detector.
    return int(matches.pop())


def calibration_files(params, groups, available=None):
    """
    Resolve wavelength calibrations and ensure selected detectors are covered.

    A scalar wavelength filename may omit its detector identity only when
    exactly one detector is selected. List entries always require identities.
    Extra calibrations are retained, but must belong to available science
    detectors when that mapping is supplied. Parameters are read, not changed.

    :param params: mapping, parameters containing GENERAL.WAVE_FILE and
                   PATHS.CALIBPATH; wavelength filenames may be scalar or list
    :param groups: mapping, selected detector labels to science filename lists
    :param available: mapping or None, all science detector labels available
    :return: dict, detector labels mapped to resolved wavelength-file paths
    :raises SossisseConstantException: if filenames are invalid or missing,
                                      detector identities conflict, or a
                                      selected detector lacks a calibration
    """
    # Preserve scalar versus list semantics for the single-detector fallback.
    value = params['GENERAL.WAVE_FILE']
    # Iterate uniformly without changing the configured wavelength value.
    filenames = value if isinstance(value, list) else [value]
    # One detector may have only one resolved wavelength solution.
    solutions = dict()
    # Validate all supplied solutions, including unselected detector entries.
    for filename in filenames:
        # Calibration resolution expects filesystem strings, not null entries.
        if not isinstance(filename, str):
            # Reject malformed configuration before attempting path operations.
            emsg = 'GENERAL.WAVE_FILE must provide detector calibrations'
            raise SossisseConstantException(emsg)
        # Absolute filenames remain absolute under os.path.join semantics.
        path = os.path.join(params['PATHS.CALIBPATH'], filename)
        # Require a real file before trying header or basename identification.
        if not os.path.isfile(path):
            # Include the resolved path rather than only the configured name.
            emsg = f'Missing wavelength calibration: {path}'
            raise SossisseConstantException(emsg)
        # Only a scalar with one selected detector may be implicitly labeled.
        default_trace = None
        if len(groups) == 1 and isinstance(value, str):
            default_trace = next(iter(groups))
        # Explicit header/name identities still take precedence over fallback.
        trace = calibration_detector(path, default_trace=default_trace)
        # Multiple solutions for one detector would make child setup ambiguous.
        if trace in solutions:
            # Keep the detector label visible in the configuration error.
            emsg = f'Duplicate NRS{trace} wavelength calibration'
            raise SossisseConstantException(emsg)
        # Check against all supplied science, not only the selected subset.
        if available is not None and trace not in available:
            # A calibration for an absent science detector is a mismatch.
            emsg = f'Mismatched wavelength calibration: {path}'
            raise SossisseConstantException(emsg)
        # Keep the resolved path for direct assignment to a child instrument.
        solutions[trace] = path
    # Extra valid solutions do not compensate for a missing selected detector.
    if any(trace not in solutions for trace in groups):
        # Every child must receive its own wavelength calibration.
        emsg = 'Missing selected detector wavelength calibration'
        raise SossisseConstantException(emsg)
    # Return all validated solutions without mutating the parent parameters.
    return solutions


def detector_children(inst):
    """
    Build and cache detector-isolated instruments for a NIRSpec parent.

    Labels 1/2 mean NRS1/NRS2, while extraction remains physical order 1.
    Validate all science pairs and calibrations before changing the parent.
    Creation normalizes the parent's trace selection, appends a selection
    suffix to its generated filenames. Children retain the configured shared
    directories while receiving their own science files, wavelength solution,
    and position-file path.

    Non-null BKGFILE and FLATFILE inputs are copied into each child's
    temporary directory under detector-specific basenames, and their
    parameters are rewritten to those copies.
    POS_FILE is always reset to a detector-specific temporary path, not
    copied, even when an explicit position file was supplied. Directories
    and copied files are filesystem side effects; no rollback is performed
    if later setup fails. A successful result is cached on the parent and
    returned unchanged on subsequent calls.

    :param inst: JWST_NIRSPEC, parent instrument or an already isolated child
    :return: dict or None, detector labels mapped to child instruments, or
             None when inst already has a detector marker
    :raises SossisseConstantException: if science coverage, detector selection
                                      or wavelength calibrations are invalid
    :raises OSError: if creating directories or copying calibrations fails
    :raises Exception: propagates parameter access and child-construction
                       errors
    """
    # A detector child must not recursively split itself into more children.
    if getattr(inst, '_detector_trace', None) is not None:
        # None signals that this instrument is already the processing unit.
        return None
    # Reuse prior children without appending suffixes or copying files again.
    if getattr(inst, '_detector_children', None) is not None:
        # Return the cached objects, including any state accumulated by them.
        return inst._detector_children
    # Pair validation occurs within grouping before detector selection.
    groups = group_files(inst.params['GENERAL.FILES'],
                         inst.params['GENERAL.TRACES'])
    # Retain all science detectors when checking extra wavelength calibrations.
    available = group_files(inst.params['GENERAL.FILES'])
    # Finish calibration validation before starting parent/path mutations.
    solutions = calibration_files(inst.params, groups, available)
    # Replace a null or user-ordered selection with sorted validated labels.
    inst.params['GENERAL.TRACES'] = list(groups)
    # Distinguish combined final products for each selected detector set.
    selected_detectors = '_'.join(f'nrs{trace}' for trace in groups)
    inst._trace_suffix = f'_{selected_detectors}'
    # Publish the cache only after all selected children have been constructed.
    children = dict()
    # Runtime setup creates this directory; direct callers may not have yet.
    os.makedirs(inst.params['PATHS.TEMP_PATH'], exist_ok=True)
    # Keep shared directory paths and the common input prefix unchanged.
    # Each child owns exactly one detector's science and calibration context.
    for trace, filenames in groups.items():
        # Prevent child parameter mutations from leaking into siblings/parent.
        params = copy.deepcopy(inst.params)
        # Retain validated exposure ordering within this detector only.
        params['GENERAL.FILES'] = filenames
        # Detector identity remains a one-element trace selection for
        # workflows.
        params['GENERAL.TRACES'] = [trace]
        # Replace any parent scalar/list with this detector's resolved
        # solution.
        params['GENERAL.WAVE_FILE'] = solutions[trace]
        # Optional image calibrations are copied; position files are reset.
        for key in ('BKGFILE', 'FLATFILE', 'POS_FILE'):
            # Read the configured source before replacing any child path.
            filename = params[f'GENERAL.{key}']
            # Retain the automatic-position flag calculation; it is currently
            # unused and does not govern the unconditional POS_FILE reset.
            automatic_pos = (key == 'POS_FILE'
                             and params.get('__AUTO_POS_FILE__', False))
            # Background and flat copies use detector-specific basenames.
            if filename is not None and key != 'POS_FILE':
                # Insert the detector identity before the source extension.
                basename, extension = os.path.splitext(
                    os.path.basename(filename))
                tagged_basename = f'{basename}_nrs{trace}{extension}'
                # Retain the configured temporary directory for the copy.
                dest = os.path.join(params['PATHS.TEMP_PATH'],
                                    tagged_basename)
                # Each detector now writes to a distinct file in this folder.
                shutil.copyfile(filename, dest)
                # Subsequent child processing refers to the isolated copy.
                params[f'GENERAL.{key}'] = dest
            # Give each generated position map its own shared-directory name.
            elif key == 'POS_FILE':
                # The trace table is created later by the detector child.
                params['GENERAL.POS_FILE'] = os.path.join(
                    params['PATHS.TEMP_PATH'], f'pos_file_nrs{trace}.fits')
        # Preserve the concrete instrument mode and its constructor overrides.
        child = type(inst)(params)
        # Mark the child so lookup, naming, and splitting identify its detector.
        child._detector_trace = trace
        # Add detector identity to products without creating more directories.
        child._trace_suffix = f'_nrs{trace}'
        # Preserve ascending detector order in the result mapping.
        children[trace] = child
    # Cache only the complete mapping; later calls reuse it without setup.
    inst._detector_children = children
    # The parent retains its original paths and common input prefix.
    return children


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
