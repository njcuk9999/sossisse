#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.general.py

General functionality in here

Created on 2022-09-20

@author: cook
"""
import os
from typing import Optional, cast

import numpy as np
from astropy.table import Table
from astropy.table import vstack

from aperocore.core import drs_log
from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.core import misc
from sossisse.general.datacube import DataCube
from sossisse.general.reconstruction import ReconstructionProducts
from sossisse.general import plots
from sossisse.instruments import Instrument

WLOG = drs_log.wlog

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.general'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define main functions
# =============================================================================
def _processing_splash_message(inst: Instrument) -> Optional[str]:
    """Describe the active detector and jump chunk for a processing splash."""
    messages = []
    detector = getattr(inst, '_detector_trace', None)
    if detector is not None:
        messages.append(f'Detector {detector}')
    jump_slice = getattr(inst, '_jump_slice', None)
    if getattr(inst, '_jump_suffix', '') and jump_slice is not None:
        start, stop = jump_slice
        messages.append(f'Jump chunk {start}-{stop - 1}')
    return '  '.join(messages) or None


def linear_recon_init(inst):
    # set the function name
    func_name = f'{__NAME__}.linear_recon'
    objname = inst.params.rget('INPUTS.OBJECTNAME', required=True,
                               func=func_name)
    # print the splash
    misc.sossart()
    # print the white light curve splash
    print(misc.art('Linear Recon ' + objname, 'blue', 'CYAN',
                   message=_processing_splash_message(inst)))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # get the stabiblity table file name
    wlc_ltbl_file = inst.get_variable('WLC_LTBL_FILE', func_name)
    # return if we have the soss_stablity file
    if os.path.exists(wlc_ltbl_file):
        use_temporary = inst.params.rget('GENERAL.USE_TEMPORARY',
                                         required=True, func=func_name)
        if use_temporary:
            try:
                products, _ = inst.load_wlc_results()
            except (OSError, ValueError, KeyError, TypeError):
                return False
            inst.save_input_residuals(products, overwrite=False)
            msg = 'File {0} exists we skip linear reconstruction step'
            WLOG(inst.params, 'info', msg.format(wlc_ltbl_file))
            # A fresh instrument does not have the fitted-column metadata in memory.
            # Restore it from the current model configuration before merging/plotting.
            inst.set_linear_output_metadata()
            return True
    # if we've got here return false
    return False



def linear_recon(inst: Instrument) -> Instrument:
    """
    Linear reconstruction functionality

    :param inst: Instrument, the instrument object
    :return: Instrument, updated instrument object
    """
    # Each detector has an independent instrument; the default is inst itself.
    detectors = inst.get_processing_detectors()
    # Validate jumps against each detector's frames, not the combined files.
    # With no jumps, each detector gets one chunk with unchanged filenames.
    jump_contexts = {key: get_jump_chunks(detector_inst)
                     for key, detector_inst in detectors.items()}
    # Keep filenames by detector so only time-adjacent chunks are merged.
    wlc_files = {key: [] for key in detectors}
    for detector, detector_inst in detectors.items():
        chunks, jump_ints = jump_contexts[detector]
        for chunk in chunks:
            # Set the integration slice and product names for this run.
            detector_inst.set_jump_context(chunk['start'], chunk['stop'],
                                           chunk['suffix'], jump_ints)
            # run the linear reconstruction for this detector and chunk
            detector_inst = linear_recon_chunk(detector_inst)
            # push the WLC product filenames into the list for this detector
            wlc_files[detector].append(get_wlc_files(detector_inst))
        detectors[detector] = detector_inst
    # Merge jump products within each detector and update the main instrument.
    return merge_linear_recon(inst, detectors, wlc_files, jump_ints)



def spectral_extraction(inst: Instrument) -> Instrument:
    """
    Spectral extraction functionality

    :param inst: Instrument, the instrument object
    :return: Instrument, updated instrument object
    """
    # Each detector has an independent instrument; the default is inst itself.
    detectors = inst.get_processing_detectors()
    # Use the same detector-specific jump slices as linear reconstruction.
    # With no jumps, each detector gets one chunk with unchanged filenames.
    jump_contexts = {key: get_jump_chunks(detector_inst)
                     for key, detector_inst in detectors.items()}
    # Store results separately so detector wavelength grids are never joined.
    storages = {key: [] for key in detectors}
    for detector, detector_inst in detectors.items():
        chunks, jump_ints = jump_contexts[detector]
        for chunk in chunks:
            # Select the matching reconstruction products and frame interval.
            detector_inst.set_jump_context(chunk['start'], chunk['stop'],
                                           chunk['suffix'], jump_ints)
            # Run the spectral extraction for this detector and chunk.
            chunk_storage = spectral_extraction_chunk(detector_inst)
            # Store the per-chunk results for later merging.
            storages[detector].append(chunk_storage)
    # Merge time chunks within each detector, then publish the final products.
    return merge_spectral_extraction(inst, detectors, storages, jump_ints)



# =============================================================================
# Define chunk functions
# =============================================================================
def linear_recon_chunk(inst: Instrument) -> Instrument:
    """
    Linear reconstruction functionality

    :param inst: Instrument, the instrument object
    :return:
    """
    # =========================================================================
    # linear recon initialization (hidden for notebook use)
    # =========================================================================
    if linear_recon_init(inst):
        return inst
    
    # =========================================================================
    # Load the image, error and data quality images
    # =========================================================================
    datacube = inst.load_data_with_dq()

    # =========================================================================
    # Apply the flat field
    # =========================================================================
    datacube = inst.apply_flat_field(datacube)

    # =========================================================================
    # Keep only certain Data Quality flags
    # =========================================================================
    datacube = inst.apply_dq(datacube)

    # =========================================================================
    # remove the background
    # =========================================================================
    datacube = inst.remove_background(datacube)

    # =========================================================================
    # low pass the data
    # =========================================================================
    datacube = inst.low_pass_filter(datacube)

    # =========================================================================
    # Patch isolated bad pixels
    # =========================================================================
    # for each slice of the cube, isolated bad pixels are interpolated with the
    # value of their 4 neighbours.
    datacube = inst.patch_isolated_bads(datacube)

    # =========================================================================
    # remove cosmic rays with a sigma cut
    # =========================================================================
    datacube = inst.remove_cosmic_rays(datacube)

    # =========================================================================
    # get the trace map
    # =========================================================================
    trace_mask = inst.get_trace_mask(cube=datacube, no_plot=False,
                                     plot_frames=[0, -1])

    # =========================================================================
    # Create the median stack
    # =========================================================================
    datacube, med, amps = inst.create_median_stack(datacube)
    # The legacy list annotation cannot express the three concrete outputs.
    datacube = cast(DataCube, datacube)
    med = cast(np.ndarray, med)
    amps = cast(np.ndarray, amps)

    # =========================================================================
    # Differential 1/f correction
    # =========================================================================
    # if you want to subtract a higher order polynomial to the 1/f noise, change
    # the value of fit_order
    datacube = inst.clean_residual_1f(datacube, med, amps, trace_mask)

    # =========================================================================
    # recenter the trace position
    # =========================================================================
    trace_mask = inst.recenter_trace_position(trace_mask, med)

    # =========================================================================
    # PCA Analysis
    # =========================================================================
    # construct the principal component model from the out of transit domain
    # using pca (we deal with not fitting the PCA inside)
    pcas = inst.fit_pca(datacube, med, trace_mask)

    # =========================================================================
    # Linear reconstruction
    # =========================================================================
    # Following Equation A1 from Lim et al. 2023 
    # (https://iopscience.iop.org/article/10.3847/2041-8213/acf7c4/pdf)
    #
    # Flux = amp[0] x M + amp[1] x dM/dx + amp[2] x dM/dy 
    #        + amp[3] x dM/dtheta + amp[4] x d2M/dy2
    # =========================================================================
    # Step 1: get the parameters to fit
    # =========================================================================
    dx, dy, rotxy, ddy, med_clean = inst.get_fit_params(med)
    # -------------------------------------------------------------------------
    # set up the mask for trace position
    mask_out = inst.get_linear_recon_mask(med, trace_mask)
    mask_trace_pos, x_order0, y_order0, x_trace_pos, y_trace_pos = mask_out
    # =========================================================================
    # Step 2: setup the mask for linear reconstruction
    # =========================================================================
    lvector = inst.setup_linear_reconstruction(med, dx, dy, rotxy, ddy,
                                               pcas)
    # =========================================================================
    # Step 3: Construct and run the linear reconstruction
    # =========================================================================
    # find the best linear combination of scale/dx/dy/rotation from lvector
    # amps is a vector with the amplitude of all 4 fitted terms
    # amps[0] -> amplitude of trace
    # amps[1] -> dx normalized on reference trace
    # amps[2] -> dy normalized on reference trace
    # amps[3] -> rotation (in radians) normalized on reference trace
    # amps[4] -> 2nd derivative in y [if option activated]

    # Flux = amp[0] x M + amp[1] x dM/dx + amp[2] x dM/dy + amp[3] x dM/dtheta 
    #      + amp[4] x d2M/dy2
    # -------------------------------------------------------------------------
    l_out = inst.get_linear_coeffs(datacube, med, mask_trace_pos,
                                   lvector, x_trace_pos, y_trace_pos,
                                   x_order0, y_order0)
    # get outputs of apply_amp_recon
    #  Note the recon model has been subtracted from the cube in order to 
    #  later to differential spectral extraction
    ltable, lproducts = l_out

    # =========================================================================
    # Linear recon analysis
    # =========================================================================
    # Add integration times to the table
    ltable = inst.add_integration_times(ltable)
    # -------------------------------------------------------------------------
    # normalize the trace but a normalization factor
    ltable = inst.normalize_sum_trace(ltable)
    # -------------------------------------------------------------------------
    # print the rms baseline for all methods
    for method in inst.rms_baselines():
        # calculate the rms for this method
        rms_method = inst.get_rms_baseline(np.asarray(ltable['amplitude']),
                          method=method)
        # print this
        msg = '{0}, rms = {1:.1f}ppm'.format(method, rms_method * 1e6)
        WLOG(inst.params, 'info', msg)
    # -------------------------------------------------------------------------
    # calculate and print the effective wavelength
    inst.get_effective_wavelength(med)

    # =========================================================================
    # Preemptively correct the cube for spectral extraction
    # =========================================================================
    # This corrects the whole time series based on the out-of-transit slope 
    # - use very carefuly.

    # This is done here so we only save the cube once to disk 
    # (and we don't have reopen it)

    # if inst.params['WLC.GENERAL.PER_PIXEL_BASELINE_CORRECTION']:
    #     WLOG(inst.params, 'info', 'Performing per-pixel baseline subtraction')
    #     products = inst.per_pixel_baseline(lproducts)

    # =========================================================================
    # write files
    # =========================================================================
    inst.save_input_residuals(lproducts)
    inst.save_wlc_results(lproducts, ltable)

    # =========================================================================
    # Plots and Summary HTML
    # =========================================================================
    # plot the stability plot
    plots.plot_stability(inst, ltable)
    # -------------------------------------------------------------------------
    # Detector children defer the shared summary page to their parent.
    if getattr(inst, '_detector_trace', None) is None:
        write_sossisse_summary(inst)
    # -------------------------------------------------------------------------
    # return the instrument object
    return inst



def spectral_extraction_chunk(inst: Instrument) -> dict:
    """
    Spectral extraction functionality

    :param inst: Instrument, the instrument object
    :return: dict, spectral extraction storage
    """
    objname = inst.params.rget('INPUTS.OBJECTNAME', required=True)
    # print the splash
    misc.sossart()
    # =========================================================================
    # Spectral extraction setup
    # =========================================================================
    # print the white light curve splash
    print(misc.art('Spectral timeseries ' + objname, 'blue', 'CYAN',
                   message=_processing_splash_message(inst)))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    products, ltable = inst.load_wlc_results()
    # -------------------------------------------------------------------------
    # plot / save storage
    storage = dict()
    # get the traces
    traces = inst.get_traces()
    # =========================================================================
    # Processing trace loop
    # =========================================================================
    # loop around traces (orders or detectors) and process each one
    for trace in traces:
        # print progress
        if len(traces) is not None:
            WLOG(inst.params, 'info', 'Processing trace {0}'.format(trace), colour='magenta')
        # ---------------------------------------------------------------------
        # load data for this trace
        indata = inst.load_input_spec_data(trace, products=products)
        med, med_clean, posmax, throughput, wavegrid = indata
        # ---------------------------------------------------------------------
        # create the SED
        sed_args = [med, products.shape, wavegrid, posmax, throughput,
                    med_clean, trace]
        sp_sed = inst.create_sed(*sed_args)
        # Only extracted (integration, wavelength) arrays survive block reads.
        spec = np.full((len(products), products.shape[2]), np.nan)
        spec_err = np.full_like(spec, np.nan)
        for integration, arrays in products.iter_integrations():
            ratio_args = [arrays['RECON'], arrays['ERR'], arrays['RESIDUAL'],
                  posmax]
            block_spec, block_err = inst.ratio_residual_to_trace(*ratio_args)
            spec[integration] = block_spec
            spec_err[integration] = block_err
            del arrays, ratio_args, block_spec, block_err
        # ---------------------------------------------------------------------
        # # remove the out-of-transit trend on the spectrum
        # if inst.params['SPEC_EXT.REMOVE_TREND']:
        #     spec = inst.remove_trend_spec(spec)
        # -----------------------------------------------------------------
        # reshape the amplitudes into an image
        amp_image = np.repeat(np.array(ltable['amplitude']), spec.shape[1])
        amp_image = amp_image.reshape(spec.shape)
        # add this gray component onto the spectrum
        spec2 = spec + amp_image
        # ---------------------------------------------------------------------
        # plot the spec2 data
        plots.plot_spectral_timeseries(inst, spec2, trace)
        # ---------------------------------------------------------------------
        # # remove the out-of-transit trend on the photometric time series
        # if inst.params['SPEC_EXT.REMOVE_TREND']:
        #     ltable = inst.remove_trend_phot(spec, ltable)
        # ---------------------------------------------------------------------
        # # compute or set transit depth
        # transit_depth = inst.get_transit_depth(ltable)
        # # -------------------------------------------------------------------
        # # get the in-transit spectrum
        # isout = inst.intransit_spectrum(spec, spec_err)
        # spec_in, spec_err_in, spec_err_out = isout
        # # -------------------------------------------------------------------
        # # bin the data by RESOLUTION_BIN
        # wave_bin, flux_bin, flux_bin_err = inst.bin_spectrum(wavegrid,
        #                                                      spec_in,
        #                                                      spec_err_in)
        # ---------------------------------------------------------------------
        # push into storage for outside loop
        storage_args = [storage, trace, wavegrid, sp_sed, throughput, spec,
                spec_err, ltable, spec2]
        storage = trace_storage(*storage_args)
        # ---------------------------------------------------------------------
        inst.save_spe_results(storage[trace], trace)
    # return storage for final output generation or merging
    return storage



# =============================================================================
# Define linear working functions
# =============================================================================
def get_wlc_files(inst: Instrument):
    """
    Get WLC product filenames from the current instrument context.

    :param inst: Instrument, the instrument object
    :return: dict, WLC product filenames
    """
    # Keep the function name for informative get_variable errors.
    func_name = f'{__NAME__}.get_wlc_files()'
    # Capture the currently suffixed WLC filenames after a chunk is processed.
    return dict(manifest=inst.get_wlc_manifest(),
                ltable=inst.get_variable('WLC_LTBL_FILE', func_name))



def merge_linear_recon(inst: Instrument, detectors: dict, wlc_files: dict,
                       jump_ints: list) -> Instrument:
    """Finish each detector's time-axis merge and update the main instrument."""
    for detector, detector_inst in detectors.items():
        if jump_ints:
            detectors[detector] = merge_linear_recon_chunks(
                detector_inst, wlc_files[detector], jump_ints)
    if next(iter(detectors.values())) is not inst:
        suffix = '_jumpmerged' if jump_ints else ''
        inst.set_jump_context(suffix=suffix, jump_ints=jump_ints)
        inst.define_filenames()
        write_sossisse_summary(inst)
    return inst



def merge_linear_recon_chunks(inst: Instrument, wlc_files: list,
                              jump_ints: list) -> Instrument:
    """
    Merge chunked linear-reconstruction products.

    :param inst: Instrument, the instrument object
    :param wlc_files: list, per-chunk WLC filenames
    :param jump_ints: list, jump integrations
    :return: Instrument, updated instrument object
    """
    # Announce the merge so logs clearly separate chunk runs from the merge.
    msg = 'Merging jump linear reconstruction products'
    WLOG(inst.params, 'info', msg)
    loaded = [ReconstructionProducts.load(files['manifest'])
              for files in wlc_files]
    paths = [path for products, _ in loaded
             for path in products.current_infiles]
    limit = max(products.max_integrations for products, _ in loaded)
    memory = min(products.max_math_memory_gb for products, _ in loaded)
    products = ReconstructionProducts(paths, limit, memory)
    ltables = [table for _, table in loaded]
    # Stack the tables in chunk order to recover the full integration series.
    ltable = vstack(ltables)
    # Switch filenames to the final merged product namespace.
    inst.set_jump_context(suffix='_jumpmerged', jump_ints=jump_ints)
    # The last chunk left DATA_N_FRAMES set to its local number of frames.
    # Reset it to the full table length before rebuilding baseline domains.
    inst.set_variable('DATA_N_FRAMES', len(ltable))
    # Recompute filenames so save_wlc_results writes *_jumpmerged products.
    inst.define_filenames()
    # Save merged WLC files in the same format as the normal no-jump products.
    inst.save_wlc_results(products, ltable)
    # Rebuild the stability plot from the merged table.
    plots.plot_stability(inst, ltable)
    # Return the instrument with filenames pointing at the merged products.
    return inst



# =============================================================================
# Define spectral extraction functions
# =============================================================================
def merge_spectral_extraction(inst: Instrument, detectors: dict,
                              storages: dict, jump_ints: list) -> Instrument:
    """Finish detector-local spectra and publish the combined products."""
    combined = {}
    for detector, detector_inst in detectors.items():
        chunks = storages[detector]
        if jump_ints:
            storage = merge_spectral_chunks(chunks)
            detector_inst.set_jump_context(suffix='_jumpmerged',
                                           jump_ints=jump_ints)
            trace = next(iter(storage))
            detector_inst.set_variable('DATA_N_FRAMES',
                                       len(storage[trace]['ltable']))
            detector_inst.define_filenames()
            for trace in storage:
                detector_inst.save_spe_results(storage[trace], trace)
        else:
            storage = chunks[0]
        finish_spectral_extraction(detector_inst, storage)
        combined.update(storage)
    if next(iter(detectors.values())) is not inst:
        suffix = '_jumpmerged' if jump_ints else ''
        inst.set_jump_context(suffix=suffix, jump_ints=jump_ints)
        inst.define_filenames()
        inst.spectral_storage = combined
        inst.save_final_outputs(combined)
        write_sossisse_summary(inst)
    return inst



def merge_spectral_chunks(storages: list) -> dict:
    """
    Merge chunked spectral extraction storage.

    :param storages: list, per-chunk spectral extraction storage
    :return: dict, merged spectral extraction storage
    """
    # Make a new storage dictionary in the same shape as the no-jump path.
    storage = dict()
    # Use the first chunk to discover which traces are present.
    traces = list(storages[0].keys())
    # Merge each trace independently.
    for trace in traces:
        # Create the per-order storage dictionary expected by later writers.
        storage_it = dict()
        # Wavelength-like arrays are fixed within one trace across chunks.
        first = storages[0][trace]
        # Preserve the 1D wavelength grid for this order.
        storage_it['wavegrid'] = np.array(first['wavegrid'])
        # Average per-chunk SED estimates to make one merged SED estimate.
        storage_it['sp_sed'] = np.nanmean([chunk[trace]['sp_sed']
                                         for chunk in storages], axis=0)
        # Average throughput arrays; these should normally be identical.
        storage_it['throughput'] = np.nanmean([chunk[trace]['throughput']
                                             for chunk in storages], axis=0)
        # These arrays are time-dependent, so merge them along time.
        for key in ['spec', 'spec_err', 'spec2', 'wavegrid_2d']:
            # Collect the same product from every chunk in chronological order.
            values = [chunk[trace][key] for chunk in storages]
            # Concatenate on axis 0 to restore the full integration series.
            storage_it[key] = np.concatenate(values, axis=0)
        # The light-curve table is also time-dependent.
        tables = [chunk[trace]['ltable'] for chunk in storages]
        # Stack the per-chunk tables in the same order as the arrays.
        storage_it['ltable'] = vstack(tables)
        # Store this trace under its original order number.
        storage[trace] = storage_it
    # Return merged storage that downstream writers can treat as normal output.
    return storage



def finish_spectral_extraction(inst: Instrument, storage: dict):
    """
    Save and summarize final spectral extraction products.

    :param inst: Instrument, the instrument object
    :param storage: dict, spectral extraction storage
    :return: None
    """
    # Expose per-trace products for the detector parent's combined output.
    inst.spectral_storage = storage
    # -------------------------------------------------------------------------
    # plot the SED
    plots.plot_full_sed(inst, storage)
    # -------------------------------------------------------------------------
    # convert sossisse to eureka products
    inst.to_eureka(storage)
    # -------------------------------------------------------------------------
    # Only the parent writes combined detector outputs; children keep theirs.
    if getattr(inst, '_detector_trace', None) is None:
        inst.save_final_outputs(storage)
    # -------------------------------------------------------------------------
    # Detector children defer the shared summary page to their parent.
    if getattr(inst, '_detector_trace', None) is None:
        write_sossisse_summary(inst)



def trace_storage(storage: dict, trace, wavegrid, sp_sed, throughput,
                  spec, spec_err, ltable, spec2):
    """
    Copy one trace's spectra and timing into the shared result mapping.

    A trace is a SOSS diffraction order, NIRSpec detector (1=NRS1, 2=NRS2),
    or None for an unnumbered instrument. Detector tables retain their own
    timestamps; SOSS orders can share reconstruction amplitudes.

    :param storage: dict, trace-to-result mapping updated in place
    :param trace: int or None, identity used as the result key
    :param wavegrid: array, wavelength grid of shape (n_pixels,)
    :param sp_sed: array, spectral energy distribution of shape (n_pixels,)
    :param throughput: array, throughput of shape (n_pixels,)
    :param spec: array, residual spectra of shape (n_integrations, n_pixels)
    :param spec_err: array, spectral errors with the same shape as spec
    :param ltable: Table, reconstruction coefficients and per-integration bjd
    :param spec2: array, amplitude-restored spectra with the same shape as spec
    :return: dict, the input mapping containing isolated copies for this trace
    """
    # reshape the wave grid into an image
    wavegrid_2d = np.tile(wavegrid, (spec.shape[0], 1))
    # save for plotting (outside the trace loop) / saving
    # must copy here to avoid shallow copying between orders
    storage_it = dict()
    storage_it['wavegrid'] = np.array(wavegrid)
    storage_it['sp_sed'] = np.array(sp_sed)
    storage_it['throughput'] = np.array(throughput)
    storage_it['spec'] = np.array(spec)
    storage_it['spec_err'] = np.array(spec_err)
    storage_it['ltable'] = Table(ltable)
    storage_it['spec2'] = np.array(spec2)
    storage_it['wavegrid_2d'] = wavegrid_2d
    # append to plot storage
    storage[trace] = storage_it

    return storage



# =============================================================================
# Define other working functions
# =============================================================================
def get_jump_chunks(inst: Instrument):
    """
    Get the validated jump chunks for this run.

    :param inst: Instrument, the instrument object
    :return: tuple, list of chunk dictionaries and list of jump integrations
    """
    # Count the input frames after any time-binning has been applied.
    nframes = inst.get_input_n_frames()
    # Read the user-supplied jump locations from the SOSSISSE WLC parameters.
    jump_ints = inst.params['WLC.INPUTS.JUMP_INTS']
    # Treat a missing value the same as an empty list of jumps.
    if jump_ints is None:
        jump_ints = []
    # The yaml value must be a list so that ordering and duplicates are explicit.
    if not isinstance(jump_ints, list):
        emsg = 'WLC.INPUTS.JUMP_INTS must be a list of integers or None'
        raise exceptions.SossisseConstantException(emsg)
    # Convert values to integers here so all downstream code gets one type.
    try:
        jump_ints = [int(jump_int) for jump_int in jump_ints]
    # Give a parameter-level error if one of the entries cannot be an integer.
    except Exception as _:
        emsg = 'WLC.INPUTS.JUMP_INTS must contain integers'
        raise exceptions.SossisseConstantException(emsg)
    # Sort jumps so chunk construction is deterministic even if yaml is not.
    jump_ints = sorted(jump_ints)
    # Duplicate jumps would create an empty chunk, so reject them early.
    if len(jump_ints) != len(set(jump_ints)):
        emsg = 'WLC.INPUTS.JUMP_INTS must not contain duplicate values'
        raise exceptions.SossisseConstantException(emsg)
    # Validate each jump against the binned integration range.
    for jump_int in jump_ints:
        # A jump marks the last frame in a chunk, so the final frame is invalid.
        if jump_int < 0 or jump_int >= nframes - 1:
            emsg = ('WLC.INPUTS.JUMP_INTS values must be between 0 and '
                    f'{nframes - 2}')
            raise exceptions.SossisseConstantException(emsg)
    # no jumps: one normal chunk with no suffix
    if len(jump_ints) == 0:
        # Use named keys so callers do not need positional tuple indexes.
        chunk = dict(start=0, stop=nframes, suffix='')
        # Return the single legacy-style chunk and the empty jump list.
        return [chunk], jump_ints
    # The first chunk always starts at integration zero.
    starts = [0]
    # Each later chunk starts immediately after the previous jump frame.
    starts.extend([jump_int + 1 for jump_int in jump_ints])
    # Each jump is stored as the final included frame, so stop is jump + 1.
    stops = [jump_int + 1 for jump_int in jump_ints]
    # The final chunk runs to the end of the time series.
    stops.append(nframes)
    # Build a list of named chunk definitions.
    chunks = []
    # Number suffixes by chunk, not by the integration number, for compact names.
    for chunk_num, start in enumerate(starts):
        # Zero-fill suffixes so filenames sort in processing order.
        suffix = '_jump{0:03d}'.format(chunk_num)
        # Store stop as exclusive, which matches normal Python slicing.
        stop = stops[chunk_num]
        # Store each chunk as a dictionary so fields are self-documenting.
        chunk = dict(start=start, stop=stop, suffix=suffix)
        # Append this chunk to the chronological processing list.
        chunks.append(chunk)
    # Return both the chunk layout and normalized jump list for metadata.
    return chunks, jump_ints



def write_sossisse_summary(inst: Instrument):
    """Write one summary page after the active processing context finishes.

    Detector children share a plot directory, so the parent writes the page
    once all detector-specific plots have been generated.

    :param inst: Instrument, the completed parent or single-trace instrument
    :return: None
    """
    # Read identity from the completed instrument configuration.
    objname = inst.params['INPUTS.OBJECTNAME']
    # Keep the instrument mode in the page title for existing users.
    imode = inst.params['INPUTS.INSTRUMENTMODE']
    # Generate one index page for the shared plot directory.
    io.summary_html(inst.params, 'SOSSISSE', 'PATHS.PLOT_PATH',
                    'INPUTS.PROJECT_NAME', 'PATHS.RUN_PATH',
                    f'{objname} [{imode}]')



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
