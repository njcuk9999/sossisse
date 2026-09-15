#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sossisse.general.py

General functionality in here

Created on 2022-09-20

@author: cook
"""
import os

import numpy as np
from astropy.table import Table
from astropy.table import vstack

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.core import misc
from sossisse.general import plots
from sossisse.instruments import Instrument

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.misc'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
def linear_recon_init(inst):
    # set the function name
    func_name = f'{__NAME__}.linear_recon'
    # print the splash
    misc.sossart()
    # get parameters from instrumental parameters
    objname = inst.params['INPUTS.OBJECTNAME']
    # print the white light curve splash
    print(misc.art('Linear Recon ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # get the stabiblity table file name
    wlc_ltbl_file = inst.get_variable('WLC_LTBL_FILE', func_name)
    # return if we have the soss_stablity file
    if os.path.exists(wlc_ltbl_file) and inst.params['GENERAL.USE_TEMPORARY']:
        msg = 'File {0} exists we skip linear reconstruction step'
        misc.printc(msg.format(wlc_ltbl_file), 'info')
        return True
    # if we've got here return false
    return False


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


def get_wlc_files(inst: Instrument):
    """
    Get WLC product filenames from the current instrument context.

    :param inst: Instrument, the instrument object
    :return: dict, WLC product filenames
    """
    # Keep the function name for informative get_variable errors.
    func_name = f'{__NAME__}.get_wlc_files()'
    # Capture the currently suffixed WLC filenames after a chunk is processed.
    return dict(err=inst.get_variable('WLC_ERR_FILE', func_name),
                res=inst.get_variable('WLC_RES_FILE', func_name),
                recon=inst.get_variable('WLC_RECON_FILE', func_name),
                ltable=inst.get_variable('WLC_LTBL_FILE', func_name))


def merge_linear_recon(inst: Instrument, wlc_files: list,
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
    misc.printc(msg, 'info')
    # Merge the per-frame error cubes along the time axis.
    err = np.concatenate([io.load_fits(files['err']) for files in wlc_files])
    # Merge the residual cubes along the time axis.
    res = np.concatenate([io.load_fits(files['res']) for files in wlc_files])
    # Merge the reconstructed model cubes along the time axis.
    recon = np.concatenate([io.load_fits(files['recon'])
                            for files in wlc_files])
    # Read each per-chunk linear-reconstruction coefficient table.
    ltables = [io.load_table(files['ltable']) for files in wlc_files]
    # Stack the tables in chunk order to recover the full integration series.
    ltable = vstack(ltables)
    # Switch filenames to the final merged product namespace.
    inst.set_jump_context(suffix='_jumpmerged', jump_ints=jump_ints)
    # Recompute filenames so save_wlc_results writes *_jumpmerged products.
    inst.define_filenames()
    # Save merged WLC files in the same format as the normal no-jump products.
    inst.save_wlc_results(res, err, recon, ltable)
    # Rebuild the stability plot from the merged table.
    plots.plot_stability(inst, ltable)
    # Return the instrument with filenames pointing at the merged products.
    return inst


def linear_recon(inst: Instrument) -> Instrument:
    """
    Linear reconstruction functionality

    :param inst: Instrument, the instrument object
    :return: Instrument, updated instrument object
    """
    # Convert the JUMP_INTS parameter into validated processing chunks.
    chunks, jump_ints = get_jump_chunks(inst)
    # normal path: keep filenames unchanged and avoid writing merged products
    if len(jump_ints) == 0:
        # Use the no-suffix context so all filenames match the legacy behavior.
        chunk = chunks[0]
        # Activate the single no-jump chunk by name rather than by tuple index.
        inst.set_jump_context(chunk['start'], chunk['stop'], chunk['suffix'],
                              jump_ints)
        # Run the original linear-reconstruction body once.
        return linear_recon_chunk(inst)
    # jump path: run each chunk independently, then merge products
    # Store the per-chunk WLC filenames so the merge step can reload them.
    wlc_files = []
    # Loop over the validated chunks in chronological order.
    for chunk in chunks:
        # Log the inclusive frame range being processed for this chunk.
        msg = 'Processing linear reconstruction chunk {0}: {1}-{2}'
        margs = [chunk['suffix'], chunk['start'], chunk['stop'] - 1]
        misc.printc(msg.format(*margs), 'alert')
        # Activate chunk-local slicing and filename suffixing.
        inst.set_jump_context(chunk['start'], chunk['stop'], chunk['suffix'],
                              jump_ints)
        # Run the original linear-reconstruction body on this one chunk.
        inst = linear_recon_chunk(inst)
        # Remember where this chunk wrote its WLC products.
        wlc_files.append(get_wlc_files(inst))
    # Concatenate the chunk products and leave the instrument on _jumpmerged.
    inst = merge_linear_recon(inst, wlc_files, jump_ints)
    # Return the same instrument object expected by notebooks and POGOS.
    return inst


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
    cube, err, dq = inst.load_data_with_dq()

    # =========================================================================
    # Apply the flat field
    # =========================================================================
    cube, err, dq = inst.apply_flat_field(cube, err, dq)

    # =========================================================================
    # Keep only certain Data Quality flags
    # =========================================================================
    cube, err = inst.apply_dq(cube, err, dq)

    # =========================================================================
    # remove the background
    # =========================================================================
    cube, err = inst.remove_background(cube, err)

    # =========================================================================
    # low pass the data
    # =========================================================================
    cube, err = inst.low_pass_filter(cube, err)

    # =========================================================================
    # Patch isolated bad pixels
    # =========================================================================
    # for each slice of the cube, isolated bad pixels are interpolated with the
    # value of their 4 neighbours.
    cube, err = inst.patch_isolated_bads(cube, err)

    # =========================================================================
    # remove cosmic rays with a sigma cut
    # =========================================================================
    cube = inst.remove_cosmic_rays(cube)

    # =========================================================================
    # optimize trace position
    # =========================================================================
    inst.optimize_trace_mask()

    # =========================================================================
    # get the trace map
    # =========================================================================
    trace_mask = inst.get_trace_mask(cube=cube, no_plot=False,
                                     plot_frames=[0, -1])

    # =========================================================================
    # Create the median stack
    # =========================================================================
    cube, med, amps = inst.create_median_stack(cube)

    # =========================================================================
    # Differential 1/f correction
    # =========================================================================
    # if you want to subtract a higher order polynomial to the 1/f noise, change
    # the value of fit_order
    cube = inst.clean_residual_1f(cube, err, med, amps, trace_mask)

    # =========================================================================
    # recenter the trace position
    # =========================================================================
    trace_mask = inst.recenter_trace_position(trace_mask, med)

    # =========================================================================
    # PCA Analysis
    # =========================================================================
    # construct the principal component model from the out of transit domain
    # using pca (we deal with not fitting the PCA inside)
    pcas = inst.fit_pca(cube, err, med, trace_mask)

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
    l_out = inst.get_linear_coeffs(cube, err, med, mask_trace_pos,
                                   lvector, x_trace_pos, y_trace_pos,
                                   x_order0, y_order0)
    # get outputs of apply_amp_recon
    #  Note the recon model has been subtracted from the cube in order to 
    #  later to differential spectral extraction
    ltable, lrecon, valid_cube, rescube = l_out

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
        rms_method = inst.get_rms_baseline(ltable['amplitude'], method=method)
        # print this
        msg = '{0}, rms = {1:.1f}ppm'.format(method, rms_method * 1e6)
        misc.printc(msg, 'number')
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
    #     misc.printc('Performing per-pixel baseline subtraction', 'info')
    #     rescube = inst.per_pixel_baseline(rescube, valid_cube)

    # =========================================================================
    # write files
    # =========================================================================
    inst.save_wlc_results(rescube, err, lrecon, ltable)

    # =========================================================================
    # Plots and Summary HTML
    # =========================================================================
    # plot the stability plot
    plots.plot_stability(inst, ltable)
    # -------------------------------------------------------------------------
    # write the yaml file to html
    objname = inst.params['INPUTS.OBJECTNAME']
    imode = inst.params['INPUTS.INSTRUMENTMODE']
    io.summary_html(inst.params, 'SOSSISSE', 'PATHS.PLOT_PATH',
                    'INPUTS.SUBDIRECTORY', 'PATHS.SUBDIRECTORY_PATH',
                    f'{objname} [{imode}]')
    # -------------------------------------------------------------------------
    # return the instrument object
    return inst


def spectral_extraction(inst: Instrument) -> Instrument:
    """
    Spectral extraction functionality

    :param inst: Instrument, the instrument object
    :return: Instrument, updated instrument object
    """
    # Convert the JUMP_INTS parameter into validated processing chunks.
    chunks, jump_ints = get_jump_chunks(inst)
    # normal path: keep filenames unchanged and finish as before
    if len(jump_ints) == 0:
        # Use the no-suffix context so all filenames match the legacy behavior.
        chunk = chunks[0]
        # Activate the single no-jump chunk by name rather than by tuple index.
        inst.set_jump_context(chunk['start'], chunk['stop'], chunk['suffix'],
                              jump_ints)
        # Run the original spectral-extraction body once and keep its storage.
        storage = spectral_extraction_chunk(inst)
        # Write Eureka/final products and summary from the single storage block.
        finish_spectral_extraction(inst, storage)
        # Return the instrument object for the normal public API.
        return inst
    # jump path: load and process each chunk independently
    # Store each chunk's spectral storage until the final time-axis merge.
    storages = []
    # Loop over the same chunk definitions used by linear reconstruction.
    for chunk in chunks:
        # Log the inclusive frame range being processed for this chunk.
        msg = 'Processing spectral extraction chunk {0}: {1}-{2}'
        margs = [chunk['suffix'], chunk['start'], chunk['stop'] - 1]
        misc.printc(msg.format(*margs), 'alert')
        # Activate the matching WLC input filenames and local frame context.
        inst.set_jump_context(chunk['start'], chunk['stop'], chunk['suffix'],
                              jump_ints)
        # Run the original spectral-extraction body for this one chunk.
        storages.append(spectral_extraction_chunk(inst))
    # Concatenate per-chunk spectral arrays and tables into full-length storage.
    storage = merge_spectral_extraction(storages)
    # Switch filenames to the final merged product namespace.
    inst.set_jump_context(suffix='_jumpmerged', jump_ints=jump_ints)
    # Recompute filenames so spectral products are saved with _jumpmerged.
    inst.define_filenames()
    # Save merged per-order spectral products for users who inspect them directly.
    for trace_order in storage:
        inst.save_spe_results(storage[trace_order], trace_order)
    # Write Eureka/final products and summary from the merged storage block.
    finish_spectral_extraction(inst, storage)
    # Return the instrument with downstream-visible filenames on _jumpmerged.
    return inst


def spectral_extraction_chunk(inst: Instrument) -> dict:
    """
    Spectral extraction functionality

    :param inst: Instrument, the instrument object
    :return: dict, spectral extraction storage
    """
    # print the splash
    misc.sossart()
    # =========================================================================
    # Spectral extraction setup
    # =========================================================================
    # get parameters from instrumental parameters
    objname = inst.params['INPUTS.OBJECTNAME']
    # print the white light curve splash
    print(misc.art('Spectral timeseries ' + objname, 'blue', 'CYAN'))
    # -------------------------------------------------------------------------
    # load temporary filenames (should be run before science starts)
    inst.define_filenames()
    # -------------------------------------------------------------------------
    # plot / save storage
    storage = dict()
    # get the trace orders
    trace_orders = inst.get_trace_orders()
    # =========================================================================
    # Processing trace order loop
    # =========================================================================
    # loop around trace orders
    for trace_order in trace_orders:
        # print progress
        misc.printc('Processing trace order {0}'.format(trace_order), 'alert')
        # ---------------------------------------------------------------------
        # load data for this trace order
        indata = inst.load_input_spec_data(trace_order)
        med, dx, dy, rotxy, ddy, med_clean, residual, err = indata[:8]
        recon, ltable, posmax, throughput, wavegrid = indata[8:]
        # ---------------------------------------------------------------------
        # create the SED
        sp_sed = inst.create_sed(med, residual, wavegrid, posmax, throughput,
                                 med_clean, trace_order)
        # ---------------------------------------------------------------------
        # load the model (and deal with masking order zero if required)
        model = inst.load_model(recon, med)
        # ---------------------------------------------------------------------
        # spectrum is the ratio of the residual to the trace model
        spec, spec_err = inst.ratio_residual_to_trace(model, err, residual,
                                                      posmax)
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
        plots.plot_spectral_timeseries(inst, spec2, trace_order)
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
        storage = trace_storage(storage, trace_order, wavegrid, sp_sed,
                                throughput, spec, spec_err, ltable, spec2)
        # ---------------------------------------------------------------------
        inst.save_spe_results(storage[trace_order], trace_order)
    # return storage for final output generation or merging
    return storage


def merge_spectral_extraction(storages: list) -> dict:
    """
    Merge chunked spectral extraction storage.

    :param storages: list, per-chunk spectral extraction storage
    :return: dict, merged spectral extraction storage
    """
    # Make a new storage dictionary in the same shape as the no-jump path.
    storage = dict()
    # Use the first chunk to discover which trace orders are present.
    trace_orders = list(storages[0].keys())
    # Merge each trace order independently.
    for trace_order in trace_orders:
        # Create the per-order storage dictionary expected by later writers.
        storage_it = dict()
        # Use the first chunk for wavelength-like arrays that should not change.
        first = storages[0][trace_order]
        # Preserve the 1D wavelength grid for this order.
        storage_it['wavegrid'] = np.array(first['wavegrid'])
        # Average per-chunk SED estimates to make one merged SED estimate.
        storage_it['sp_sed'] = np.nanmean([chunk[trace_order]['sp_sed']
                                           for chunk in storages], axis=0)
        # Average throughput arrays; these should normally be identical.
        storage_it['throughput'] = np.nanmean([chunk[trace_order]['throughput']
                                               for chunk in storages], axis=0)
        # These arrays are time-dependent, so merge them along time.
        for key in ['spec', 'spec_err', 'spec2', 'wavegrid_2d']:
            # Collect the same product from every chunk in chronological order.
            values = [chunk[trace_order][key] for chunk in storages]
            # Concatenate on axis 0 to restore the full integration series.
            storage_it[key] = np.concatenate(values, axis=0)
        # The light-curve table is also time-dependent.
        tables = [chunk[trace_order]['ltable'] for chunk in storages]
        # Stack the per-chunk tables in the same order as the arrays.
        storage_it['ltable'] = vstack(tables)
        # Store this trace order under its original order number.
        storage[trace_order] = storage_it
    # Return merged storage that downstream writers can treat as normal output.
    return storage


def finish_spectral_extraction(inst: Instrument, storage: dict):
    """
    Save and summarize final spectral extraction products.

    :param inst: Instrument, the instrument object
    :param storage: dict, spectral extraction storage
    :return: None
    """
    # -------------------------------------------------------------------------
    # plot the SED
    plots.plot_full_sed(inst, storage)
    # -------------------------------------------------------------------------
    # convert sossisse to eureka products
    inst.to_eureka(storage)
    # -------------------------------------------------------------------------
    # save final outputs
    inst.save_final_outputs(storage)
    # -------------------------------------------------------------------------
    # write the yaml file to html
    objname = inst.params['INPUTS.OBJECTNAME']
    imode = inst.params['INPUTS.INSTRUMENTMODE']
    io.summary_html(inst.params, 'SOSSISSE', 'PATHS.PLOT_PATH',
                    'INPUTS.SUBDIRECTORY', 'PATHS.SUBDIRECTORY_PATH',
                    f'{objname} [{imode}]')


def trace_storage(storage: dict, trace_order, wavegrid, sp_sed, throughput,
                  spec, spec_err, ltable, spec2):
    # reshape the wave grid into an image
    wavegrid_2d = np.tile(wavegrid, (spec.shape[0], 1))
    # save for plotting (outside the trace_order loop) / saving
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
    storage[trace_order] = storage_it

    return storage



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
