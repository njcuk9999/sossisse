#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check independent NIRSpec detector traces and shared SOSS amplitudes."""

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from aperocore.constants.param_functions import ParamDict

from sossisse.core.exceptions import SossisseConstantException
from sossisse.core import const_funcs
from sossisse.core import constants
from sossisse.instruments import jwst_nirspec
from sossisse.instruments.default import Instrument
from sossisse.general import general


@pytest.mark.parametrize('value', ['wave.fits',
                                  ['wave_nrs1.fits', 'wave_nrs2.fits']])
def test_wavelength_constant_preserves_scalar_or_list(value):
    """Keep scalar/list types through validation, copying and nested CDicts."""
    copied = constants.CDict.copy(source='test')
    constant = copied.get('GENERAL.WAVE_FILE')
    assert constant.validate(test_value=value) == value
    constant.value = value
    assert constant.validate() is True
    assert constant.true_value == value
    nested = constants.ConstDict('test')
    nested.add_sub_cdict('SOSSISSE', copied, source='test')
    nested_constant = nested.get('SOSSISSE.GENERAL.WAVE_FILE')
    assert nested_constant.validate(test_value=value) == value


@pytest.mark.parametrize('value', ['', [], [None], [1],
                                  dict(NRS1='wave.fits')])
def test_wavelength_constant_rejects_invalid_values(value):
    """Reject empty filenames, invalid list entries and detector mappings."""
    with pytest.raises(SossisseConstantException, match='WAVE_FILE'):
        constants.CDict.get('GENERAL.WAVE_FILE').validate(test_value=value)


@pytest.mark.parametrize('value', [[1.9], [True], [], [1, 1]])
def test_trace_constant_does_not_coerce_invalid_identifiers(value):
    """Reject fractional, boolean, empty and duplicate trace selections."""
    trace_constant = constants.CDict.get('GENERAL.TRACES')
    validated = trace_constant.validate(test_value=value)
    trace_params = dict()
    trace_params['GENERAL.TRACES'] = validated
    inst = Instrument(ParamDict(trace_params))
    with pytest.raises(SossisseConstantException, match='GENERAL.TRACES'):
        inst.get_traces()


def science_file(tmp_path, detector, segment=1, start=1, stop=2, time=0):
    """
    Write a small detector segment with explicit integration identities.

    :param tmp_path: Path, directory receiving the generated FITS file
    :param detector: str, primary-header identity (NRS1, NRS2 or invalid)
    :param segment: int, exposure segment number
    :param start: int, first integration number, inclusive
    :param stop: int, final integration number, inclusive
    :param time: number, detector-specific offset added to timestamps in days
    :return: str, FITS path with science, error, DQ and timing extensions
    """
    # Keep identities paired even when detector timestamps differ.
    path = tmp_path / f'{detector}-{segment}-{start}-{time}.fits'
    header_values = dict(DETECTOR=detector, EXPOSURE=1, SUBSIZE1=4,
                         EXSEGNUM=segment, INTSTART=start, INTEND=stop)
    header = fits.Header(header_values)
    times = fits.BinTableHDU.from_columns([
        fits.Column(name='integration_number', format='J',
                    array=np.arange(start, stop + 1)),
        fits.Column(name='int_mid_BJD_TDB', format='D',
                    array=np.arange(start, stop + 1) + 2400000.5 + time),
        fits.Column(name='int_mid_MJD_UTC', format='D',
                    array=np.arange(start, stop + 1) + time)],
        name='INT_TIMES')
    shape = (stop - start + 1, 2, 4)
    level = int(detector[-1]) if detector in ('NRS1', 'NRS2') else 0
    # Distinct constant detector levels expose accidental time concatenation.
    science = np.full(shape, level, dtype=float)
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.ImageHDU(science, name='SCI'),
                  fits.ImageHDU(np.ones(shape), name='ERR'),
                  fits.ImageHDU(np.zeros(shape, dtype=np.uint32), name='DQ'),
                  times]).writeto(path)
    return str(path)


def test_group_independent_times_and_selection(tmp_path):
    """Pair by integration identity, preserving detector-specific times."""
    first = science_file(tmp_path, 'NRS1')
    second = science_file(tmp_path, 'NRS2', time=10)
    expected = dict()
    expected[1] = [first]
    expected[2] = [second]
    assert jwst_nirspec.group_files([second, first]) == expected
    selected = dict()
    selected[2] = [second]
    assert jwst_nirspec.group_files([second, first], [2]) == selected
    assert jwst_nirspec.group_files([second]) == selected


def test_reject_incomplete_pairs_even_when_unselected(tmp_path):
    """Validate all supplied detector pairs before narrowing the selection."""
    filenames = [science_file(tmp_path, 'NRS1'),
                 science_file(tmp_path, 'NRS2'),
                 science_file(tmp_path, 'NRS2', segment=2, start=3, stop=4)]
    with pytest.raises(SossisseConstantException, match='Incomplete'):
        jwst_nirspec.group_files(filenames, [1])


def test_reject_duplicates_unknown_and_missing(tmp_path):
    """Reject duplicate inputs, missing detectors and unknown labels."""
    first = science_file(tmp_path, 'NRS1')
    with pytest.raises(SossisseConstantException, match='Duplicate'):
        jwst_nirspec.group_files([first, first])
    with pytest.raises(SossisseConstantException, match='Missing selected'):
        jwst_nirspec.group_files([first], [2])
    with pytest.raises(SossisseConstantException, match='Unknown'):
        jwst_nirspec.group_files([science_file(tmp_path, 'OTHER')])


def test_segment_overlap_and_numeric_order(tmp_path):
    """Order segments numerically and reject overlapping integrations."""
    first = science_file(tmp_path, 'NRS1', segment=1, start=1, stop=2)
    later = science_file(tmp_path, 'NRS1', segment=10, start=10, stop=11)
    middle = science_file(tmp_path, 'NRS1', segment=2, start=3, stop=4)
    expected = dict()
    expected[1] = [first, middle, later]
    assert jwst_nirspec.group_files([later, middle, first]) == expected
    overlap = science_file(tmp_path, 'NRS1', segment=3, start=2, stop=3)
    with pytest.raises(SossisseConstantException, match='Overlapping'):
        jwst_nirspec.group_files([first, overlap])


def test_pair_integration_identity_not_just_count(tmp_path):
    """Reject equal-sized detector segments with different integration IDs."""
    first = science_file(tmp_path, 'NRS1')
    second = science_file(tmp_path, 'NRS2', start=3, stop=4)
    with pytest.raises(SossisseConstantException, match='Incomplete'):
        jwst_nirspec.group_files([first, second], [1])


def test_reject_nonconsecutive_integration_numbers(tmp_path):
    """Reject gaps in explicit integration IDs even without header bounds."""
    filename = science_file(tmp_path, 'NRS1')
    with fits.open(filename, mode='update') as hdus:
        del hdus[0].header['INTSTART']
        del hdus[0].header['INTEND']
        hdus['INT_TIMES'].data['integration_number'] = [1, 3]
    integration_kwargs = dict(match='Incomplete integrations')
    with pytest.raises(SossisseConstantException, **integration_kwargs):
        jwst_nirspec.group_files([filename])


def make_instrument(tmp_path, traces=None):
    """
    Build a two-detector instrument with isolated temporary calibrations.

    :param tmp_path: Path, directory receiving science and wavelength files
    :param traces: list or None, selected detectors (1=NRS1, 2=NRS2)
    :return: JWST_NIRSPEC, instrument whose detector times differ by ten days
    """
    # Keep all output categories local to the test's temporary directory.
    params = ParamDict()
    for key in ('TEMP_PATH', 'PLOT_PATH', 'OTHER_PATH', 'OUT_PATH',
                'RUN_PATH'):
        params.set(f'PATHS.{key}', str(tmp_path / key), source='test')
    params.set('PATHS.CALIBPATH', str(tmp_path), source='test')
    science_files = [science_file(tmp_path, 'NRS1'),
                     science_file(tmp_path, 'NRS2', time=10)]
    params.set('GENERAL.FILES', science_files, source='test')
    params.set('GENERAL.TRACES', traces, source='test')
    waves = []
    for trace in (1, 2):
        path = tmp_path / f'wave_nrs{trace}.fits'
        # Header identities associate each wavelength grid with its detector.
        wave_columns = dict(xpix=np.arange(4),
                            wavevector=np.arange(4.0) + trace)
        wave_table = Table(wave_columns)
        fits.HDUList([
            fits.PrimaryHDU(header=fits.Header(dict(DETECTOR=f'NRS{trace}'))),
            fits.BinTableHDU(wave_table)]).writeto(path)
        waves.append(path.name)
    params.set('GENERAL.WAVE_FILE', waves, source='test')
    params.set('GENERAL.WAVE_FILE_TYPE', 'fits', source='test')
    params.set('GENERAL.PREFIX', 'target', source='test')
    params.set('WLC.INPUTS.JUMP_INTS', None, source='test')
    for key in ('BKGFILE', 'FLATFILE', 'POS_FILE'):
        params.set(f'GENERAL.{key}', None, source='test')
    return jwst_nirspec.JWST_NIRSPEC(params)


def test_children_isolate_parameters_paths_and_calibrations(tmp_path):
    """Cache children while isolating detector parameters and output paths."""
    inst = make_instrument(tmp_path)
    assert inst.get_traces() == [1, 2]
    children = jwst_nirspec.detector_children(inst)
    assert jwst_nirspec.detector_children(inst) is children
    assert inst.params['GENERAL.TRACES'] == [1, 2]
    assert inst.params['GENERAL.PREFIX'] == 'target'
    for trace, child in children.items():
        assert child.params is not inst.params
        assert child.params['GENERAL.TRACES'] == [trace]
        assert child.get_traces() == [trace]
        child_files = child.params['GENERAL.FILES']
        assert all(f'NRS{trace}' in filename for filename in child_files)
        wavelength_file = child.params['GENERAL.WAVE_FILE']
        assert wavelength_file.endswith(f'wave_nrs{trace}.fits')
        assert jwst_nirspec.detector_children(child) is None
        assert child._trace_suffix == f'_nrs{trace}'
        position_file = child.params['GENERAL.POS_FILE']
        assert position_file.endswith(f'pos_file_nrs{trace}.fits')
        for key in ('TEMP_PATH', 'PLOT_PATH', 'OTHER_PATH', 'OUT_PATH',
                'RUN_PATH'):
            assert child.params[f'PATHS.{key}'] == inst.params[f'PATHS.{key}']
    children[1].params['GENERAL.FILES'].clear()
    assert len(inst.params['GENERAL.FILES']) == 2
    assert len(children[2].params['GENERAL.FILES']) == 1


@pytest.mark.parametrize('selection', [[1], [2], [1, 2]])
def test_selection_output_namespaces(tmp_path, selection):
    """Keep shared paths and suffix child products by detector identity."""
    inst = make_instrument(tmp_path, selection)
    children = jwst_nirspec.detector_children(inst)
    assert list(children) == selection
    assert inst.params['PATHS.OUT_PATH'] == str(tmp_path / 'OUT_PATH')
    assert inst.params['GENERAL.PREFIX'] == 'target'
    selected_suffix = '_' + '_'.join(f'nrs{trace}' for trace in selection)
    combined_output = inst.add_product_suffix(str(tmp_path / 'target_slc.fits'))
    assert combined_output.endswith(f'target_slc{selected_suffix}.fits')
    for trace, child in children.items():
        output = child.add_product_suffix(
            str(tmp_path / 'target_slc.fits'))
        assert output.endswith(f'target_slc_nrs{trace}.fits')


def test_calibration_validation(tmp_path):
    """Check calibration identities and unique optional calibration copies."""
    inst = make_instrument(tmp_path)
    params = inst.params
    first, second = params['GENERAL.WAVE_FILE']
    params['GENERAL.WAVE_FILE'] = first
    first_group = dict()
    first_group[1] = []
    second_group = dict()
    second_group[2] = []
    assert list(jwst_nirspec.calibration_files(params, first_group)) == [1]
    with pytest.raises(SossisseConstantException, match='Missing selected'):
        jwst_nirspec.calibration_files(params, second_group)
    params['GENERAL.WAVE_FILE'] = [first, first]
    with pytest.raises(SossisseConstantException, match='Duplicate'):
        jwst_nirspec.calibration_files(params, first_group)
    params['GENERAL.WAVE_FILE'] = ['absent_nrs1.fits']
    with pytest.raises(SossisseConstantException, match='Missing wavelength'):
        jwst_nirspec.calibration_files(params, first_group)
    assert jwst_nirspec.calibration_detector(
        str(tmp_path / 'wave_nrs2.h5')) == 2
    with pytest.raises(SossisseConstantException, match='Ambiguous'):
        jwst_nirspec.calibration_detector(str(tmp_path / 'wave.h5'))
    params['GENERAL.WAVE_FILE'] = [first, second]
    params['GENERAL.FLATFILE'] = str(tmp_path / first)
    children = jwst_nirspec.detector_children(inst)
    for trace, child in children.items():
        flat_file = child.params['GENERAL.FLATFILE']
        assert flat_file.endswith(f'wave_nrs1_nrs{trace}.fits')
        assert os.path.isfile(flat_file)


def storage_for(trace):
    """
    Build independent synthetic products for one NIRSpec detector.

    :param trace: int, detector label (1=NRS1, 2=NRS2)
    :return: dict, copied spectra, wavelength grid and detector-specific times
    """
    # Separated timestamps expose accidental reuse of the first detector table.
    times = np.arange(2.0) + 10 * trace
    table_columns = dict(bjd=times, amplitude=np.ones(2) * trace,
                         amplitude_error=np.ones(2) * 0.01)
    table = Table(table_columns)
    wavegrid = np.arange(4.0) + trace
    spec = np.ones((2, 4)) * trace
    storage_args = [dict(), trace, wavegrid, np.ones(4), np.ones(4), spec,
                    spec * 0.01, table, spec]
    storage = general.trace_storage(*storage_args)
    return storage[trace]


def test_combined_fits_and_parent_templates(tmp_path):
    """Write combined detector products using the parent's templates."""
    inst = make_instrument(tmp_path)
    spectral_template = str(tmp_path / '{prefix}_custom_slc.fits')
    inst.set_variable('OUT_SPEC_LC_FILE', spectral_template)
    wlc_template = str(tmp_path / '{prefix}_custom_wlc.fits')
    inst.set_variable('OUT_WLC_FILE', wlc_template)
    storage = {trace: storage_for(trace) for trace in (1, 2)}
    inst.out_spec_lc_file(storage, dict(), Table(dict(name=['test'])))
    inst.out_wlc_file(storage, dict(), Table(dict(name=['test'])))
    with fits.open(tmp_path / 'target_custom_slc.fits') as hdus:
        for trace in storage:
            for name in ('WAVELENGTH', 'RELFLUX', 'RELFLUX_ERROR', 'BJD'):
                assert f'{name}_{trace}' in hdus
            np.testing.assert_array_equal(hdus[f'BJD_{trace}'].data,
                                          storage[trace]['ltable']['bjd'])
    with fits.open(tmp_path / 'target_custom_wlc.fits') as hdus:
        for trace in storage:
            detector_times = hdus[f'WLC_TABLE_{trace}'].data['Time']
            np.testing.assert_array_equal(detector_times,
                                          storage[trace]['ltable']['bjd'])


def test_soss_numbered_wlc_retains_shared_amplitude(tmp_path):
    """Label numbered SOSS tables as shared reconstruction amplitudes."""
    soss_params = dict()
    soss_params['GENERAL.PREFIX'] = 'soss'
    inst = Instrument(ParamDict(soss_params))
    inst.name = 'JWST.NIRISS.SOSS'
    inst.set_variable('OUT_WLC_FILE', str(tmp_path / '{prefix}_wlc.fits'))
    shared = storage_for(1)
    storage = dict()
    storage[1] = shared
    storage[2] = shared
    inst.out_wlc_file(storage, dict(), Table(dict(name=['test'])))
    with fits.open(tmp_path / 'soss_wlc.fits') as hdus:
        assert hdus[0].header['WLCMODEL'] == 'SHARED'
        np.testing.assert_array_equal(hdus['WLC_TABLE_1'].data,
                                      hdus['WLC_TABLE_2'].data)


@pytest.mark.parametrize('jumps', [None, [0]])
def test_dispatch_runs_children_and_combines_storage(tmp_path, monkeypatch,
                                                    jumps):
    """Combine detector outputs after running separate optional jump chunks."""
    inst = make_instrument(tmp_path)
    inst.params['WLC.INPUTS.JUMP_INTS'] = jumps
    chunk_calls = []
    saved = []
    summaries = []
    monkeypatch.setattr(general, 'write_sossisse_summary', summaries.append)
    instrument_class = jwst_nirspec.JWST_NIRSPEC
    monkeypatch.setattr(instrument_class, 'define_filenames',
                        lambda self: None)
    monkeypatch.setattr(instrument_class, 'get_input_n_frames', lambda self: 2)
    monkeypatch.setattr(instrument_class, 'save_spe_results',
                        lambda *values: None)

    def run_chunk(child):
        """Record one detector chunk and slice only its integration axis."""
        trace = child._detector_trace
        chunk_calls.append((trace, child._jump_suffix))
        storage = storage_for(trace)
        start, stop = child._jump_slice
        for key in ('spec', 'spec_err', 'spec2', 'wavegrid_2d', 'ltable'):
            storage[key] = storage[key][start:stop]
        chunk_storage = dict()
        chunk_storage[trace] = storage
        return chunk_storage

    monkeypatch.setattr(general, 'spectral_extraction_chunk', run_chunk)
    finish_args = [general, 'finish_spectral_extraction',
                   lambda child, storage: setattr(child, 'spectral_storage',
                                                  storage)]
    monkeypatch.setattr(*finish_args)
    monkeypatch.setattr(inst, 'save_final_outputs',
                        lambda storage: saved.append(storage))
    assert general.spectral_extraction(inst) is inst
    assert summaries == [inst]
    assert list(saved[0]) == [1, 2]
    assert all(trace in (1, 2) for trace, suffix in chunk_calls)
    assert len(chunk_calls) == (2 if jumps is None else 4)
    assert inst._jump_suffix == ('' if jumps is None else '_jumpmerged')
    np.testing.assert_array_equal(saved[0][2]['ltable']['bjd'], [20, 21])


@pytest.mark.parametrize('jumps', [None, [0]])
def test_linear_dispatch_isolated_jump_contexts(tmp_path, monkeypatch, jumps):
    """Run linear reconstruction in independent detector and jump contexts."""
    inst = make_instrument(tmp_path)
    inst.params['WLC.INPUTS.JUMP_INTS'] = jumps
    calls = []
    summaries = []
    monkeypatch.setattr(general, 'write_sossisse_summary', summaries.append)
    instrument_class = jwst_nirspec.JWST_NIRSPEC
    monkeypatch.setattr(instrument_class, 'define_filenames',
                        lambda self: None)
    monkeypatch.setattr(instrument_class, 'get_input_n_frames', lambda self: 2)

    def run_chunk(child):
        """Record the detector, jump interval and unchanged input file list."""
        calls.append((child._detector_trace, child._jump_slice,
                      child.params['GENERAL.FILES']))
        return child

    def merge_chunks(child, files, jump_ints):
        """Expose the merged jump suffix without performing disk I/O."""
        child.set_jump_context(suffix='_jumpmerged', jump_ints=jump_ints)
        return child

    monkeypatch.setattr(general, 'linear_recon_chunk', run_chunk)
    monkeypatch.setattr(general, 'get_wlc_files', lambda child: dict())
    monkeypatch.setattr(general, 'merge_linear_recon', merge_chunks)
    assert general.linear_recon(inst) is inst
    assert summaries == [inst]
    assert len(calls) == (2 if jumps is None else 4)
    assert inst._jump_suffix == ('' if jumps is None else '_jumpmerged')
    for trace, chunk, files in calls:
        assert len(files) == 1
        assert f'NRS{trace}' in files[0]
        assert chunk in ([(0, 2)] if jumps is None else [(0, 1), (1, 2)])


def test_automatic_position_ignores_stale_parent_map(tmp_path):
    """Replace a stale automatic parent position map with detector paths."""
    inst = make_instrument(tmp_path)
    path = tmp_path / 'pos_file.fits'
    fits.PrimaryHDU(np.zeros((2, 4))).writeto(path)
    inst.params['GENERAL.POS_FILE'] = str(path)
    inst.params['__AUTO_POS_FILE__'] = True
    for trace, child in jwst_nirspec.detector_children(inst).items():
        position_file = child.params['GENERAL.POS_FILE']
        assert position_file.endswith(f'pos_file_nrs{trace}.fits')


def test_scalar_headerless_wave_remains_supported(tmp_path):
    """Allow an unlabeled scalar calibration only for one selected detector."""
    inst = make_instrument(tmp_path, [1])
    path = tmp_path / 'wavelength.fits'
    fits.PrimaryHDU(np.arange(4.0)).writeto(path)
    inst.params['GENERAL.WAVE_FILE'] = str(path)
    child = jwst_nirspec.detector_children(inst)[1]
    assert child.params['GENERAL.WAVE_FILE'] == str(path)
    inst = jwst_nirspec.JWST_NIRSPEC(inst.params)
    inst.params['GENERAL.TRACES'] = [1, 2]
    with pytest.raises(SossisseConstantException, match='Ambiguous'):
        jwst_nirspec.detector_children(inst)


def test_real_child_filenames_wave_and_cube_loading(tmp_path):
    """Load real child cubes and preserve each detector's calibration times."""
    minimal = make_instrument(tmp_path)
    params = const_funcs.load_functions.load_parameters([constants.CDict])
    for key, value in minimal.params.items():
        params.set(key, value, source='test')
    params['GENERAL.ALLOW_TEMPORARY'] = False
    params['GENERAL.USE_TEMPORARY'] = False
    inst = jwst_nirspec.JWST_NIRSPEC(params)
    children = jwst_nirspec.detector_children(inst)
    inst.define_filenames()
    parent_output = inst.get_variable('OUT_SPEC_LC_FILE', 'test')
    prefix = inst.params['GENERAL.PREFIX']
    expected_output = 'OUT_PATH/target_slc_nrs1_nrs2.fits'
    assert parent_output.format(prefix=prefix).endswith(expected_output)
    paths = []
    for trace, child in children.items():
        child.define_filenames()
        expected_wavelength = np.arange(4.0) + trace
        np.testing.assert_array_equal(child.get_wavegrid(),
                          expected_wavelength)
        cube, error, dq = child.load_data_with_dq()
        assert cube.shape == (2, 2, 4)
        np.testing.assert_array_equal(cube, np.full(cube.shape, trace))
        np.testing.assert_array_equal(error, np.ones(cube.shape))
        assert not np.any(dq)
        path = child.get_variable('TEMP_INI_CUBE', 'test')
        assert path.endswith(f'temporary_initial_cube_nrs{trace}.fits')
        assert os.path.dirname(path) == child.params['PATHS.TEMP_PATH']
        paths.append(path)
        expected_times = (np.arange(1, 3) + 2400000.5
                  + (10 if trace == 2 else 0))
        np.testing.assert_array_equal(child.get_int_times(), expected_times)
    assert paths[0] != paths[1]


def test_soss_order_products_keep_ord_suffixes(tmp_path):
    """Place SOSS order labels before extensions in shared output paths."""
    params = dict()
    params['PATHS.TEMP_PATH'] = str(tmp_path)
    params['PATHS.OTHER_PATH'] = str(tmp_path)
    params['PATHS.OUT_PATH'] = str(tmp_path)
    inst = Instrument(ParamDict(params))
    inst._trace_suffix = '_ord2'
    source = str(tmp_path / 'target_slc.fits')
    assert inst.add_product_suffix(source) == (
        str(tmp_path / 'target_slc_ord2.fits'))


def test_parent_cannot_load_mixed_detector_cube(tmp_path):
    """Reject a parent cube that would concatenate simultaneous detectors."""
    inst = make_instrument(tmp_path)
    mixed_cube_kwargs = dict(match='Mixed NIRSpec detector cube')
    with pytest.raises(SossisseConstantException, **mixed_cube_kwargs):
        inst.load_data_with_dq()
    with pytest.raises(SossisseConstantException, **mixed_cube_kwargs):
        inst.load_cube(4, [2, 4], False)