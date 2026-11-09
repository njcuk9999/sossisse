#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration checks for the staged DataCube loader API."""

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.io.fits.hdu.image import Section

from aperocore.constants.param_functions import ParamDict
from aperocore.constants import load_functions
from sossisse.core import constants
from sossisse.core import const_funcs
from sossisse.core import exceptions
from sossisse.general import general
from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from sossisse.instruments.jwst_niriss import JWST_NIRISS_FGS
from test_datacube_fits import collect
from test_datacube_fits import write_raw


def make_loader(tmp_path, monkeypatch, limit=1, bin_size=1,
                instrument_class=Instrument):
    """Set up only the initial-loader parameters and variable paths."""
    params = ParamDict()
    params.set('GENERAL.MAX_INTEGRATIONS', limit, source='test')
    params.set('GENERAL.MAX_MATH_MEMORY_GB', 8.0, source='test')
    params.set('GENERAL.USE_TEMPORARY', False, source='test')
    params.set('WLC.LMODEL.DATA_BIN_TIME', bin_size > 1, source='test')
    params.set('WLC.LMODEL.DATA_BIN_SIZE', bin_size, source='test')
    instrument = instrument_class.__new__(instrument_class)
    instrument.params = params
    instrument._jump_slice = None
    variables = dict()
    monkeypatch.setattr(instrument, 'set_variable', variables.__setitem__)
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, func: str(tmp_path / f'{key}.fits'))
    monkeypatch.setattr(instrument, 'update_meta_data', lambda: None)

    def forbidden(*unused_args, **unused_kwargs):
        pytest.fail('The initial loader must not use whole-array loading')

    monkeypatch.setattr(instrument, 'load_data', forbidden)
    monkeypatch.setattr(instrument, 'load_cube', forbidden)
    monkeypatch.setattr(instrument, 'bin_cube', forbidden)
    return instrument, variables


def write_segment(tmp_path, start, count=3, synthetic=False):
    """Write a segment with values that expose ordering mistakes."""
    data = np.full((count, 2, 4), start, dtype=float)
    header = fits.Header(dict(INTSTART=start, INTEND=start + count - 1))
    hdus = [fits.PrimaryHDU(header=header), fits.ImageHDU(data, name='SCI')]
    if not synthetic:
        hdus.extend([fits.ImageHDU(data + 1, name='ERR'),
                     fits.ImageHDU(np.zeros_like(data), name='DQ')])
    filename = str(tmp_path / f'segment-{start}.fits')
    fits.HDUList(hdus).writeto(filename)
    return filename


@pytest.mark.parametrize('limit', [1, 6])
def test_loader_order_and_header_only_frame_count(tmp_path, monkeypatch,
                                                   limit):
    instrument, variables = make_loader(tmp_path, monkeypatch, limit)
    first = write_segment(tmp_path, 1)
    second = write_segment(tmp_path, 4)
    instrument.params.set('GENERAL.FILES', [second, first], source='test')
    original_section = Section.__getitem__
    reads = []

    def checked_section(section, selection):
        reads.append(selection)
        return original_section(section, selection)

    monkeypatch.setattr(Section, '__getitem__', checked_section)
    assert instrument.get_input_n_frames() == 6
    assert not reads
    datacube = instrument.load_data_with_dq()
    assert isinstance(datacube, DataCube)
    assert reads
    assert len(datacube.current_infiles) == (6 if limit == 1 else 2)
    assert datacube.filenames == (first, second)
    assert instrument.params['GENERAL.FILES'] == [first, second]
    assert variables['DATA_N_FRAMES'] == 6
    assert variables['DATA_Y_SIZE'] == 2
    assert variables['DATA_X_SIZE'] == 4
    blocks = [arrays['FLUX'] for _, arrays in datacube.iter_integrations()]
    actual = np.concatenate(blocks)
    np.testing.assert_equal(actual[:3], np.full((3, 2, 4), 1))
    np.testing.assert_equal(actual[3:], np.full((3, 2, 4), 4))


def test_fgs_loader_binning_and_jump(tmp_path, monkeypatch):
    loader_kwargs = dict()
    loader_kwargs['bin_size'] = 3
    loader_kwargs['instrument_class'] = JWST_NIRISS_FGS
    instrument, variables = make_loader(tmp_path, monkeypatch, **loader_kwargs)
    filename = write_segment(tmp_path, 1, count=9, synthetic=True)
    instrument.params.set('GENERAL.FILES', [filename], source='test')
    instrument._jump_slice = (1, 2)
    assert instrument.get_input_n_frames() == 3
    datacube = instrument.load_data_with_dq()
    assert variables['DATA_N_FRAMES'] == 1
    arrays = datacube.read_integrations()
    np.testing.assert_equal(arrays['FLUX'], np.full((1, 2, 4), 3))
    np.testing.assert_equal(arrays['ERR'], np.sqrt(np.full((1, 2, 4), 3)))
    np.testing.assert_equal(arrays['DQ'], np.zeros((1, 2, 4)))


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_loader_rejects_invalid_limit(tmp_path, monkeypatch, limit):
    instrument, _ = make_loader(tmp_path, monkeypatch, limit)
    with pytest.raises(exceptions.SossisseConstantException,
                       match='MAX_INTEGRATIONS'):
        instrument.load_data_with_dq()


def test_limit_default():
    constant = constants.CDict.get('GENERAL.MAX_INTEGRATIONS')
    assert constant.value == 1000
    assert constant.validate(test_value=1) == 1


@pytest.mark.parametrize('value', [True, False, 0, -1, '256', np.nan, np.inf])
def test_invalid_math_memory(tmp_path, monkeypatch, value):
    """Reject invalid YAML/loader values before numeric dtype conversion."""
    constant = constants.CDict.get('GENERAL.MAX_MATH_MEMORY_GB')
    for candidate in (constant, constant.copy(), constant.copy(source='test')):
        with pytest.raises(exceptions.SossisseConstantException,
                           match='MAX_MATH_MEMORY_GB'):
            candidate.validate(test_value=value)
    instrument, _ = make_loader(tmp_path, monkeypatch)
    instrument.params.set('GENERAL.MAX_MATH_MEMORY_GB', value, source='test')
    with pytest.raises(exceptions.SossisseConstantException,
                       match='MAX_MATH_MEMORY_GB'):
        instrument.load_data_with_dq()


def test_math_memory_default_and_loader(tmp_path, monkeypatch):
    """Retain the user budget when normalizing or reusing an initial stage."""
    constant = constants.CDict.get('GENERAL.MAX_MATH_MEMORY_GB')
    assert constant.value == 8.0
    assert constant.validate(test_value=0.25 / 1024) == 0.25 / 1024
    instrument, _ = make_loader(tmp_path, monkeypatch)
    filename = write_segment(tmp_path, 1)
    instrument.params.set('GENERAL.FILES', [filename], source='test')
    instrument.params.set('GENERAL.MAX_MATH_MEMORY_GB', 0.25 / 1024,
                          source='test')
    first = instrument.load_data_with_dq()
    assert first.max_math_memory_gb == 0.25 / 1024
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    instrument.params.set('GENERAL.MAX_MATH_MEMORY_GB', 0.5 / 1024,
                          source='test')
    second = instrument.load_data_with_dq()
    assert second.max_math_memory_gb == 0.5 / 1024
    assert second.current_infiles == first.current_infiles


@pytest.mark.parametrize('value', [True, False, 0, -1, '256', np.nan, np.inf])
def test_yaml_math_memory_rejects_invalid(tmp_path, value):
    """The actual YAML loader must not coerce invalid budgets into floats."""
    params = load_functions.load_parameters([constants.CDict])
    document = dict(GENERAL=dict(MAX_MATH_MEMORY_GB=value))
    filename = tmp_path / 'invalid-memory.yaml'
    filename.write_text(yaml.safe_dump(document))
    with pytest.raises(exceptions.SossisseConstantException,
                       match='MAX_MATH_MEMORY_GB'):
        const_funcs._load_yaml_parameters(str(filename), params)


def test_yaml_math_memory_round_trip(tmp_path):
    """Preserve memory validation in dumps, reloads and nested CDicts."""
    copied = constants.CDict.copy(source='test')
    copied.title = 'Math memory round trip'
    params = load_functions.load_parameters([copied])
    params['GENERAL.MAX_MATH_MEMORY_GB'] = 0.125 / 1024
    filename = str(tmp_path / 'memory.yaml')
    copied.save_yaml(params=params, outpath=filename, log=False)
    reloaded = const_funcs._load_yaml_parameters(filename, params)
    assert reloaded['GENERAL.MAX_MATH_MEMORY_GB'] == 0.125 / 1024
    nested = constants.ConstDict('test')
    nested.add_sub_cdict('SOSSISSE', copied, source='test')
    constant = nested.get('SOSSISSE.GENERAL.MAX_MATH_MEMORY_GB')
    assert constant.validate(test_value=0.25 / 1024) == 0.25 / 1024
    with pytest.raises(exceptions.SossisseConstantException):
        constant.validate(test_value=True)


def test_reconstruction_marks_migration_boundary(tmp_path, monkeypatch):
    instrument, _ = make_loader(tmp_path, monkeypatch)
    filename = write_segment(tmp_path, 1)
    instrument.params.set('GENERAL.FILES', [filename], source='test')
    monkeypatch.setattr(general, 'linear_recon_init', lambda inst: False)
    stages = []

    def flat_stage(datacube):
        stages.append(('flat', datacube))
        return datacube

    def dq_stage(datacube):
        stages.append(('dq', datacube))
        return datacube

    def background_stage(datacube):
        stages.append(('background', datacube))
        return datacube

    def lowpass_stage(datacube):
        stages.append(('lowpass', datacube))
        return datacube

    def isolated_stage(datacube):
        stages.append(('isolated', datacube))
        return datacube

    def cosmics_stage(datacube):
        stages.append(('cosmics', datacube))
        return datacube

    def trace_mask_stage(med=None, cube=None, log=True, no_plot=True,
                         plot_frames=None):
        stages.append(('trace_mask', cube))
        assert med is None and log and not no_plot
        assert plot_frames == [0, -1]
        return np.ones((2, 4), dtype=bool)

    def median_stack_stage(datacube):
        stages.append(('median_stack', datacube))
        return [datacube, np.ones((2, 4)), np.ones(len(datacube))]

    def residual_stage(datacube, med, amps, trace_mask):
        stages.append(('subtract_1f', datacube))
        assert med.shape == (2, 4)
        assert amps.shape == (len(datacube),)
        assert trace_mask.shape == (2, 4)
        return datacube

    def recenter_stage(trace_mask, med):
        stages.append(('recenter', stages[0][1]))
        assert trace_mask.shape == med.shape
        return trace_mask

    monkeypatch.setattr(instrument, 'apply_flat_field', flat_stage)
    monkeypatch.setattr(instrument, 'apply_dq', dq_stage)
    monkeypatch.setattr(instrument, 'remove_background', background_stage)
    monkeypatch.setattr(instrument, 'low_pass_filter', lowpass_stage)
    monkeypatch.setattr(instrument, 'patch_isolated_bads', isolated_stage)
    monkeypatch.setattr(instrument, 'remove_cosmic_rays', cosmics_stage)
    monkeypatch.setattr(instrument, 'get_trace_mask', trace_mask_stage)
    monkeypatch.setattr(instrument, 'create_median_stack',
                        median_stack_stage)
    monkeypatch.setattr(instrument, 'clean_residual_1f', residual_stage)
    monkeypatch.setattr(instrument, 'recenter_trace_position', recenter_stage)
    with pytest.raises(NotImplementedError, match='PCA and later'):
        general.linear_recon_chunk(instrument)
    assert [stage[0] for stage in stages] == [
        'flat', 'dq', 'background', 'lowpass', 'isolated', 'cosmics',
        'trace_mask', 'median_stack', 'subtract_1f', 'recenter']
    assert all(stage[1] is stages[0][1] for stage in stages)


def test_working_products_are_required_and_reuse_is_opt_in():
    assert constants.CDict.get('GENERAL.USE_TEMPORARY').value is False
    assert 'GENERAL.ALLOW_TEMPORARY' not in constants.CDict


def test_cds_settings_invalidate_normalized_reuse(tmp_path, monkeypatch):
    instrument, _ = make_loader(tmp_path, monkeypatch, limit=2)
    filename, _ = write_raw(tmp_path, cds=True)
    instrument.params.set('GENERAL.FILES', [str(filename)], source='test')
    instrument.params.set('WLC.INPUTS', dict(CDS_IDS=[0, 1], CDS_RON=4),
                          source='test')
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    first = instrument.load_data_with_dq()
    first_error = collect(first)[1]
    instrument.params['WLC.INPUTS']['CDS_RON'] = 9
    second = instrument.load_data_with_dq()
    assert not set(first.current_infiles).intersection(second.current_infiles)
    np.testing.assert_allclose(collect(second)[1] ** 2, first_error ** 2 + 5)