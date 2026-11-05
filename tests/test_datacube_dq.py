#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared spatial DQ masks, strict configuration and staged persistence."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.hdu.image import Section
import yaml

from aperocore.constants import load_functions
from sossisse.core import base
from sossisse.core import constants
from sossisse.core import const_funcs
from sossisse.core import exceptions
from test_datacube_fits import collect
from test_datacube_loader import make_loader


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_dq'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def make_dq_case(tmp_path, monkeypatch, limit=2, selection=None):
    """Write unequal segments exposing selection-wide DQ decisions."""
    instrument, _ = make_loader(tmp_path, monkeypatch, limit=limit)
    science = np.arange(48.).reshape(6, 2, 4)
    error = np.full(science.shape, 8.)
    quality = np.zeros_like(science)
    quality[:3, 0, 0] = 2
    quality[:4, 0, 1] = 2
    quality[-1, 0, 2] = 4
    quality[0, 0, 3] = np.nan
    quality[-1, 1, 0] = np.inf
    quality[:, 1, 1] = 2
    quality[0, 1, 3] = 3
    files = []
    for start, stop in [(0, 1), (1, 6)]:
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus: list[fits.PrimaryHDU | fits.ImageHDU]
        hdus = [fits.PrimaryHDU(header=header)]
        for name, array in [('SCI', science), ('ERR', error), ('DQ', quality)]:
            hdus.append(fits.ImageHDU(array[start:stop], name=name))
        filename = tmp_path / f'raw-{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        files.append(str(filename))
    instrument.params.set('GENERAL.FILES', files[::-1], source='test')
    instrument.params.set('WLC.INPUTS.APPLY_DQ_FLAGS', True, source='test')
    instrument.params.set('WLC.INPUTS.VALID_DQ', {0: 1.0, 2: 0.5},
                          source='test')
    instrument.params.set('WLC.INPUTS.DQ_THRESHOLD', 0.0, source='test')
    instrument._jump_slice = selection
    manager = instrument.load_data_with_dq()
    active = slice(None) if selection is None else slice(*selection)
    original = tuple(array[active] for array in (science, error, quality))
    return instrument, manager, original


@pytest.mark.parametrize('limit', [1, 2, 100])
def test_shared_dq_mask_across_segments(tmp_path, monkeypatch, limit):
    """Global fractions and unknown values ignore block boundaries."""
    instrument, manager, original = make_dq_case(tmp_path, monkeypatch, limit)
    before = manager.current_infiles.copy()
    block_lengths = []
    original_blocks = manager.iter_integrations

    def bounded_blocks():
        for integration_range, arrays in original_blocks():
            block_lengths.append(arrays['DQ'].shape[0])
            assert all(arrays[name].shape[0] <= limit
                       for name in ('CUBE', 'ERR', 'DQ'))
            yield integration_range, arrays

    monkeypatch.setattr(manager, 'iter_integrations', bounded_blocks)
    assert instrument.apply_dq(manager) is manager
    assert manager.current_infiles != before
    assert all('_dq_' in name for name in manager.current_infiles)
    expected_lengths = {1: [1] * 6, 2: [1, 2, 2, 1], 100: [1, 5]}
    assert block_lengths == expected_lengths[limit] * 2
    science, error, quality = collect(manager)
    mask = np.array([[True, False, False, False],
                     [False, False, True, False]])
    expected_science = original[0].copy()
    expected_error = original[1].copy()
    expected_science[:, ~mask] = np.nan
    expected_error[:, ~mask] = np.inf
    np.testing.assert_equal(science, expected_science)
    np.testing.assert_equal(error, expected_error)
    np.testing.assert_equal(quality, original[2])


def test_disabled_dq_reads_only_switch(tmp_path, monkeypatch):
    """Disabling needs no whitelist, reuse setting or image/file access."""
    instrument, manager, _ = make_dq_case(tmp_path, monkeypatch)
    before = manager.current_infiles.copy()
    instrument.params.set('WLC.INPUTS.APPLY_DQ_FLAGS', False, source='test')
    original_read = instrument.params.rget
    reads = []

    def switch_only(key, *read_args, **read_kwargs):
        reads.append(key)
        assert key == 'WLC.INPUTS.APPLY_DQ_FLAGS'
        return original_read(key, *read_args, **read_kwargs)

    def forbidden(*unused_args, **unused_kwargs):
        pytest.fail('Disabled DQ must not access products')

    monkeypatch.setattr(instrument.params, 'rget', switch_only)
    for method in ('plan_stage', 'iter_integrations', 'commit_products'):
        monkeypatch.setattr(manager, method, forbidden)
    assert instrument.apply_dq(manager) is manager
    assert reads == ['WLC.INPUTS.APPLY_DQ_FLAGS']
    assert manager.current_infiles == before


@pytest.mark.parametrize('limits', [
    {0: 1.0, 2: 0.0}, {0: 1.0, 2: 1.0},
    {0: 0.5, 2: 0.5}, {0: 0.0, 2: 1.0},
])
def test_category_zero_one_and_inclusive_limits(tmp_path, monkeypatch,
                                               limits):
    """Limits constrain accepted categories, not the fraction of bad flags."""
    instrument, manager, original = make_dq_case(tmp_path, monkeypatch)
    instrument.params.set('WLC.INPUTS.VALID_DQ', limits, source='test')
    instrument.params.set('WLC.INPUTS.DQ_THRESHOLD', 'unused', source='test')
    quality = original[2]
    mask = np.all(np.isin(quality, list(limits)), axis=0)
    for value, fraction in limits.items():
        mask &= np.sum(quality == value, axis=0) <= fraction * len(manager)
    instrument.apply_dq(manager)
    science, error, actual_dq = collect(manager)
    np.testing.assert_equal(np.all(np.isfinite(science), axis=0), mask)
    np.testing.assert_equal(np.all(np.isfinite(error), axis=0), mask)
    np.testing.assert_equal(actual_dq, quality)


@pytest.mark.parametrize('threshold', [0.0, 0.5, 1.0])
def test_legacy_list_scalar_fallback(tmp_path, monkeypatch, threshold):
    """The default zero fallback masks everything, while one permits mixes."""
    instrument, manager, original = make_dq_case(tmp_path, monkeypatch)
    instrument.params.set('WLC.INPUTS.VALID_DQ', [0, 2], source='test')
    instrument.params.set('WLC.INPUTS.DQ_THRESHOLD', threshold, source='test')
    quality = original[2]
    mask = np.all(np.isin(quality, [0, 2]), axis=0)
    for value in (0, 2):
        mask &= np.sum(quality == value, axis=0) <= threshold * len(manager)
    instrument.apply_dq(manager)
    np.testing.assert_equal(np.isfinite(collect(manager)[0]),
                            np.broadcast_to(mask, quality.shape))
    if threshold == 0:
        assert not mask.any()


@pytest.mark.parametrize('selection', [(0, 2), (2, 6), (1, 5)])
@pytest.mark.parametrize('limit', [1, 2, 100])
def test_dq_counts_only_active_jump(tmp_path, monkeypatch, selection, limit):
    """Jump fractions exclude unselected samples and cross segment borders."""
    case_kwargs = dict(limit=limit, selection=selection)
    outputs = make_dq_case(tmp_path, monkeypatch, **case_kwargs)
    instrument, manager, original = outputs
    quality = original[2]
    mask = np.all(np.isin(quality, [0, 2]), axis=0)
    mask &= np.sum(quality == 2, axis=0) <= len(manager) * 0.5
    instrument.apply_dq(manager)
    science, error, actual_dq = collect(manager)
    np.testing.assert_equal(np.isfinite(science),
                            np.broadcast_to(mask, quality.shape))
    np.testing.assert_equal(np.isfinite(error), np.isfinite(science))
    np.testing.assert_equal(actual_dq, quality)


@pytest.mark.parametrize('reuse', [False, True])
def test_dq_reuse_skips_all_image_reads(tmp_path, monkeypatch, reuse):
    """Complete reuse hits skip both passes; reuse off always rewrites."""
    instrument, manager, _ = make_dq_case(tmp_path, monkeypatch)
    instrument.apply_dq(manager)
    paths = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    manager = instrument.load_data_with_dq()
    writes = []
    original_write = manager.write_product

    def tracked_write(product, arrays):
        writes.append(product.filename)
        return original_write(product, arrays)

    def forbidden(*unused_args, **unused_kwargs):
        pytest.fail('A complete DQ cache hit must not read image blocks')

    monkeypatch.setattr(manager, 'write_product', tracked_write)
    if reuse:
        monkeypatch.setattr(manager, 'iter_integrations', forbidden)
        monkeypatch.setattr(manager, 'read_integrations', forbidden)
        monkeypatch.setattr(Section, '__getitem__', forbidden)
    assert instrument.apply_dq(manager) is manager
    assert manager.current_infiles == paths
    assert writes == ([] if reuse else paths)


@pytest.mark.parametrize('damage', ['missing', 'header', 'shape'])
def test_dq_partial_cache_is_rewritten(tmp_path, monkeypatch, damage):
    """A cache hit requires every planned product to be valid."""
    instrument, manager, _ = make_dq_case(tmp_path, monkeypatch)
    instrument.apply_dq(manager)
    paths = manager.current_infiles.copy()
    if damage == 'missing':
        Path(paths[-1]).unlink()
    else:
        with fits.open(paths[-1], mode='update') as hdul:
            if damage == 'header':
                primary = cast(fits.PrimaryHDU, hdul[0])
                primary.header['PRODUCT'] = 'wrong-stage-identity'
            else:
                hdul['DQ'] = fits.ImageHDU(np.zeros((1, 1, 1)), name='DQ')
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    manager = instrument.load_data_with_dq()
    writes = []
    original_write = manager.write_product

    def tracked_write(product, arrays):
        writes.append(product.filename)
        return original_write(product, arrays)

    monkeypatch.setattr(manager, 'write_product', tracked_write)
    instrument.apply_dq(manager)
    assert writes == paths
    assert manager.current_infiles == paths


def test_failed_dq_stage_does_not_commit(tmp_path, monkeypatch):
    """Partial writes leave original products active and cannot be reused."""
    instrument, manager, original = make_dq_case(tmp_path, monkeypatch)
    before = manager.current_infiles.copy()
    original_write = manager.write_product
    writes = []

    def fail_second(product, arrays):
        writes.append(product.filename)
        if len(writes) == 2:
            raise OSError('simulated disk failure')
        return original_write(product, arrays)

    monkeypatch.setattr(manager, 'write_product', fail_second)
    with pytest.raises(OSError, match='disk failure'):
        instrument.apply_dq(manager)
    assert manager.current_infiles == before
    for actual, expected in zip(collect(manager), original):
        np.testing.assert_equal(actual, expected)
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    writes.clear()

    def tracked_write(product, arrays):
        writes.append(product.filename)
        return original_write(product, arrays)

    monkeypatch.setattr(manager, 'write_product', tracked_write)
    instrument.apply_dq(manager)
    assert writes == manager.current_infiles
    assert manager.current_infiles != before


def test_dq_canonical_signature_and_changed_limits(tmp_path, monkeypatch):
    """Reordering preserves reuse, while a changed fraction invalidates it."""
    instrument, manager, _ = make_dq_case(tmp_path, monkeypatch)
    instrument.apply_dq(manager)
    paths = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    instrument.params.set('WLC.INPUTS.VALID_DQ', {2: 0.5, 0: 1},
                          source='test')
    manager = instrument.load_data_with_dq()
    instrument.apply_dq(manager)
    assert manager.current_infiles == paths
    instrument.params.set('WLC.INPUTS.VALID_DQ', {0: 1, 2: 1}, source='test')
    manager = instrument.load_data_with_dq()
    instrument.apply_dq(manager)
    assert not set(paths).intersection(manager.current_infiles)
    assert np.isfinite(collect(manager)[0][:, 0, 1]).all()


@pytest.mark.parametrize('value', [[], {}, [0, 0], [1.0], {0: np.nan}])
def test_runtime_invalid_config_before_io(tmp_path, monkeypatch, value):
    """Direct ParamDict mutation cannot bypass strict runtime validation."""
    instrument, manager, _ = make_dq_case(tmp_path, monkeypatch)
    before = manager.current_infiles.copy()
    instrument.params.set('WLC.INPUTS.VALID_DQ', value, source='test')

    def forbidden(*unused_args, **unused_kwargs):
        pytest.fail('Invalid configuration must not access products')

    monkeypatch.setattr(manager, 'plan_stage', forbidden)
    with pytest.raises(exceptions.SossisseConstantException):
        instrument.apply_dq(manager)
    assert manager.current_infiles == before


@pytest.mark.parametrize('value', [
    {0: 1.0, 2: 1.0}, {0: 1.0, 2: 0.5}, [0, 2],
])
def test_dq_configuration_yaml_round_trip(tmp_path, value):
    """Actual ConstDict dump/read retains mapping keys or a legacy list."""
    copied = constants.CDict.copy(source='test')
    copied.title = 'DQ configuration round trip'
    constant = copied.get('WLC.INPUTS.VALID_DQ')
    constant.value = value
    assert constant.validate() is True
    params = load_functions.load_parameters([copied])
    assert params['WLC.INPUTS.VALID_DQ'] == value
    filename = str(tmp_path / 'dq.yaml')
    copied.save_yaml(params=params, outpath=filename, log=False)
    saved = yaml.safe_load(Path(filename).read_text())
    assert saved['WLC']['INPUTS']['VALID_DQ'] == value
    reloaded = const_funcs._load_yaml_parameters(filename, params)
    assert reloaded['WLC.INPUTS.VALID_DQ'] == value
    assert isinstance(reloaded['WLC.INPUTS.VALID_DQ'], type(value))
    nested = constants.ConstDict('test')
    nested.add_sub_cdict('SOSSISSE', copied, source='test')
    nested_constant = nested.get('SOSSISSE.WLC.INPUTS.VALID_DQ')
    assert nested_constant.validate(test_value=value) == value


@pytest.mark.parametrize('value', [
    {}, [], {'2': 1.0}, {2.0: 1.0}, {True: 1.0}, {-1: 1.0},
    {0: True}, {0: '1'}, {0: np.inf}, {0: -0.5},
])
def test_yaml_dq_rejects_noninteger_keys_and_invalid_values(tmp_path, value):
    """Raw YAML validation must not repair quoted, float or Boolean keys."""
    params = load_functions.load_parameters([constants.CDict])
    document = dict(WLC=dict(INPUTS=dict(VALID_DQ=value)))
    filename = tmp_path / 'invalid-dq.yaml'
    filename.write_text(yaml.safe_dump(document))
    with pytest.raises(exceptions.SossisseConstantException):
        const_funcs._load_yaml_parameters(str(filename), params)


def test_default_dq_configuration():
    """Defaults survive stored validation and provenance-changing copies."""
    constant = constants.CDict.get('WLC.INPUTS.VALID_DQ')
    expected = {0: 1.0, 2: 1.0}
    assert constant.dtype is dict
    assert constant.value == expected
    assert constant.validate() is True
    assert constant.true_value == expected
    for source in (None, 'test'):
        copied = constant.copy(source=source)
        assert isinstance(copied, constants.ValidDQConst)
        assert copied.value == expected
        assert copied.value is not constant.value
        assert copied.validate(test_value=[0, 2]) == [0, 2]
    threshold = constants.CDict.get('WLC.INPUTS.DQ_THRESHOLD')
    assert threshold.value == 0.0
    assert threshold.validate() is True


@pytest.mark.parametrize('value', [
    None, [], {}, [0, 0], [True], [-1], [1.0], ['2'],
    {'2': 1}, {True: 1}, {1.0: 1}, {-1: 1},
    {0: True}, {0: -0.1}, {0: 1.1}, {0: np.nan}, {0: np.inf}, {0: '1'},
])
def test_invalid_dq_configuration(value):
    """Reject coercible keys and invalid fractions instead of guessing."""
    with pytest.raises(exceptions.SossisseConstantException):
        constants.ValidDQConst.configuration(value)


@pytest.mark.parametrize('value', [
    True, '0', None, -1, 2, np.nan, np.inf, 10 ** 400,
])
def test_invalid_dq_threshold(value):
    """The scalar fallback is subject to the same strict fraction rules."""
    threshold = constants.CDict.get('WLC.INPUTS.DQ_THRESHOLD').copy('test')
    threshold.value = value
    with pytest.raises(exceptions.SossisseConstantException):
        threshold.validate()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================