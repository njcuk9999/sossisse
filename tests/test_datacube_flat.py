#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flat-field parity, bounded persistence, reuse and failure regressions."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sossisse.general import plots
from sossisse.general.datacube import DataCube
from test_datacube_fits import collect
from test_datacube_loader import make_loader


def make_flat_case(tmp_path, monkeypatch, limit=2):
    """Place the global minimum in the last block and zeros in every block."""
    instrument, _ = make_loader(tmp_path, monkeypatch, limit=limit)
    science = np.arange(40.).reshape(5, 2, 4)
    error = np.full(science.shape, 8.)
    error[:, 0, 0] = 0
    error[-1, 1, 1] = 0.25
    error[0, 1, 0] = np.nan
    quality = np.arange(40.).reshape(science.shape) % 4
    quality[-1, 0, 0] = 2 ** 31 + 1
    header = fits.Header(dict(INTSTART=1, INTEND=5))
    hdus = [fits.PrimaryHDU(header=header)]
    for name, array in [('SCI', science), ('ERR', error), ('DQ', quality)]:
        hdus.append(fits.ImageHDU(array, name=name))
    filename = tmp_path / 'raw.fits'
    fits.HDUList(hdus).writeto(filename)
    instrument.params.set('GENERAL.FILES', [str(filename)], source='test')
    flat = np.array([[1., 2., np.nan, 4.], [2., 1., 4., 2.]])
    monkeypatch.setattr(instrument, 'get_flat', lambda shape: (flat, False))
    images = []

    def record_plot(inst, before, after):
        images.append((before.copy(), after.copy()))

    monkeypatch.setattr(plots, 'plot_flat_field', record_plot)
    manager = instrument.load_data_with_dq()
    return instrument, manager, (science, error, quality), flat, images


@pytest.mark.parametrize('limit', [1, 2, 100])
def test_flat_parity_and_global_zero_replacement(tmp_path, monkeypatch, limit):
    outputs = make_flat_case(tmp_path, monkeypatch, limit)
    instrument, manager, original, flat, images = outputs
    initial_files = manager.current_infiles.copy()
    result = instrument.apply_flat_field(manager)
    assert result is manager
    assert result.current_infiles != initial_files
    assert all('_flat_' in name for name in result.current_infiles)
    assert all(Path(name).exists() for name in result.current_infiles)
    expected_science = original[0] / flat
    expected_error = original[1] / flat
    minimum = np.nanmin(expected_error[expected_error != 0])
    expected_error[expected_error == 0] = minimum
    for actual, expected in zip(collect(result),
                                  (expected_science, expected_error,
                                   original[2])):
        np.testing.assert_equal(actual, expected)
        assert actual.dtype == float
    np.testing.assert_equal(images[0][0], original[0][0])
    np.testing.assert_equal(images[0][1], expected_science[0])


def test_no_flat_preserves_active_products(tmp_path, monkeypatch):
    instrument, manager, original, _, images = make_flat_case(tmp_path,
                                                             monkeypatch)
    before = manager.current_infiles.copy()
    monkeypatch.setattr(instrument, 'get_flat', lambda shape: (None, True))
    assert instrument.apply_flat_field(manager) is manager
    assert manager.current_infiles == before
    assert images == []
    for actual, expected in zip(collect(manager), original):
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize('reuse', [False, True])
def test_flat_reuse_control_and_deterministic_names(tmp_path, monkeypatch,
                                                   reuse):
    instrument, manager, _, _, images = make_flat_case(tmp_path, monkeypatch)
    instrument.apply_flat_field(manager)
    paths = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    manager = instrument.load_data_with_dq()
    writes = []
    original = DataCube.write_product

    def tracked_write(self, product, arrays):
        writes.append(product.filename)
        return original(self, product, arrays)

    monkeypatch.setattr(DataCube, 'write_product', tracked_write)
    instrument.apply_flat_field(manager)
    assert writes == ([] if reuse else paths)
    assert manager.current_infiles == paths
    assert len(images) == 2


@pytest.mark.parametrize('damage', ['missing', 'header', 'shape'])
def test_flat_requires_complete_valid_products(tmp_path, monkeypatch, damage):
    instrument, manager, original, flat, _ = make_flat_case(tmp_path,
                                                          monkeypatch)
    instrument.apply_flat_field(manager)
    paths = manager.current_infiles.copy()
    if damage == 'missing':
        Path(paths[-1]).unlink()
    else:
        with fits.open(paths[-1], mode='update') as hdul:
            if damage == 'header':
                hdul[0].header['PRODUCT'] = 'wrong-stage-identity'
            else:
                hdul['DQ'] = fits.ImageHDU(np.zeros((1, 1, 1)), name='DQ')
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    manager = instrument.load_data_with_dq()
    writes = []
    original_write = DataCube.write_product

    def tracked_write(self, product, arrays):
        writes.append(product.filename)
        return original_write(self, product, arrays)

    monkeypatch.setattr(DataCube, 'write_product', tracked_write)
    instrument.apply_flat_field(manager)
    assert writes == paths
    np.testing.assert_equal(collect(manager)[0], original[0] / flat)


def test_failed_flat_stage_does_not_commit(tmp_path, monkeypatch):
    instrument, manager, original, _, images = make_flat_case(tmp_path,
                                                            monkeypatch)
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
        instrument.apply_flat_field(manager)
    assert manager.current_infiles == before
    assert images == []
    for actual, expected in zip(collect(manager), original):
        np.testing.assert_equal(actual, expected)
    monkeypatch.setattr(manager, 'write_product', original_write)
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    instrument.apply_flat_field(manager)
    assert manager.current_infiles != before


def test_changed_flat_invalidates_reuse(tmp_path, monkeypatch):
    instrument, manager, _, flat, _ = make_flat_case(tmp_path, monkeypatch)
    instrument.apply_flat_field(manager)
    paths = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    manager = instrument.load_data_with_dq()
    monkeypatch.setattr(instrument, 'get_flat',
                        lambda shape: (flat * 2, False))
    instrument.apply_flat_field(manager)
    assert not set(paths).intersection(manager.current_infiles)


def test_plot_failure_does_not_commit(tmp_path, monkeypatch):
    instrument, manager, _, _, _ = make_flat_case(tmp_path, monkeypatch)
    before = manager.current_infiles.copy()

    def failed_plot(*plot_args):
        raise RuntimeError('plot failed')

    monkeypatch.setattr(plots, 'plot_flat_field', failed_plot)
    with pytest.raises(RuntimeError, match='plot failed'):
        instrument.apply_flat_field(manager)
    assert manager.current_infiles == before