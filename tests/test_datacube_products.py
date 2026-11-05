#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression checks for normalized on-disk integration products."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits

from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_fits import write_raw


def test_named_read_and_iterator_payloads(tmp_path):
    """Direct reads and iteration expose aligned named float64 arrays."""
    filename, expected = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    for selection, arrays in manager.iter_integrations():
        assert isinstance(arrays, dict)
        assert set(arrays) == {'CUBE', 'ERR', 'DQ'}
        direct = manager.read_integrations(selection.start, selection.stop)
        assert isinstance(direct, dict)
        assert set(direct) == {'CUBE', 'ERR', 'DQ'}
        for name, wanted in zip(('CUBE', 'ERR', 'DQ'), expected):
            assert arrays[name].dtype == np.float64
            np.testing.assert_equal(arrays[name], wanted[selection])
            np.testing.assert_equal(arrays[name], direct[name])


@pytest.mark.parametrize('missing,extra', [
    ('CUBE', None), ('ERR', None), ('DQ', None),
    (None, 'SCI'), (None, 'EXTRA'), ('CUBE', 'SCI'),
])
def test_write_product_requires_exact_payload_keys(tmp_path, missing, extra):
    """Missing and unexpected names fail before replacing a valid file."""
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    product = manager.plan_stage('next', 'test')[0]
    arrays = manager.read_integrations(0, 2)
    manager.write_product(product, arrays)
    before = Path(product.filename).read_bytes()
    active = manager.current_infiles.copy()
    if missing is not None:
        del arrays[missing]
    if extra is not None:
        arrays[extra] = arrays['ERR']
    with pytest.raises(ValueError, match='exactly CUBE/ERR/DQ keys') as error:
        manager.write_product(product, arrays)
    if missing is not None:
        assert f"missing: ['{missing}']" in str(error.value)
    if extra is not None:
        assert f"unexpected: ['{extra}']" in str(error.value)
    assert Path(product.filename).read_bytes() == before
    assert manager.current_infiles == active


@pytest.mark.parametrize('name', ['CUBE', 'ERR', 'DQ'])
def test_write_product_rejects_named_shape(tmp_path, name):
    """Every payload shape is validated by name before creating outputs."""
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    product = manager.plan_stage('next', 'test')[0]
    arrays = manager.read_integrations(0, 2)
    arrays[name] = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match=f'Product {name} shape'):
        manager.write_product(product, arrays)
    assert not Path(product.filename).exists()


def test_write_product_preserves_inputs_and_normalizes_dtype(tmp_path):
    """Named writes ignore insertion order and retain caller-owned arrays."""
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    product = manager.plan_stage('next', 'test')[0]
    arrays = dict()
    arrays['DQ'] = np.ones(product.shape, dtype=bool)
    arrays['ERR'] = np.full(product.shape, 7, dtype=np.int16)
    arrays['CUBE'] = np.full(product.shape, 3, dtype=np.float32)
    arrays['CUBE'][0, 0, 0] = np.nan
    originals = arrays.copy()
    snapshots = dict()
    for name in ('CUBE', 'ERR', 'DQ'):
        snapshots[name] = arrays[name].copy()
        arrays[name].setflags(write=False)
    manager.write_product(product, arrays)
    assert tuple(arrays) == ('DQ', 'ERR', 'CUBE')
    with fits.open(product.filename) as hdul:
        assert len(hdul) == 4
        assert cast(fits.PrimaryHDU, hdul[0]).name == 'PRIMARY'
        extensions = [('CUBE', 'SCI'), ('ERR', 'ERR'), ('DQ', 'DQ')]
        for index, (name, extension) in enumerate(extensions, start=1):
            image = cast(fits.ImageHDU, hdul[index])
            assert image.name == extension
            assert arrays[name] is originals[name]
            assert arrays[name].dtype == snapshots[name].dtype
            np.testing.assert_equal(arrays[name], snapshots[name])
            assert image.header['BITPIX'] == -64
            np.testing.assert_equal(image.data, snapshots[name])


def test_normalized_products_preserve_segments(tmp_path):
    filenames = []
    for start, count in [(1, 3), (4, 2)]:
        header = fits.Header(dict(INTSTART=start, INTEND=start + count - 1))
        science = np.full((count, 2, 4), start, dtype=float)
        hdus = [fits.PrimaryHDU(header=header)]
        for name, array in [('SCI', science), ('ERR', science + 1),
                            ('DQ', np.zeros_like(science))]:
            hdus.append(fits.ImageHDU(array, name=name))
        filename = tmp_path / f'raw-{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        filenames.append(filename)
    manager = DataCube(max_integrations=10)
    manager.normalize(filenames[::-1], tmp_path / 'initial.fits')
    assert manager.shape == (5, 2, 4)
    assert manager.filenames == tuple(filenames)
    assert len(manager.current_infiles) == 2
    assert not hasattr(manager, 'cube')
    assert not hasattr(manager, 'in_memory')
    assert [part.stop - part.start for part, _ in
            manager.iter_integrations()] == [3, 2]


def test_reopening_requires_complete_selection(tmp_path):
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    for paths in (manager.current_infiles[1:], manager.current_infiles[:-1]):
        with pytest.raises(ValueError, match='coverage'):
            DataCube(paths, max_integrations=2)


@pytest.mark.parametrize('limit,bin_size,selection,counts', [
    (100, 1, None, [8, 6]),
    (3, 1, None, [3, 3, 2, 3, 3]),
    (3, 1, (2, 13), [3, 3, 3, 2]),
    (3, 3, (1, 3), [1, 1]),
])
def test_segment_jump_limit_boundaries(tmp_path, limit, bin_size, selection,
                                       counts):
    first, _ = write_raw(tmp_path, 'first.fits', start=1, count=8)
    second, _ = write_raw(tmp_path, 'second.fits', start=9, count=6)
    manager = DataCube(max_integrations=limit)
    normalize_kwargs = dict(bin_size=bin_size, integration_slice=selection)
    manager.normalize([second, first], tmp_path / 'ini.fits',
                      **normalize_kwargs)
    lengths = [part.stop - part.start for part, _ in
               manager.iter_integrations()]
    assert lengths == counts
    for filename in manager.current_infiles:
        header = fits.getheader(filename)
        assert header['BSTOP'] - header['BSTART'] <= limit
        assert header['GSTART'] >= header['SELSTART']
        assert header['GSTOP'] <= header['SELSTOP']
        assert header['BSTOP'] <= (
            header['SEGEND'] - header['SEGSTART'] + 1) // bin_size


def test_deterministic_detector_and_jump_products(tmp_path):
    filename, _ = write_raw(tmp_path)
    instrument = Instrument.__new__(Instrument)
    names = set()
    for detector in ('_nrs1', '_nrs2'):
        for jump in ('_jump000', '_jump001'):
            instrument._trace_suffix = detector
            instrument._jump_suffix = jump
            prefix = instrument.add_product_suffix(str(tmp_path / 'ini.fits'))
            manager = DataCube(max_integrations=2)
            manager.normalize([filename], prefix, integration_slice=(1, 4))
            paths = manager.current_infiles.copy()
            manager = DataCube(max_integrations=2)
            manager.normalize([filename], prefix, integration_slice=(1, 4))
            assert manager.current_infiles == paths
            assert all(detector + jump in path for path in paths)
            assert all('_seg000007-000011_int' in path for path in paths)
            assert not names.intersection(paths)
            names.update(paths)


@pytest.mark.parametrize('reuse', [False, True])
def test_normalized_reuse_control(tmp_path, monkeypatch, reuse):
    filename, expected = write_raw(tmp_path)
    prefix = tmp_path / 'ini.fits'
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], prefix)
    paths = manager.current_infiles.copy()
    writes = []
    original = DataCube.write_product

    def tracked_write(self, product, arrays):
        writes.append(product.filename)
        return original(self, product, arrays)

    monkeypatch.setattr(DataCube, 'write_product', tracked_write)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], prefix, reuse=reuse)
    assert writes == ([] if reuse else paths)
    assert manager.current_infiles == paths
    for actual, wanted in zip(collect(manager), expected):
        np.testing.assert_equal(actual, wanted)


@pytest.mark.parametrize('damage', ['missing', 'header', 'shape', 'truncated'])
def test_malformed_normalized_products_are_not_reused(tmp_path, monkeypatch,
                                                      damage):
    filename, expected = write_raw(tmp_path)
    prefix = tmp_path / 'ini.fits'
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], prefix)
    paths = manager.current_infiles.copy()
    damaged = paths[-1]
    if damage == 'missing':
        Path(damaged).unlink()
    elif damage == 'truncated':
        with open(damaged, 'r+b') as stream:
            stream.truncate(5760)
    else:
        with fits.open(damaged, mode='update') as hdul:
            if damage == 'header':
                hdul[0].header['GSTART'] += 1
            else:
                hdul['ERR'] = fits.ImageHDU(np.zeros((1, 1, 1)), name='ERR')
    writes = []
    original = DataCube.write_product

    def tracked_write(self, product, arrays):
        writes.append(product.filename)
        return original(self, product, arrays)

    monkeypatch.setattr(DataCube, 'write_product', tracked_write)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], prefix, reuse=True)
    assert writes == paths
    for actual, wanted in zip(collect(manager), expected):
        np.testing.assert_equal(actual, wanted)


def test_failed_stage_and_alignment_commit_are_atomic(tmp_path):
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    before = manager.current_infiles.copy()
    products = manager.plan_stage('next', 'test')
    _, arrays = next(manager.iter_integrations())
    manager.write_product(products[0], arrays)
    with pytest.raises(ValueError, match='Incomplete'):
        manager.commit_products(products)
    assert manager.current_infiles == before
    for product, (_, arrays) in zip(products, manager.iter_integrations()):
        manager.write_product(product, arrays)
    products[0].header['SOURCE'] = 'another-source'
    manager.write_product(products[0], manager.read_integrations(0, 2))
    with pytest.raises(ValueError, match='alignment'):
        manager.commit_products(products)
    assert manager.current_infiles == before


@pytest.mark.parametrize('start,stop', [(0, 6), (-1, 1), (2, 2),
                                       (0, 1.5), (True, 2)])
def test_invalid_read_requests(tmp_path, start, stop):
    filename, _ = write_raw(tmp_path)
    manager = DataCube(max_integrations=2)
    manager.normalize([filename], tmp_path / 'ini.fits')
    with pytest.raises(ValueError, match='range'):
        manager.read_integrations(start, stop)


def test_read_across_segment_boundary(tmp_path):
    first, left = write_raw(tmp_path, count=2)
    second, right = write_raw(tmp_path, 'second.fits', start=9, count=2)
    manager = DataCube(max_integrations=3)
    manager.normalize([second, first], tmp_path / 'ini.fits')
    arrays = manager.read_integrations(1, 4)
    for name, before, after in zip(('CUBE', 'ERR', 'DQ'), left, right):
        np.testing.assert_equal(arrays[name],
                                np.concatenate([before, after])[1:4])