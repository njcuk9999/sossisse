#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exercise exact reductions through independent in-memory tile readers."""

import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
import pytest

from sossisse.core import base
from sossisse.general.blockmath import block_math, sample_reader


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_blockmath'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
@pytest.fixture
def source() -> NDArray[np.float64]:
    """Provide nontrivial samples without constructing a storage manager.

    :return: float64 (7, 5, 7) source with NaNs and opposite infinities
    """
    samples = np.random.default_rng(42).normal(size=(7, 5, 7))
    samples[:, 0, 0] = np.nan
    samples[:3, 1, 2] = np.nan
    samples[0, 2, 1] = np.inf
    samples[1, 2, 1] = -np.inf
    return samples


def _forbidden_reader(_indices: NDArray[np.intp], _spatial: tuple[slice, ...]
                      ) -> NDArray[np.float64]:
    """Fail if validation or empty-source handling attempts sample access.

    :param _indices: requested time coordinates, which must not be used
    :param _spatial: requested spatial slices, which must not be used
    :return: never returns a tile
    :raises AssertionError: on any reader call
    """
    raise AssertionError('No sample reads are allowed')


def test_default_budget() -> None:
    """Keep the reusable engine default at 8.0 GiB without large arrays.

    :return: None; assert the public signature default
    """
    parameter = inspect.signature(block_math).parameters['max_memory_gb']
    assert parameter.default == 8.0


@pytest.mark.parametrize('operation,required_bytes', [
    ('nansum', 176), ('nanmean', 176), ('nanmedian', 192),
    ('nanpercentile', 200)])
@pytest.mark.parametrize('boundary', ['below', 'exact', 'above'])
def test_binary_budget_boundary(operation: str, required_bytes: int,
                               boundary: str) -> None:
    """Respect byte-exact GiB limits without rounding up fractional bytes.

    :param operation: reduction whose conservative workspace is checked
    :param required_bytes: metadata, output and one-sample workspace bytes
    :param boundary: representable budget below, at or above the byte limit
    :return: None; assert exact preflight and callback behavior
    """
    budget = required_bytes / 1024 ** 3
    if boundary != 'exact':
        direction = 0.0 if boundary == 'below' else np.inf
        budget = float(np.nextafter(budget, direction))
    reads = 0

    def read_tile(indices: NDArray[np.intp], spatial: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Read a single sample only after successful budget preflight.

        :param indices: validated time indices
        :param spatial: bounded spatial slices
        :return: float64 one-sample source tile
        """
        nonlocal reads
        reads += 1
        return np.ones((1, 1, 1), dtype=np.float64)[(indices,) + spatial]

    memory_kwargs: dict[str, Any] = dict(max_memory_gb=budget)
    if operation == 'nanpercentile':
        memory_kwargs['q'] = 50.0
    if boundary == 'below':
        with pytest.raises(MemoryError, match='budget'):
            block_math(operation, (1, 1, 1), read_tile, **memory_kwargs)
        assert reads == 0
    else:
        actual = block_math(operation, (1, 1, 1), read_tile, **memory_kwargs)
        np.testing.assert_equal(actual, np.ones((1, 1)))
        assert reads == 1


@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                      'nanpercentile'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1, (0, 1), (1, 2), None,
                                (), (-1, 0)])
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('tight', [False, True])
def test_numpy_parity(source: NDArray[np.float64], operation: str,
                      axis: Any, keepdims: bool, tight: bool) -> None:
    """Compare exact multi-axis output and check each requested tile domain.

    :param source: independent float64 source
    :param operation: supported reduction name
    :param axis: NumPy reduced-axis specification
    :param keepdims: whether reduced axes survive as singletons
    :param tight: force multiple tiles when unreduced axes allow them
    :return: None; assert NumPy parity and bounded reader coordinates
    """
    requests: list[tuple[int, int, int]] = []
    axis0_reads: dict[tuple[int, ...], list[int]] = dict()
    axes = tuple(range(3)) if axis is None else axis
    if not isinstance(axes, tuple):
        axes = (axes,)
    axes = tuple(index % 3 for index in axes)

    def read_tile(indices: NDArray[np.intp], spatial: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Gather only requested samples and record shape, not tile storage.

        :param indices: validated selected time indices
        :param spatial: two bounded detector-coordinate slices
        :return: requested float64 samples in selected order
        """
        assert indices.dtype == np.dtype(np.intp)
        assert len(spatial) == 2
        for part, length in zip(spatial, source.shape[1:]):
            assert 0 <= part.start < part.stop <= length
            assert part.step is None
        tile = source[(indices,) + spatial]
        requests.append(tile.shape)
        key = tuple(bound for part in spatial
                    for bound in (part.start, part.stop))
        axis0_reads.setdefault(key, []).extend(indices.tolist())
        for reduced_axis in axes:
            if reduced_axis != 0:
                assert tile.shape[reduced_axis] == source.shape[reduced_axis]
        return tile

    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=keepdims)
    if operation == 'nanpercentile':
        reduce_kwargs['q'] = [10., 50., 90.]
    # A fully reduced domain cannot be tiled; give that case its full budget.
    budget = 256.0 / 1024
    if tight and len(axes) < 3:
        budget = (0.012 if axes == () else 0.004) / 1024
        if set(axes) == {0, 2}:
            budget = 0.006 / 1024
    with warnings.catch_warnings(record=True):
        actual = block_math(operation, source.shape, read_tile,
                            max_memory_gb=budget, **reduce_kwargs)
        expected = getattr(np, operation)(source, **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)
    assert actual.dtype == np.dtype(np.float64)
    assert requests
    # Axis-0 pieces of one block must together cover the whole reduced axis.
    if 0 in axes:
        for block_indices in axis0_reads.values():
            assert block_indices == list(range(source.shape[0]))
    if tight and len(axes) < 3:
        assert len(requests) > 1


@pytest.mark.parametrize('selection', [slice(None, None, -2),
                                      [5, 0, -1, 0, 3],
                                      [True, False, True, False,
                                       False, True, False], [],
                                      slice(2, 2), [False] * 7,
                                      np.array([3, 0, 2, 0], np.uint64)])
@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                      'nanpercentile'])
@pytest.mark.parametrize('axis', [0, 1, ()])
def test_selections(source: NDArray[np.float64], selection: Any,
                    operation: str, axis: Any) -> None:
    """Preserve selection order, duplicate weights and empty NumPy behavior.

    :param source: independent float64 source
    :param selection: reverse, duplicate, mask, unsigned or empty selection
    :param operation: supported reduction name
    :param axis: reduction across or outside the selected time axis
    :return: None; assert values, output shape and unmodified selection
    """
    before = np.asarray(selection).copy()
    reads = 0

    def read_tile(indices: NDArray[np.intp], spatial: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Read an arbitrary in-memory source through the callback contract.

        :param indices: validated time indices, including ordered duplicates
        :param spatial: row and column slices
        :return: selected float64 tile
        """
        nonlocal reads
        reads += 1
        return source[(indices,) + spatial]

    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=True)
    if operation == 'nanpercentile':
        reduce_kwargs.update(q=[25., 75.], method='nearest')
    with warnings.catch_warnings(record=True):
        actual = block_math(operation, source.shape, read_tile,
                            selection=selection, **reduce_kwargs)
        expected = getattr(np, operation)(source[selection], **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)
    np.testing.assert_equal(np.asarray(selection), before)
    if source[selection].size == 0:
        assert reads == 0


@pytest.mark.parametrize('shape', [(0, 0, 0), (0, 3, 4), (2, 0, 4),
                                 (2, 3, 0)])
@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                      'nanpercentile'])
@pytest.mark.parametrize('axis', [0, None, ()])
def test_empty_source(shape: tuple[int, int, int], operation: str,
                      axis: Any) -> None:
    """Use NumPy's empty-source behavior without invoking a reader.

    :param shape: valid shape containing a zero-length dimension
    :param operation: supported reduction name
    :param axis: reduced axes
    :return: None; assert exact empty-result shape and values
    """
    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=True)
    if operation == 'nanpercentile':
        reduce_kwargs['q'] = [25., 75.]
    with warnings.catch_warnings(record=True):
        actual = block_math(operation, shape, _forbidden_reader,
                            **reduce_kwargs)
        expected = getattr(np, operation)(np.empty(shape), **reduce_kwargs)
    np.testing.assert_equal(actual, expected)
    assert actual.shape == np.shape(expected)


@pytest.mark.parametrize('budget', [True, np.bool_(False), 0, -1, np.nan,
                                   np.inf, '2', 10 ** 400])
def test_invalid_budget(budget: Any) -> None:
    """Validate the standalone budget without a DataCube constructor.

    :param budget: invalid real-memory limit
    :return: None; assert early ValueError before reader access
    """
    with pytest.raises(ValueError, match='max_memory_gb'):
        block_math('nanmean', (7, 5, 7), _forbidden_reader,
                   max_memory_gb=budget)


@pytest.mark.parametrize('shape', [(1, 2), (1, 2, 3, 4), (-1, 2, 3),
                                 (True, 2, 3), (np.bool_(True), 2, 3),
                                 (1.5, 2, 3), [1, 2, 3]])
def test_invalid_shape(shape: Any) -> None:
    """Reject malformed dimensions before metadata arithmetic or I/O.

    :param shape: malformed three-dimensional source shape
    :return: None; assert early shape validation
    """
    with pytest.raises(ValueError, match='source_shape'):
        block_math('nanmean', shape, _forbidden_reader)


@pytest.mark.parametrize('options,error', [
    (dict(axis=(0, -3)), ValueError), (dict(axis=3), IndexError),
    (dict(axis=True), ValueError), (dict(q=101), ValueError),
    (dict(q=np.nan), ValueError), (dict(q=[[50]]), ValueError),
    (dict(q=50, method='invalid'), ValueError),
    (dict(selection=[True]), ValueError),
    (dict(selection=[[1]]), ValueError),
    (dict(selection=[1.5]), ValueError),
    (dict(selection=[7]), IndexError),
    (dict(selection=[-8]), IndexError),
    (dict(selection=1), ValueError),
    (dict(selection=slice(None, None, 0)), ValueError)])
def test_invalid_options(options: dict[str, Any], error: type[Exception]
                         ) -> None:
    """Validate standalone options before invoking the callback.

    :param options: invalid selection, axis or percentile options
    :param error: expected exception type
    :return: None; assert validation does not read samples
    """
    with pytest.raises(error):
        block_math('nanpercentile', (7, 5, 7), _forbidden_reader, **options)


@pytest.mark.parametrize('axis', [None, (0, 1, 2), 0])
def test_impossible_budget(axis: Any) -> None:
    """Reject an impossible output or reduced domain before sample reads.

    :param axis: full reduction or large final output
    :return: None; assert conservative memory preflight
    """
    with pytest.raises(MemoryError, match='budget'):
        block_math('nanmedian', (7, 5, 7), _forbidden_reader,
                   axis=axis, max_memory_gb=0.001 / 1024)


@pytest.mark.parametrize('options,match', [
    (dict(max_memory_gb=0.0001 / 1024), 'selection metadata'),
    (dict(selection=np.zeros(200, dtype=int)), 'selection metadata'),
    (dict(q=np.zeros(200)), 'quantile metadata')])
def test_metadata_budget(options: dict[str, Any], match: str) -> None:
    """Reserve selection and quantile metadata before allocating large maps.

    :param options: oversized metadata or undersized metadata budget
    :param match: expected memory-error category
    :return: None; assert rejection without reader access
    """
    memory_kwargs: dict[str, Any] = dict(max_memory_gb=0.004 / 1024, q=50)
    memory_kwargs.update(options)
    with pytest.raises(MemoryError, match=match):
        block_math('nanpercentile', (7, 5, 7), _forbidden_reader,
                   **memory_kwargs)


@pytest.mark.parametrize('tile', [np.ones((7, 5, 6)),
                                np.ones((7, 5, 7), dtype=np.float32),
                                [1., 2.]])
def test_bad_reader_tile(tile: Any) -> None:
    """Reject malformed callbacks without adding coercion or copy storage.

    :param tile: wrong shape, wrong dtype or non-array return value
    :return: None; assert a descriptive callback contract error
    """
    def read_tile(_indices: NDArray[np.intp], _spatial: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Return the deliberately malformed test tile.

        :param _indices: requested time indices
        :param _spatial: requested spatial slices
        :return: deliberately invalid callback value
        """
        return tile

    with pytest.raises(ValueError, match='reader must return a float64 tile'):
        block_math('nanmean', (7, 5, 7), read_tile)


def test_dispatch_and_reader_validation() -> None:
    """Reject unknown operations, irrelevant options and non-readers.

    :return: None; assert standalone dispatch validation
    """
    with pytest.raises(ValueError, match='Unsupported'):
        block_math('sum', (1, 1, 1), _forbidden_reader)
    option_cases: tuple[dict[str, Any], ...] = (dict(q=50),
                                               dict(method='nearest'))
    for options in option_cases:
        with pytest.raises(ValueError, match='only valid for nanpercentile'):
            block_math('nanmean', (1, 1, 1), _forbidden_reader, **options)
    invalid_reader: Any = None
    with pytest.raises(TypeError, match='reader must be callable'):
        block_math('nanmean', (1, 1, 1), invalid_reader)


def test_reader_exception() -> None:
    """Propagate backend failures without changing their identity or type.

    :return: None; assert the callback exception is not wrapped
    """
    failure = OSError('source unavailable')

    def read_tile(_indices: NDArray[np.intp], _spatial: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Model a failing arbitrary source.

        :param _indices: requested time indices
        :param _spatial: requested spatial slices
        :return: never returns a tile
        :raises OSError: the original backend exception
        """
        raise failure

    with pytest.raises(OSError) as caught:
        block_math('nanmean', (1, 1, 1), read_tile)
    assert caught.value is failure


@pytest.mark.parametrize('extension', [0, 1, 'SCI'])
def test_sample_reader_tiles(tmp_path: Path, extension: int | str,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Read ordered cross-file sections without loading full FITS images.

    :param tmp_path: temporary FITS directory
    :param extension: primary, numeric image or named image extension
    :param monkeypatch: replace full image access and record bounded reads
    :return: None; assert headers-only setup, ownership and section bounds
    """
    samples = np.arange(6 * 4 * 5, dtype=np.int16).reshape(6, 4, 5)
    paths = []
    parts = (samples[:2], samples[2:3], samples[3:])
    for file_index, part in enumerate(parts):
        path = tmp_path / f'{file_index}.fits'
        if extension == 0:
            fits.PrimaryHDU(part).writeto(path)
        else:
            hdus = fits.HDUList([fits.PrimaryHDU(),
                                 fits.ImageHDU(part, name='SCI')])
            hdus.writeto(path)
        paths.append(path)
    requests = []
    section_type = fits.hdu.image.Section
    original_section = section_type.__getitem__
    read_depth = 0

    def forbid_data(_hdu: Any) -> None:
        """Reject full payload access while permitting header/section reads.

        :param _hdu: selected image HDU
        :return: never returns image data
        :raises AssertionError: on any full image access
        """
        raise AssertionError('Full FITS payload access is forbidden')

    def record_section(section: Any, key: Any) -> NDArray[Any]:
        """Record each requested bounded section before forwarding it.

        :param section: Astropy image section accessor
        :param key: axis-0 coordinate and two bounded slices
        :return: original section values
        """
        nonlocal read_depth
        if read_depth == 0:
            requests.append(key)
        read_depth += 1
        try:
            return original_section(section, key)
        finally:
            read_depth -= 1

    monkeypatch.setattr(fits.PrimaryHDU, 'data', property(forbid_data))
    monkeypatch.setattr(fits.ImageHDU, 'data', property(forbid_data))
    monkeypatch.setattr(section_type, '__getitem__', record_section)
    shape, reader = sample_reader(paths, extension=extension)
    assert shape == samples.shape
    assert not requests
    paths.reverse()
    indices = np.array([4, 0, 4, 2, 5, 1, 3], dtype=np.intp)
    axis_slices = (slice(1, 3), slice(0, 2))
    actual = reader(indices, axis_slices)
    np.testing.assert_equal(actual, samples[(indices,) + axis_slices])
    assert actual.dtype == np.dtype(np.float64)
    assert actual.flags.owndata
    assert [key[0] for key in requests] == [1, 0, 1, 0, 2, 1, 0]
    assert all(key[1:] == axis_slices for key in requests)
    requests.clear()
    empty = reader(np.array([], dtype=np.intp), axis_slices)
    assert empty.shape == (0, 2, 2)
    assert empty.dtype == np.dtype(np.float64)
    assert empty.flags.owndata
    assert not requests
    actual.fill(-1)
    np.testing.assert_equal(reader(indices, axis_slices),
                            samples[(indices,) + axis_slices])


@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                      'nanpercentile'])
@pytest.mark.parametrize('axis', [0, 1, 2, (0, 1), (1, 2), None, ()])
@pytest.mark.parametrize('selection', [[5, 0, -1, 0, 2],
                                      [True, False, True, False, True, False]])
def test_sample_reader_reductions(tmp_path: Path, operation: str, axis: Any,
                                  selection: Any) -> None:
    """Match NumPy with cross-file selections and forced engine tiling.

    :param tmp_path: temporary FITS directory
    :param operation: supported reduction name
    :param axis: selected axes to reduce
    :param selection: ordered duplicates/negative indices or Boolean mask
    :return: None; assert parity and multiple tiles for surviving axes
    """
    samples = np.arange(6 * 4 * 5, dtype=np.float32).reshape(6, 4, 5)
    samples[2, 1, 1] = np.nan
    paths = []
    parts = (samples[:2], samples[2:3], samples[3:])
    for file_index, part in enumerate(parts):
        path = tmp_path / f'{file_index}.fits'
        fits.PrimaryHDU(part).writeto(path)
        paths.append(path)
    shape, reader = sample_reader(paths)
    requests = []

    def read_tile(indices: NDArray[np.intp], axis_slices: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Record requested tile shapes without retaining tile arrays.

        :param indices: normalized axis-0 coordinates
        :param axis_slices: bounded slices for axes 1 and 2
        :return: reference FITS tile
        """
        tile = reader(indices, axis_slices)
        requests.append(tile.shape)
        return tile

    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=True)
    if operation == 'nanpercentile':
        reduce_kwargs.update(q=[25., 75.], method='nearest')
    budget = (3000 if axis is not None else 20000) / 1024 ** 3
    selected = samples[selection].astype(np.float64)
    with warnings.catch_warnings(record=True):
        actual = block_math(operation, shape, read_tile, selection=selection,
                            max_memory_gb=budget, **reduce_kwargs)
        expected = getattr(np, operation)(selected, **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)
    if axis is not None:
        assert len(requests) > 1


@pytest.mark.parametrize('shapes', [[], [(2, 3)], [(1, 2, 3, 4)],
                                   [(0, 2, 3)], [(1, 0, 3)], [(1, 2, 0)],
                                   [(1, 2, 3), (2, 3, 3)],
                                   [(1, 2, 3), (2, 2, 4)]])
def test_sample_reader_invalid_shapes(tmp_path: Path,
                                      shapes: list[tuple[int, ...]]) -> None:
    """Reject invalid images before exposing a logical source reader.

    :param tmp_path: temporary FITS directory
    :param shapes: absent, non-3-D, empty or incompatible image shapes
    :return: None; assert factory validation
    """
    paths = []
    for file_index, shape in enumerate(shapes):
        path = tmp_path / f'{file_index}.fits'
        fits.PrimaryHDU(np.zeros(shape)).writeto(path)
        paths.append(path)
    with pytest.raises(ValueError):
        sample_reader(paths)


@pytest.mark.parametrize('extension,error', [('MISSING', KeyError),
                                            (1, IndexError)])
def test_sample_reader_missing_extension(tmp_path: Path, extension: int | str,
                                         error: type[Exception]) -> None:
    """Propagate explicit extension lookup failures without fallback.

    :param tmp_path: temporary FITS directory
    :param extension: missing extension name or index
    :param error: original Astropy lookup exception
    :return: None; assert missing extensions are not silently replaced
    """
    path = tmp_path / 'source.fits'
    fits.PrimaryHDU(np.ones((2, 3, 4))).writeto(path)
    with pytest.raises(error):
        sample_reader([path], extension=extension)


@pytest.mark.parametrize('fail_read', [False, True])
def test_sample_reader_closes_files(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    fail_read: bool) -> None:
    """Close header and section handles, also for scaled data or read errors.

    :param tmp_path: temporary FITS directory
    :param monkeypatch: track file handles and optionally fail section access
    :param fail_read: whether to raise an original section read exception
    :return: None; assert scaled values and closure on success or failure
    """
    samples = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / 'scaled.fits'
    hdu = fits.PrimaryHDU(samples)
    hdu.header['BSCALE'] = 2
    hdu.header['BZERO'] = 10
    hdu.writeto(path)
    opened = []
    original_open = fits.open
    failure = OSError('section read failed')

    def track_open(*args: Any, **kwargs: Any) -> Any:
        """Retain handle metadata to check closure, not image payloads.

        :param args: original FITS open positional arguments
        :param kwargs: original FITS open keyword arguments
        :return: original HDU list
        """
        hdus = original_open(*args, **kwargs)
        opened.append(hdus._file)
        return hdus

    def fail_section(_section: Any, _key: Any) -> Any:
        """Raise the same exception object during a section read.

        :param _section: unused section accessor
        :param _key: unused section coordinates
        :return: never returns values
        :raises OSError: the original failure object
        """
        raise failure

    monkeypatch.setattr(fits, 'open', track_open)
    shape, reader = sample_reader([path])
    assert shape == samples.shape
    assert len(opened) == 1
    assert all(handle.closed for handle in opened)
    if fail_read:
        section_type = fits.hdu.image.Section
        monkeypatch.setattr(section_type, '__getitem__', fail_section)
    indices = np.array([1, 0, 1], dtype=np.intp)
    axis_slices = (slice(1, 3), slice(1, 4))
    if fail_read:
        with pytest.raises(OSError) as caught:
            reader(indices, axis_slices)
        assert caught.value is failure
        assert len(opened) == 2
    else:
        actual = reader(indices, axis_slices)
        expected = 2.0 * samples[(indices,) + axis_slices] + 10.0
        np.testing.assert_equal(actual, expected)
        assert actual.flags.owndata
        assert len(opened) == 4
    assert all(handle.closed for handle in opened)


def test_independent_import() -> None:
    """Load the engine itself with storage-related imports blocked.

    Bypass the existing general package initializer, whose eager imports are
    unrelated to this module's numerical dependencies.

    :return: None; assert the engine needs only NumPy and base metadata
    """
    import_probe = '''
import builtins
import importlib.util
import sys
from typing import Any
import numpy as np
from numpy.typing import NDArray
original_import = builtins.__import__

def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Reject storage dependencies while forwarding other imports."""
    if name.startswith('astropy.io') or name.endswith('datacube'):
        raise AssertionError('Storage dependency: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('standalone_math', sys.argv[1])
assert spec is not None and spec.loader is not None
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

def read_tile(indices: NDArray[np.intp], spatial: tuple[slice, ...]
              ) -> NDArray[np.float64]:
    """Return the requested source samples without storage dependencies."""
    return source[(indices,) + spatial]

actual = engine.block_math('nanmean', source.shape, read_tile)
np.testing.assert_equal(actual, np.nanmean(source, axis=0))
'''
    module_path = sys.modules[block_math.__module__].__file__
    assert module_path is not None
    probe_args = [sys.executable, '-c', import_probe, module_path]
    completed = subprocess.run(probe_args,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================