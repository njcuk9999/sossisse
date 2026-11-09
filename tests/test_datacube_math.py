#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact tiled reductions, selection semantics and memory preflight checks."""

from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits
from astropy.io.fits.hdu.image import Section

from sossisse.core import base
from sossisse.general.datacube import DataCube


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_math'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def test_default_math_budget() -> None:
    """Keep empty manager construction allocation-light with an 8.0 GiB cap."""
    manager = DataCube()
    assert manager.max_math_memory_gb == 8.0
    assert manager.shape == (0, 0, 0)
    assert manager.current_infiles == []


@pytest.fixture(scope='module')
def math_case(tmp_path_factory: pytest.TempPathFactory
              ) -> tuple[DataCube, NDArray[np.float64]]:
    """Normalize unequal raw segments with NaNs and opposite infinities."""
    directory = tmp_path_factory.mktemp('math')
    science = np.random.default_rng(42).normal(size=(9, 5, 7))
    science[:4, 1, 2] = np.nan
    science[:, 0, 0] = np.nan
    science[0, 2, 1] = np.inf
    science[1, 2, 1] = -np.inf
    science[2, 2, 1] = np.inf
    files = []
    for start, stop in ((0, 4), (4, 9)):
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=header)]
        for name, offset in (('SCI', 0), ('ERR', 10), ('DQ', 20)):
            hdus.append(fits.ImageHDU(science[start:stop] + offset,
                                      name=name))
        filename = directory / f'raw{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        files.append(filename)
    manager = DataCube(max_integrations=2)
    manager.normalize(files[::-1], directory / 'initial.fits',
                      integration_slice=(1, 8))
    return manager, science[1:8]


@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                       'nanpercentile'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1, (0, 1), (1, 2), None,
                                 (), (-1, 0)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_parity(math_case: tuple[DataCube, NDArray[np.float64]],
                      operation: str, axis: Any, keepdims: bool) -> None:
    """Compare each reduction against the complete selected NumPy cube."""
    manager, science = math_case
    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=keepdims)
    if operation == 'nanpercentile':
        reduce_kwargs['q'] = [16., 50., 84.]
    with warnings.catch_warnings(record=True):
        actual = manager.block_math(operation, **reduce_kwargs)
        expected = getattr(np, operation)(science, **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)


@pytest.mark.parametrize('selection', [slice(None, None, -2),
                                       [5, 0, -1, 0, 3],
                                       [True, False, True, False,
                                        False, True, False], [],
                                       slice(2, 2), [False] * 7])
@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                       'nanpercentile'])
def test_selections(math_case: tuple[DataCube, NDArray[np.float64]],
                     selection: Any, operation: str) -> None:
    """Keep selected order and duplicate samples, including empty inputs."""
    manager, science = math_case
    reduce_kwargs: dict[str, Any] = dict(axis=0, keepdims=True)
    if operation == 'nanpercentile':
        reduce_kwargs['q'] = [25., 75.]
    with warnings.catch_warnings(record=True):
        actual = manager.block_math(operation, selection=selection,
                                    **reduce_kwargs)
        expected = getattr(np, operation)(science[selection], **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)


@pytest.mark.parametrize('data_key,offset', [('FLUX', 0), ('ERR', 10),
                                            ('DQ', 20)])
def test_methods_and_keys(math_case: tuple[DataCube, NDArray[np.float64]],
                          data_key: str, offset: int) -> None:
    """Exercise ergonomic named-payload methods and scalar percentiles."""
    manager, science = math_case
    with warnings.catch_warnings(record=True):
        for operation in ('nansum', 'nanmean', 'nanmedian'):
            actual = getattr(manager, operation)(data_key, axis=(0, 2))
            expected = getattr(np, operation)(science + offset, axis=(0, 2))
            np.testing.assert_allclose(actual, expected)
        for method in ('linear', 'nearest', 'midpoint'):
            actual = manager.nanpercentile(33, data_key, method=method)
            expected = np.nanpercentile(science + offset, 33, axis=0,
                                         method=method)
            np.testing.assert_allclose(actual, expected)


def test_nanmean_defaults_to_flux(
    math_case: tuple[DataCube, NDArray[np.float64]]) -> None:
    """The default DataCube reduction reads the FITS SCI flux payload."""
    manager, flux = math_case
    actual = manager.nanmean()
    expected = np.nanmean(flux, axis=0)
    np.testing.assert_allclose(actual, expected)


def test_block_math_rejects_legacy_cube_key(
        math_case: tuple[DataCube, NDArray[np.float64]]) -> None:
    """CUBE is not an alias for the in-memory flux payload."""
    manager, _ = math_case
    with pytest.raises(ValueError, match='data_key must be FLUX, ERR or DQ'):
        manager.nanmean('CUBE')


def test_tight_budget_sections(math_case: tuple[DataCube, NDArray[np.float64]],
                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Force spatial tiling with more selected samples than the read limit."""
    original, science = math_case
    manager = DataCube(original.current_infiles, max_integrations=2,
                       max_math_memory_gb=0.004 / 1024)
    reads = []
    original_section = Section.__getitem__

    def record_section(section: Section, selection: Any) -> Any:
        # Astropy recursively reads individual planes inside a section.
        if all(isinstance(part, slice) for part in selection):
            reads.append(selection)
        return original_section(section, selection)

    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        pytest.fail('Math must not use whole-frame integration reads')

    monkeypatch.setattr(Section, '__getitem__', record_section)
    monkeypatch.setattr(manager, 'read_integrations', forbidden)
    with warnings.catch_warnings(record=True):
        actual = manager.nanmedian()
        expected = np.nanmedian(science, axis=0)
    np.testing.assert_allclose(actual, expected)
    assert len(reads) > len(manager.current_infiles)
    for time, rows, columns in reads:
        assert time.stop - time.start <= 2
        assert (rows.stop - rows.start < science.shape[1]
                or columns.stop - columns.start < science.shape[2])


@pytest.mark.parametrize('axis', [None, (0, 1, 2), 0])
def test_impossible_budget_before_pixels(
        math_case: tuple[DataCube, NDArray[np.float64]],
        monkeypatch: pytest.MonkeyPatch, axis: Any) -> None:
    """Reject oversized output/reduced domains before reading any pixels."""
    original, _ = math_case
    manager = DataCube(original.current_infiles, max_integrations=2,
                       max_math_memory_gb=0.001 / 1024)

    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        pytest.fail('Impossible budgets must fail before pixel reads')

    monkeypatch.setattr(Section, '__getitem__', forbidden)
    with pytest.raises(MemoryError, match='budget'):
        manager.nanmedian(axis=axis)


@pytest.mark.parametrize('operation', ['nansum', 'nanmean', 'nanmedian',
                                       'nanpercentile'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1, (0, 1), (1, 2), ()])
@pytest.mark.parametrize('keepdims', [False, True])
def test_tight_multi_axis(math_case: tuple[DataCube, NDArray[np.float64]],
                          operation: str, axis: Any, keepdims: bool) -> None:
    """Force output tiling for each axis layout, including quantile axes."""
    original, science = math_case
    budget = (0.012 if axis == () else 0.004) / 1024
    manager = DataCube(original.current_infiles, max_integrations=2,
                       max_math_memory_gb=budget)
    reduce_kwargs: dict[str, Any] = dict(axis=axis, keepdims=keepdims)
    if operation == 'nanpercentile':
        reduce_kwargs['q'] = [10., 50., 90.]
    with warnings.catch_warnings(record=True):
        actual = manager.block_math(operation, **reduce_kwargs)
        expected = getattr(np, operation)(science, **reduce_kwargs)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == np.shape(expected)


def test_selected_budget_and_unsigned_indices(
        math_case: tuple[DataCube, NDArray[np.float64]]) -> None:
    """Preflight duplicate-map memory and accept valid unsigned indices."""
    original, science = math_case
    manager = DataCube(original.current_infiles, max_integrations=2,
                       max_math_memory_gb=0.004 / 1024)
    with pytest.raises(MemoryError, match='selection metadata'):
        manager.nanmean(selection=np.zeros(200, dtype=int))
    with pytest.raises(MemoryError, match='quantile metadata'):
        manager.nanpercentile(np.zeros(200))
    selection = np.array([3, 0, 2, 0], dtype=np.uint64)
    with warnings.catch_warnings(record=True):
        actual = manager.nanmean(selection=selection)
        expected = np.nanmean(science[selection], axis=0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('selection,error', [([True], ValueError),
                                             ([[1]], ValueError),
                                             ([1.5], ValueError),
                                             ([7], IndexError),
                                             ([-8], IndexError),
                                             (1, ValueError),
                                             (slice(None, None, 0),
                                              ValueError)])
def test_invalid_selection(math_case: tuple[DataCube, NDArray[np.float64]],
                            selection: Any, error: type[Exception]) -> None:
    """Reject malformed selections without silently coercing indices."""
    manager, _ = math_case
    with pytest.raises(error):
        manager.nanmean(selection=selection)


@pytest.mark.parametrize('reduce_kwargs', [dict(axis=(0, -3)), dict(axis=3),
                                          dict(axis=True), dict(q=101),
                                          dict(q=np.nan), dict(q=[[50]]),
                                          dict(q=50, method='invalid')])
def test_invalid_options(math_case: tuple[DataCube, NDArray[np.float64]],
                          reduce_kwargs: dict[str, Any]) -> None:
    """Validate axes and percentile options before FITS reads."""
    manager, _ = math_case
    with pytest.raises((ValueError, IndexError)):
        manager.block_math('nanpercentile', **reduce_kwargs)


@pytest.mark.parametrize('budget', [True, False, 0, -1, np.nan, np.inf, '2',
                                    10 ** 400])
def test_invalid_budget(budget: Any) -> None:
    """Do not coerce Boolean, string or nonfinite memory limits."""
    with pytest.raises(ValueError, match='max_math_memory_gb'):
        DataCube(max_math_memory_gb=budget)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================