#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Median-stack parity using exact DataCube medians and scratch products."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits

from sossisse.core import base
from sossisse.general.datacube import DataCube
from sossisse.instruments import default
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_tracemask import make_trace_case


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_medstack'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
StackCase = tuple[Instrument, DataCube, dict[str, Any],
                  tuple[FloatArray, ...]]
TINY_BUDGET = 0.02 / 1024
BASELINES = ['none', 'mask', 'index', 'no-has']


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def reference_median_stack(cube: FloatArray, baseline_domain: Any,
                           med_baseline: bool, has_baseline: bool
                           ) -> dict[str, FloatArray]:
    """Reproduce the original full-cube create_median_stack mathematics.

    :param cube: full active science cube (returned unnormalized)
    :param baseline_domain: Boolean mask or index vector of baseline frames
    :param med_baseline: WLC.GENERAL.MEDIAN_BASELINE setting
    :param has_baseline: HAS_BASELINE variable
    :return: original cube, final median, amplitudes and normalized cube
    """
    nframes = cube.shape[0]
    cube2 = np.array(cube, dtype=float)
    with warnings.catch_warnings(record=True) as _:
        if med_baseline and has_baseline:
            med = np.nanmedian(cube2[baseline_domain], axis=0)
        else:
            med = np.nanmedian(cube2, axis=0)
    amps = np.zeros(nframes)
    for iframe in range(nframes):
        valid = np.isfinite(cube2[iframe]) & np.isfinite(med)
        part1a = np.nansum(cube2[iframe][valid] * med[valid])
        part2a = np.nansum(med[valid] ** 2)
        amps[iframe] = part1a / part2a
        cube2[iframe] /= amps[iframe]
    with warnings.catch_warnings(record=True) as _:
        if med_baseline:
            med = np.nanmedian(cube2[baseline_domain], axis=0)
        else:
            med = np.nanmedian(cube2, axis=0)
    return dict(cube=cube, med=med, amps=amps, normalized=cube2)


def make_stack_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                    limit: int = 2, jump: tuple[int, int] | None = None,
                    baseline: str = 'mask', zero_frame: int | None = None
                    ) -> StackCase:
    """Create an instrument and DataCube ready for create_median_stack.

    :param tmp_path: pytest directory for raw, working and temporary files
    :param monkeypatch: pytest patcher for variables and loaders
    :param limit: integrations per normalized product block
    :param jump: optional active half-open integration selection
    :param baseline: none, mask, index or no-has baseline configuration
    :param zero_frame: optional raw frame index whose science is all zero
    :return: instrument, manager, variables and selected SCI/ERR/DQ arrays
    """
    trace_args = [tmp_path, monkeypatch, limit, jump, zero_frame]
    manager, original = make_trace_case(*trace_args)
    instrument = Instrument.__new__(Instrument)
    instrument.name = 'TEST'
    instrument.params = default.ParamDict()
    instrument.params.set('GENERAL.USE_TEMPORARY', False, source='test')
    instrument.params.set('WLC.LMODEL.FIT_PCA', False, source='test')
    instrument.params.set('WLC.GENERAL.MEDIAN_BASELINE', baseline != 'none',
                          source='test')
    mask = np.zeros(len(manager), dtype=bool)
    mask[[0, 1, len(manager) - 1]] = True
    domain = np.flatnonzero(mask) if baseline == 'index' else mask
    variables = dict(HAS_BASELINE=baseline != 'no-has',
                     BASELINE_DOMAIN=domain,
                     DATA_N_FRAMES=len(manager),
                     MEDIAN_IMAGE_FILE=str(tmp_path / 'median.fits'),
                     TEMP_AMP_FILE=str(tmp_path / 'amps.fits'),
                     TEMP_PCA_FILE=str(tmp_path / 'pcas.fits'),
                     CLEAN_CUBE_FILE=str(tmp_path / 'cleaned_cube.fits'))
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, func: variables[key])
    monkeypatch.setattr(instrument, 'get_baseline_params', lambda: None)
    return instrument, manager, variables, original


def reference_for(instrument: Instrument, variables: dict[str, Any],
                  science: FloatArray) -> dict[str, FloatArray]:
    """Run the reference with the case's baseline configuration.

    :param instrument: configured instrument holding MEDIAN_BASELINE
    :param variables: HAS_BASELINE and BASELINE_DOMAIN values
    :param science: active science cube
    :return: reference outputs
    """
    med_baseline = instrument.params['WLC.GENERAL.MEDIAN_BASELINE']
    reference_args = [science, variables['BASELINE_DOMAIN'], med_baseline,
                      variables['HAS_BASELINE']]
    return reference_median_stack(*reference_args)


def spy_writes(monkeypatch: pytest.MonkeyPatch, manager: DataCube
               ) -> list[str]:
    """Record product filenames written through one manager.

    :param monkeypatch: pytest patcher restoring the method
    :param manager: DataCube whose writes are recorded
    :return: mutable ordered list of written filenames
    """
    writes: list[str] = []
    original_write = manager.write_product

    def record_write(*write_args: Any) -> None:
        original_write(*write_args)
        writes.append(write_args[0].filename)

    monkeypatch.setattr(manager, 'write_product', record_write)
    return writes


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('budget', [TINY_BUDGET, 256.0 / 1024])
@pytest.mark.parametrize('jump', [None, (1, 6)])
@pytest.mark.parametrize('baseline', BASELINES)
def test_median_stack_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             limit: int, budget: float,
                             jump: tuple[int, int] | None,
                             baseline: str) -> None:
    """Match the original across blocks, tiles, segments, jumps, baselines."""
    case_args = [tmp_path, monkeypatch, limit, jump, baseline]
    instrument, manager, variables, original = make_stack_case(*case_args)
    science, error, quality = original
    expected = reference_for(instrument, variables, science)
    manager.max_math_memory_gb = budget
    parents = manager.current_infiles.copy()
    shape = manager.shape
    writes = spy_writes(monkeypatch, manager)
    tiles: dict[int, int] = dict()
    original_tile = DataCube._math_tile

    def record_tile(self: DataCube, *tile_args: Any) -> Any:
        tiles[id(self)] = tiles.get(id(self), 0) + 1
        return original_tile(self, *tile_args)

    monkeypatch.setattr(DataCube, '_math_tile', record_tile)
    with warnings.catch_warnings(record=True):
        outputs = instrument.create_median_stack(manager)
    assert isinstance(outputs, list) and len(outputs) == 3
    returned, med, amps = outputs
    # The unnormalized active stage is returned and never replaced.
    assert returned is manager
    assert manager.current_infiles == parents
    assert manager.shape == shape
    np.testing.assert_equal(collect(manager)[0], science)
    np.testing.assert_equal(med, expected['med'])
    np.testing.assert_equal(amps, expected['amps'])
    assert amps.shape == (len(manager),)
    # Both medians use the configured budget, including the scratch stage.
    assert id(manager) in tiles and len(tiles) == 2
    if budget == TINY_BUDGET:
        assert all(count > 1 for count in tiles.values())
    # Normalized scratch products: FLUX / amps, ERR and DQ unchanged.
    assert len(writes) == len(parents)
    assert all('_mednorm_' in name for name in writes)
    normalized = DataCube(writes, max_integrations=limit)
    norm_science, norm_error, norm_quality = collect(normalized)
    np.testing.assert_equal(norm_science, expected['normalized'])
    np.testing.assert_equal(norm_error, error)
    np.testing.assert_equal(norm_quality, quality)
    # Only the median image is written, exactly as before.
    np.testing.assert_equal(fits.getdata(variables['MEDIAN_IMAGE_FILE']),
                            expected['med'])
    assert not Path(variables['CLEAN_CUBE_FILE']).exists()
    assert not Path(variables['TEMP_AMP_FILE']).exists()


def test_median_stack_warning_handling(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch
                                       ) -> None:
    """Hide median warnings only; a zero amplitude still warns as before."""
    case_args = [tmp_path, monkeypatch, 2, None, 'none', 3]
    instrument, manager, variables, original = make_stack_case(*case_args)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, med, amps = instrument.create_median_stack(manager)
    with warnings.catch_warnings(record=True) as reference_caught:
        warnings.simplefilter('always')
        expected = reference_for(instrument, variables, original[0])
    messages = [str(warning.message) for warning in caught]
    reference_messages = [str(warning.message)
                          for warning in reference_caught]
    assert not any('All-NaN' in message for message in messages)
    assert any('invalid value' in message for message in messages)
    assert set(messages) == set(reference_messages)
    assert amps[3] == 0
    np.testing.assert_equal(amps, expected['amps'])
    np.testing.assert_equal(med, expected['med'])


@pytest.mark.parametrize('state', ['off', 'hit', 'missing-amps',
                                   'missing-median', 'invalid-amps',
                                   'invalid-median', 'missing-pca',
                                   'pca-hit'])
def test_median_stack_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                            state: str) -> None:
    """Reuse valid median/amplitude files only when USE_TEMPORARY is set."""
    instrument, manager, variables, original = make_stack_case(
        tmp_path, monkeypatch)
    expected = reference_for(instrument, variables, original[0])
    with warnings.catch_warnings(record=True):
        instrument.create_median_stack(manager)
    # Cached values are distinguishable from freshly computed ones.
    cached_med = expected['med'] * 2
    cached_amps = expected['amps'] * 3
    fits.writeto(variables['MEDIAN_IMAGE_FILE'], cached_med, overwrite=True)
    fits.writeto(variables['TEMP_AMP_FILE'], cached_amps, overwrite=True)
    instrument.params.set('GENERAL.USE_TEMPORARY', state != 'off',
                          source='test')
    if state in ('missing-pca', 'pca-hit'):
        instrument.params.set('WLC.LMODEL.FIT_PCA', True, source='test')
    if state == 'pca-hit':
        fits.writeto(variables['TEMP_PCA_FILE'], np.ones((2, 3)))
    if state == 'missing-amps':
        Path(variables['TEMP_AMP_FILE']).unlink()
    if state == 'missing-median':
        Path(variables['MEDIAN_IMAGE_FILE']).unlink()
    if state == 'invalid-amps':
        fits.writeto(variables['TEMP_AMP_FILE'], cached_amps[:-1],
                     overwrite=True)
    if state == 'invalid-median':
        Path(variables['MEDIAN_IMAGE_FILE']).write_bytes(b'truncated')
    parents = manager.current_infiles.copy()
    writes = spy_writes(monkeypatch, manager)
    reads = []
    original_iter = manager.iter_integrations

    def record_iter() -> Any:
        reads.append(True)
        return original_iter()

    monkeypatch.setattr(manager, 'iter_integrations', record_iter)
    with warnings.catch_warnings(record=True):
        returned, med, amps = instrument.create_median_stack(manager)
    assert returned is manager
    assert manager.current_infiles == parents
    hit = state in ('hit', 'pca-hit')
    if hit:
        # A cache hit reads neither the cube nor any product.
        np.testing.assert_equal(med, cached_med)
        np.testing.assert_equal(amps, cached_amps)
        assert not reads
    else:
        np.testing.assert_equal(med, expected['med'])
        np.testing.assert_equal(amps, expected['amps'])
        np.testing.assert_equal(fits.getdata(variables['MEDIAN_IMAGE_FILE']),
                                expected['med'])
    # Valid normalized products are rewritten only without USE_TEMPORARY.
    if state == 'off':
        assert len(writes) == len(parents)
    else:
        assert not writes
    assert not Path(variables['CLEAN_CUBE_FILE']).exists()


def test_median_stack_signature_invalidates_products(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed baseline selection writes new normalized products."""
    instrument, manager, variables, original = make_stack_case(
        tmp_path, monkeypatch)
    first_writes = spy_writes(monkeypatch, manager)
    with warnings.catch_warnings(record=True):
        instrument.create_median_stack(manager)
    # Snapshot now: the first spy stays chained under the second one.
    first_writes = first_writes.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    mask = np.zeros(len(manager), dtype=bool)
    mask[2:5] = True
    variables['BASELINE_DOMAIN'] = mask
    second_writes = spy_writes(monkeypatch, manager)
    with warnings.catch_warnings(record=True):
        _, med, amps = instrument.create_median_stack(manager)
    expected = reference_for(instrument, variables, original[0])
    assert second_writes
    assert not set(second_writes).intersection(first_writes)
    np.testing.assert_equal(med, expected['med'])
    np.testing.assert_equal(amps, expected['amps'])


@pytest.mark.parametrize('failure', ['write', 'reopen', 'median'])
def test_failed_median_stack_keeps_state(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch,
                                         failure: str) -> None:
    """A failure leaves the active stage and median file untouched."""
    instrument, manager, variables, original = make_stack_case(
        tmp_path, monkeypatch)
    parents = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise RuntimeError('injected median stack failure')

    if failure == 'write':
        original_write = manager.write_product
        writes = []

        def partial_write(*write_args: Any) -> None:
            if writes:
                raise RuntimeError('injected median stack failure')
            original_write(*write_args)
            writes.append(write_args[0].filename)

        monkeypatch.setattr(manager, 'write_product', partial_write)
    elif failure == 'reopen':
        monkeypatch.setattr(default, 'DataCube', broken)
    else:
        monkeypatch.setattr(default.fits, 'writeto', broken)
    with warnings.catch_warnings(record=True):
        with pytest.raises(RuntimeError, match='injected median stack'):
            instrument.create_median_stack(manager)
    assert manager.current_infiles == parents
    assert not Path(variables['MEDIAN_IMAGE_FILE']).exists()
    np.testing.assert_equal(collect(manager)[0], original[0])
    # A later successful run recovers the reference outputs.
    monkeypatch.undo()
    case_args = [tmp_path / 'retry', monkeypatch]
    Path(case_args[0]).mkdir()
    retry, retry_manager, retry_variables, retry_original = make_stack_case(
        *case_args)
    with warnings.catch_warnings(record=True):
        _, med, amps = retry.create_median_stack(retry_manager)
    expected = reference_for(retry, retry_variables, retry_original[0])
    np.testing.assert_equal(med, expected['med'])
    np.testing.assert_equal(amps, expected['amps'])


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
