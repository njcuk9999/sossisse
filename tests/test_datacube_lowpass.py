#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low-pass scientific parity and transactional stage persistence."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest

from sossisse.core import base
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments import default
from test_datacube_background import make_background_case
from test_datacube_fits import collect


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_lowpass'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def reference_lowpass(science: NDArray[np.float64]
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the original full-array mask, filter and plot accumulation."""
    cube = science.copy()
    with warnings.catch_warnings(record=True):
        mean = np.nanmean(cube, axis=0)
        mask = np.zeros_like(mean, dtype=float)
        for column in range(mean.shape[1]):
            cutoff = np.nanpercentile(mean[:, column], 50)
            mask[:, column] = mean[:, column] < cutoff
        mask[mask != 1] = np.nan
        correction = np.zeros_like(mean)
        for frame in cube:
            median = np.nanmedian(frame * mask, axis=0)
            lowpass = default.mp.lowpassfilter(median, 25)
            tiled = np.tile(lowpass, mean.shape[0]).reshape(mean.shape)
            frame -= tiled
            correction += tiled
    return cube, correction / cube.shape[0]


@pytest.mark.parametrize('limit', [1, 2, 20])
@pytest.mark.parametrize('budget', [0.02 / 1024, 256.0 / 1024])
@pytest.mark.parametrize('jump', [None, (1, 6)])
def test_lowpass_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                        limit: int, budget: float,
                        jump: tuple[int, int] | None) -> None:
    """Match the original after a prior stage and within active jumps."""
    instrument, manager, _, _, _ = make_background_case(
        tmp_path, monkeypatch, limit=limit, jump=jump)
    # Make the low-pass input a real prior processed stage, not raw files.
    instrument.remove_background(manager)
    science, error, quality = collect(manager)
    expected, correction = reference_lowpass(science)
    manager.max_math_memory_gb = budget
    parents = manager.current_infiles.copy()
    images = []

    def record_plot(inst: Any, before: Any, after: Any, mean: Any) -> None:
        images.append((before.copy(), after.copy(), mean.copy()))

    monkeypatch.setattr(plots, 'plot_lowpass', record_plot)
    with warnings.catch_warnings(record=True):
        assert instrument.low_pass_filter(manager) is manager
    assert not set(parents).intersection(manager.current_infiles)
    for filename in manager.current_infiles:
        assert '_background_' in filename and '_lowpass_' in filename
    actual, actual_error, actual_quality = collect(manager)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_equal(actual_error, error)
    np.testing.assert_equal(actual_quality, quality)
    np.testing.assert_equal(images[0][0], science[0])
    np.testing.assert_allclose(images[0][1], expected[0])
    np.testing.assert_allclose(images[0][2], correction)
    assert len(images) == 1


@pytest.mark.parametrize('reuse,missing', [(False, False), (True, False),
                                          (True, True)])
def test_lowpass_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                       reuse: bool, missing: bool) -> None:
    """Reuse only a complete validated stage, independent of math budget."""
    instrument, manager, original, _, _ = make_background_case(
        tmp_path, monkeypatch, limit=2)
    inputs = manager.current_infiles.copy()
    images = []

    def record_plot(inst: Any, before: Any, after: Any, mean: Any) -> None:
        images.append((before.copy(), after.copy(), mean.copy()))

    monkeypatch.setattr(plots, 'plot_lowpass', record_plot)
    with warnings.catch_warnings(record=True):
        instrument.low_pass_filter(manager)
    outputs = manager.current_infiles.copy()
    if missing:
        Path(outputs[-1]).unlink()
    reopened = DataCube(inputs, max_integrations=2,
                        max_math_memory_gb=0.02 / 1024)
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    original_write = reopened.write_product
    writes = []

    def record_write(*write_args: Any) -> None:
        writes.append(write_args[0].filename)
        original_write(*write_args)

    monkeypatch.setattr(reopened, 'write_product', record_write)
    with warnings.catch_warnings(record=True):
        instrument.low_pass_filter(reopened)
    assert bool(writes) is (not reuse or missing)
    assert reopened.current_infiles == outputs
    expected, correction = reference_lowpass(original[0])
    np.testing.assert_allclose(collect(reopened)[0], expected)
    np.testing.assert_allclose(images[-1][1], expected[0])
    np.testing.assert_allclose(images[-1][2], correction)


@pytest.mark.parametrize('failure', ['write', 'plot', 'validate'])
def test_failed_stage_not_committed(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    failure: str) -> None:
    """Do not advance the active stage after any write or diagnostic error."""
    instrument, manager, _, _, _ = make_background_case(
        tmp_path, monkeypatch, limit=2)
    inputs = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise RuntimeError('injected lowpass failure')

    monkeypatch.setattr(plots, 'plot_lowpass', lambda *plot_args: None)
    if failure == 'write':
        original_write = manager.write_product
        writes = []

        def partial_write(*write_args: Any) -> None:
            if writes:
                raise RuntimeError('injected lowpass failure')
            original_write(*write_args)
            writes.append(write_args[0].filename)

        monkeypatch.setattr(manager, 'write_product', partial_write)
    elif failure == 'plot':
        monkeypatch.setattr(plots, 'plot_lowpass', broken)
    else:
        monkeypatch.setattr(manager, 'commit_products', broken)
    with warnings.catch_warnings(record=True):
        with pytest.raises(RuntimeError, match='injected lowpass'):
            instrument.low_pass_filter(manager)
    assert manager.current_infiles == inputs


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================