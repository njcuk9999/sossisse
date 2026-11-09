#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DataCube parity and transactional checks for residual 1/f subtraction."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits

from aperocore.constants.param_functions import ParamDict

from sossisse.core import base
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments import default
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_tracemask import make_trace_case


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_subtract1f'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
ORIGINAL_POLYFIT = np.polyfit


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def reference_subtract_1f(residuals: FloatArray, cube: FloatArray,
                          error: FloatArray, trace_mask: NDArray[np.bool_],
                          degree: int, fail_column: int | None = None
                          ) -> tuple[FloatArray, FloatArray]:
    """Run the original full-cube subtraction algorithm in place.

    :param residuals: residual cube computed from the median image
    :param cube: full science cube, modified in place
    :param error: unchanged uncertainty cube
    :param trace_mask: True for pixels inside the diffraction trace
    :param degree: configured 1/f polynomial degree
    :param fail_column: optional column whose polynomial fit raises
    :return: corrected cube and full per-integration correction cube
    """
    # Allocate the full reference diagnostics as the legacy implementation did.
    scorr = np.zeros_like(cube)
    # Degree zero uses a row-median reduction independently per frame.
    if degree == 0:
        with warnings.catch_warnings(record=True):
            noise_1f = np.nanmedian(residuals, axis=1)
        # Subtract one correction value from every row in each column.
        for iframe in range(cube.shape[0]):
            for column in range(cube.shape[2]):
                scorr[iframe, :, column] = noise_1f[iframe, column]
                cube[iframe, :, column] -= noise_1f[iframe, column]
    else:
        # Fit each frame and detector column using the original weights.
        for iframe in range(cube.shape[0]):
            frame_residual = residuals[iframe]
            frame_error = np.array(error[iframe])
            for column in range(cube.shape[2]):
                column_residual = frame_residual[:, column]
                if np.sum(np.isfinite(column_residual)) < degree + 3:
                    continue
                column_error = frame_error[:, column]
                index = np.arange(frame_residual.shape[0], dtype=float)
                valid = np.isfinite(column_residual + column_error)
                valid &= ~trace_mask[:, column]
                valid &= np.abs(column_residual / column_error) < 5
                try:
                    if column == fail_column:
                        raise np.linalg.LinAlgError('injected fit failure')
                    fit = ORIGINAL_POLYFIT(
                        index[valid], column_residual[valid], degree,
                        w=1 / column_error[valid])
                    correction = np.polyval(fit, index)
                    cube[iframe, :, column] -= correction
                    scorr[iframe, :, column] = correction
                except Exception:
                    cube[iframe, :, column] = np.nan
                    scorr[iframe, :, column] = np.nan
    return cube, scorr


def reference_clean(science: FloatArray, error: FloatArray, median: FloatArray,
                    amplitudes: FloatArray, trace_mask: NDArray[np.bool_],
                    degree: int, fail_column: int | None = None
                    ) -> dict[str, FloatArray]:
    """Reproduce the original full-cube residual and subtraction stage.

    :param science: full input SCI cube
    :param error: full input ERR cube
    :param median: median image used for residual amplitudes
    :param amplitudes: mutable input amplitude vector
    :param trace_mask: True for pixels inside the diffraction trace
    :param degree: configured 1/f polynomial degree
    :param fail_column: optional polynomial-fit failure column
    :return: corrected SCI, mutated amplitudes and full scorr cube
    """
    # Preserve the input cube for the before/after comparison reference.
    cube = np.array(science, dtype=float)
    residuals = np.zeros_like(cube)
    # Recompute amplitudes in global order, mutating the caller's vector.
    for iframe in range(cube.shape[0]):
        valid = np.isfinite(cube[iframe]) & np.isfinite(median)
        valid &= trace_mask
        numerator = np.nansum(cube[iframe][valid] * median[valid])
        denominator = np.nansum(median[valid] ** 2)
        amplitudes[iframe] = numerator / denominator
        residuals[iframe] = cube[iframe] - median * amplitudes[iframe]
    # The helper intentionally mutates cube, matching the old implementation.
    corrected, scorr = reference_subtract_1f(
        residuals, cube, error, trace_mask, degree, fail_column)
    return dict(cube=corrected, amps=amplitudes, scorr=scorr)


def make_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              limit: int = 2, jump: tuple[int, int] | None = None,
              degree: int = 0) -> tuple[Instrument, DataCube, dict[str, str],
                                       tuple[FloatArray, ...], FloatArray,
                                       NDArray[np.bool_]]:
    """Build unequal input segments and a configured 1/f instrument.

    :param tmp_path: temporary root for raw, stage and median files
    :param monkeypatch: pytest fixture for isolated instrument behavior
    :param limit: maximum integrations per working product
    :param jump: optional active half-open integration selection
    :param degree: configured polynomial degree
    :return: instrument, DataCube, paths, input arrays, median and trace mask
    """
    # Reuse the loader fixture's unequal segment DataCube and source arrays.
    manager, original = make_trace_case(tmp_path, monkeypatch, limit, jump)
    science, error, quality = original
    instrument = Instrument.__new__(Instrument)
    instrument.name = 'TEST'
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.USE_TEMPORARY', False, source='test')
    instrument.params.set('WLC.INPUTS.APPLY_1F_CORR', True, source='test')
    instrument.params.set('WLC.GENERAL.DEGREE_1F_CORR', degree,
                          source='test')
    paths = dict(MEDIAN_IMAGE_FILE=str(tmp_path / 'median.fits'))
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, func: paths[key])
    # Use a detector median and a fixed trace map for reproducible amplitudes.
    with warnings.catch_warnings(record=True):
        median = np.nanmedian(science, axis=0)
    median[:, 0] = np.nan
    trace_mask = np.zeros(median.shape, dtype=bool)
    trace_mask[3:8, 2:17] = True
    return instrument, manager, paths, original, median, trace_mask


def record_plots(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Capture diagnostic plot arrays without writing figures.

    :param monkeypatch: pytest fixture restoring plot functions
    :return: mutable list containing scorr and comparison plot inputs
    """
    calls: list[tuple[Any, ...]] = []

    def record_scorr(inst: Any, first: FloatArray, last: FloatArray,
                     mean: FloatArray) -> None:
        calls.append(('scorr', first.copy(), last.copy(), mean.copy()))

    def record_comp(inst: Any, before: FloatArray, after: FloatArray) -> None:
        calls.append(('comp', before.copy(), after.copy()))

    monkeypatch.setattr(plots, 'plot_subtract_1f_scorr', record_scorr)
    monkeypatch.setattr(plots, 'plot_subtract_1f_comp', record_comp)
    return calls


def expected_for(instrument: Instrument, original: tuple[FloatArray, ...],
                 median: FloatArray, trace_mask: NDArray[np.bool_],
                 fail_column: int | None = None) -> dict[str, FloatArray]:
    """Build an independent legacy reference from fresh input copies.

    :param instrument: instrument containing the 1/f degree
    :param original: original SCI, ERR and DQ arrays
    :param median: residual reference image
    :param trace_mask: True for pixels inside the diffraction trace
    :param fail_column: optional polynomial-fit failure column
    :return: expected SCI, amplitude vector and full scorr cube
    """
    science, error, _ = original
    amplitudes = np.full(science.shape[0], -17.0)
    degree = instrument.params['WLC.GENERAL.DEGREE_1F_CORR']
    return reference_clean(science, error, median, amplitudes, trace_mask,
                           degree, fail_column)


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('jump', [None, (1, 6)])
@pytest.mark.parametrize('degree', [0, 1, 2])
def test_subtract1f_parity(tmp_path: Path,
                           monkeypatch: pytest.MonkeyPatch,
                           limit: int, jump: tuple[int, int] | None,
                           degree: int) -> None:
    """Match old full-cube math across block limits and segment boundaries."""
    case = make_case(tmp_path, monkeypatch, limit, jump, degree)
    instrument, manager, paths, original, median, trace_mask = case
    science, error, quality = original
    fail_column = 1 if degree > 0 else None
    expected = expected_for(instrument, original, median, trace_mask,
                            fail_column)
    calls = record_plots(monkeypatch)
    amplitudes = np.full(len(manager), -17.0)
    before_parents = manager.current_infiles.copy()
    if degree > 0:
        # Column zero skips, making column one the first fit in every frame.
        fit_calls = [0]

        def selective_polyfit(x: Any, y: Any, order: int,
                              **kwargs: Any) -> FloatArray:
            fail_fit = fit_calls[0] % (science.shape[2] - 1) == 0
            fit_calls[0] += 1
            if fail_fit:
                raise np.linalg.LinAlgError('injected fit failure')
            return ORIGINAL_POLYFIT(x, y, order, **kwargs)

        monkeypatch.setattr(default.np, 'polyfit', selective_polyfit)
    with warnings.catch_warnings(record=True):
        returned = instrument.clean_residual_1f(
            manager, median, amplitudes, trace_mask)
    assert returned is manager
    assert manager.current_infiles != before_parents
    assert all('_subtract_1f_' in name for name in manager.current_infiles)
    actual, actual_error, actual_quality = collect(manager)
    np.testing.assert_allclose(actual, expected['cube'], equal_nan=True)
    np.testing.assert_allclose(amplitudes, expected['amps'], equal_nan=True)
    np.testing.assert_equal(actual_error, original[1])
    np.testing.assert_equal(actual_quality, quality)
    np.testing.assert_equal(fits.getdata(paths['MEDIAN_IMAGE_FILE']), median)
    # The plot receives only bounded diagnostics and the true before frame.
    assert [call[0] for call in calls] == ['scorr', 'comp']
    with warnings.catch_warnings(record=True):
        mean_scorr = np.nanmean(expected['scorr'], axis=0)
    np.testing.assert_allclose(calls[0][1], expected['scorr'][0],
                               equal_nan=True)
    np.testing.assert_allclose(calls[0][2], expected['scorr'][-1],
                               equal_nan=True)
    np.testing.assert_allclose(calls[0][3], mean_scorr, equal_nan=True)
    np.testing.assert_equal(calls[1][1], science[0])
    np.testing.assert_allclose(calls[1][2], expected['cube'][0],
                               equal_nan=True)
    assert not np.array_equal(calls[1][1], calls[1][2], equal_nan=True)


def test_disabled_subtract1f_has_no_file_io(monkeypatch: pytest.MonkeyPatch
                                            ) -> None:
    """A disabled correction returns before planning, reading or writing."""
    instrument = Instrument.__new__(Instrument)
    instrument.name = 'TEST'
    instrument.params = ParamDict()
    instrument.params.set('WLC.INPUTS.APPLY_1F_CORR', False,
                          source='test')
    manager = DataCube(max_integrations=2)

    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        pytest.fail('disabled 1/f correction performed file I/O')

    monkeypatch.setattr(manager, 'plan_stage', forbidden)
    assert instrument.clean_residual_1f(
        manager, np.ones((2, 4)), np.ones(0), np.ones((2, 4), dtype=bool)
    ) is manager
    assert manager.current_infiles == []


@pytest.mark.parametrize('reuse,missing', [(False, False), (True, False),
                                           (True, True)])
def test_subtract1f_reuse(tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch,
                          reuse: bool, missing: bool) -> None:
    """Reuse a complete cached stage only when temporary reuse is enabled."""
    case = make_case(tmp_path, monkeypatch, limit=2, degree=0)
    instrument, manager, _, original, median, trace_mask = case
    plots_seen = record_plots(monkeypatch)
    inputs = manager.current_infiles.copy()
    with warnings.catch_warnings(record=True):
        instrument.clean_residual_1f(
            manager, median, np.full(len(manager), -1.0), trace_mask)
    outputs = manager.current_infiles.copy()
    if missing:
        Path(outputs[-1]).unlink()
    reopened = DataCube(inputs, max_integrations=2)
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    writes: list[str] = []
    original_write = reopened.write_product

    def record_write(product: Any, arrays: Any) -> None:
        writes.append(product.filename)
        original_write(product, arrays)

    monkeypatch.setattr(reopened, 'write_product', record_write)
    amplitudes = np.full(len(reopened), -5.0)
    with warnings.catch_warnings(record=True):
        instrument.clean_residual_1f(reopened, median, amplitudes, trace_mask)
    assert bool(writes) is (not reuse or missing)
    assert reopened.current_infiles == outputs
    expected = expected_for(instrument, original, median, trace_mask)
    np.testing.assert_allclose(collect(reopened)[0], expected['cube'],
                               equal_nan=True)
    np.testing.assert_allclose(amplitudes, expected['amps'], equal_nan=True)
    assert len(plots_seen) == 4


@pytest.mark.parametrize('change', ['degree', 'median'])
def test_subtract1f_signature_invalidates_reuse(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    """Degree and median bytes participate in the cached-stage signature."""
    case = make_case(tmp_path, monkeypatch, limit=2, degree=0)
    instrument, manager, _, _, median, trace_mask = case
    record_plots(monkeypatch)
    inputs = manager.current_infiles.copy()
    instrument.clean_residual_1f(manager, median, np.zeros(len(manager)),
                                 trace_mask)
    old_outputs = manager.current_infiles.copy()
    reopened = DataCube(inputs, max_integrations=2)
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    if change == 'degree':
        instrument.params.set('WLC.GENERAL.DEGREE_1F_CORR', 1,
                              source='test')
    else:
        median = median.copy()
        median[1, 1] += 1
    instrument.clean_residual_1f(reopened, median,
                                 np.zeros(len(reopened)), trace_mask)
    assert not set(old_outputs).intersection(reopened.current_infiles)


@pytest.mark.parametrize('failure', ['write', 'plot'])
def test_subtract1f_failure_does_not_commit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        failure: str) -> None:
    """A product write or diagnostic failure leaves active inputs unchanged."""
    case = make_case(tmp_path, monkeypatch, limit=2, degree=0)
    instrument, manager, _, _, median, trace_mask = case
    record_plots(monkeypatch)
    parents = manager.current_infiles.copy()
    if failure == 'write':
        def broken_write(*unused_args: Any, **unused_kwargs: Any) -> None:
            raise RuntimeError('injected subtract_1f failure')

        monkeypatch.setattr(manager, 'write_product', broken_write)
    else:
        def broken_plot(*unused_args: Any, **unused_kwargs: Any) -> None:
            raise RuntimeError('injected subtract_1f failure')

        monkeypatch.setattr(plots, 'plot_subtract_1f_scorr', broken_plot)
    with pytest.raises(RuntimeError, match='injected subtract_1f'):
        instrument.clean_residual_1f(
            manager, median, np.zeros(len(manager)), trace_mask)
    assert manager.current_infiles == parents