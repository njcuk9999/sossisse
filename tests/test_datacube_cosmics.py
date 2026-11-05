#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cosmic-ray sigma-cut parity and transactional stage persistence."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest

from aperocore.constants.param_functions import ParamDict

from sossisse.core import base
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_isolated import make_patch_case


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_cosmics'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
TINY_BUDGET = 0.02 / 1024


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def reference_cosmics(science: FloatArray, sig_cut: float
                      ) -> dict[str, FloatArray]:
    """Reproduce the original full-cube percentile sigma cut.

    :param science: full active science cube, copied before masking
    :param sig_cut: upper cut in percentile-sigma units
    :return: masked cube, heat map and the first original frame
    """
    cube = science.copy()
    with warnings.catch_warnings(record=True):
        p16, p84 = np.nanpercentile(cube, [16, 84], axis=0)
    sigma = (p84 - p16) / 2
    mean = (p84 + p16) / 2
    heat_map = np.zeros(cube.shape[1:], dtype=float)
    frame_before = np.array(cube[0])
    with warnings.catch_warnings(record=True):
        for iframe in range(cube.shape[0]):
            frame = cube[iframe]
            nsig = (frame - mean) / sigma
            cmask = nsig > sig_cut
            heat_map[cmask] += 1
            cube[iframe, cmask] = np.nan
    return dict(cube=cube, heat_map=heat_map, before=frame_before)


def record_heatmap(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record heat-map plot inputs without writing figures.

    :param monkeypatch: pytest patcher restoring the plot function
    :return: mutable list of recorded heat map, before, after and labels
    """
    images: list[tuple] = []

    def record_plot(inst: Any, heat: Any, before: Any, after: Any,
                    *labels: Any) -> None:
        images.append((heat.copy(), before.copy(), after.copy(), labels))

    monkeypatch.setattr(plots, 'plot_heatmap', record_plot)
    return images


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('budget', [TINY_BUDGET, 256.0 / 1024])
@pytest.mark.parametrize('jump', [None, (1, 6)])
def test_cosmics_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                        limit: int, budget: float,
                        jump: tuple[int, int] | None) -> None:
    """Match the original across blocks, math tiles, segments and jumps."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch,
                                                    limit, jump)
    science, error, quality = original
    expected = reference_cosmics(science, 3.0)
    assert expected['heat_map'].sum() > 0
    images = record_heatmap(monkeypatch)
    manager.max_math_memory_gb = budget
    parents = manager.current_infiles.copy()
    tiles = []
    original_tile = manager._math_tile

    def record_tile(*tile_args: Any) -> Any:
        tiles.append(tile_args[1])
        return original_tile(*tile_args)

    monkeypatch.setattr(manager, '_math_tile', record_tile)
    with warnings.catch_warnings(record=True):
        assert instrument.remove_cosmic_rays(manager) is manager
    if budget == TINY_BUDGET:
        # The tight budget must force several spatial percentile tiles.
        assert len(tiles) > 1
    assert not set(parents).intersection(manager.current_infiles)
    assert all('_cosmics_' in name for name in manager.current_infiles)
    actual, actual_error, actual_quality = collect(manager)
    np.testing.assert_equal(actual, expected['cube'])
    np.testing.assert_equal(actual_error, error)
    np.testing.assert_equal(actual_quality, quality)
    assert len(images) == 1
    heat, before, after, labels = images[0]
    np.testing.assert_equal(heat, expected['heat_map'])
    np.testing.assert_equal(before, expected['before'])
    np.testing.assert_equal(after, expected['cube'][0])
    assert labels == ('cosmic rays rate', 'cosmic_rays_corr',
                      'Number of cosmic rays')


def test_cosmics_warning_handling(tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide only percentile warnings; the frame cut warns as before."""
    instrument, manager, _ = make_patch_case(tmp_path, monkeypatch)
    record_heatmap(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        instrument.remove_cosmic_rays(manager)
    messages = [str(warning.message) for warning in caught]
    assert not any('All-NaN' in message for message in messages)
    assert any('invalid value' in message for message in messages)


def test_disabled_cosmics_reads_only_enable() -> None:
    """A disabled stage requires no files, threshold or reuse settings."""
    instrument = Instrument.__new__(Instrument)
    instrument.params = ParamDict()
    instrument.params.set('WLC.GENERAL.REMOVE_COSMIC_RAYS', False,
                          source='test')
    manager = DataCube(max_integrations=2)
    assert instrument.remove_cosmic_rays(manager) is manager
    assert manager.current_infiles == []


@pytest.mark.parametrize('reuse,missing', [(False, False), (True, False),
                                          (True, True)])
def test_cosmics_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                       reuse: bool, missing: bool) -> None:
    """Reuse only a complete stage, independent of the math budget."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch)
    images = record_heatmap(monkeypatch)
    inputs = manager.current_infiles.copy()
    with warnings.catch_warnings(record=True):
        instrument.remove_cosmic_rays(manager)
    outputs = manager.current_infiles.copy()
    if missing:
        Path(outputs[-1]).unlink()
    reopened = DataCube(inputs, max_integrations=2,
                        max_math_memory_gb=TINY_BUDGET)
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    original_write = reopened.write_product
    writes = []

    def record_write(*write_args: Any) -> None:
        writes.append(write_args[0].filename)
        original_write(*write_args)

    monkeypatch.setattr(reopened, 'write_product', record_write)
    with warnings.catch_warnings(record=True):
        instrument.remove_cosmic_rays(reopened)
    assert bool(writes) is (not reuse or missing)
    assert reopened.current_infiles == outputs
    expected = reference_cosmics(original[0], 3.0)
    np.testing.assert_equal(collect(reopened)[0], expected['cube'])
    assert len(images) == 2
    np.testing.assert_equal(images[-1][0], expected['heat_map'])
    np.testing.assert_equal(images[-1][1], expected['before'])
    np.testing.assert_equal(images[-1][2], expected['cube'][0])


def test_sigma_change_invalidates_reuse(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch
                                        ) -> None:
    """A changed threshold must write new products and new values."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch)
    record_heatmap(monkeypatch)
    inputs = manager.current_infiles.copy()
    with warnings.catch_warnings(record=True):
        instrument.remove_cosmic_rays(manager)
    outputs = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    instrument.params.set('WLC.GENERAL.COSMIC_RAY_SIGMA', 1.0,
                          source='test')
    reopened = DataCube(inputs, max_integrations=2)
    with warnings.catch_warnings(record=True):
        instrument.remove_cosmic_rays(reopened)
    assert not set(reopened.current_infiles).intersection(outputs)
    expected = reference_cosmics(original[0], 1.0)
    np.testing.assert_equal(collect(reopened)[0], expected['cube'])


@pytest.mark.parametrize('failure', ['write', 'plot', 'validate'])
def test_failed_cosmics_not_committed(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch,
                                      failure: str) -> None:
    """Do not advance the active stage after any write or diagnostic error."""
    instrument, manager, _ = make_patch_case(tmp_path, monkeypatch)
    record_heatmap(monkeypatch)
    inputs = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise RuntimeError('injected cosmics failure')

    if failure == 'write':
        original_write = manager.write_product
        writes = []

        def partial_write(*write_args: Any) -> None:
            if writes:
                raise RuntimeError('injected cosmics failure')
            original_write(*write_args)
            writes.append(write_args[0].filename)

        monkeypatch.setattr(manager, 'write_product', partial_write)
    elif failure == 'plot':
        monkeypatch.setattr(plots, 'plot_heatmap', broken)
    else:
        monkeypatch.setattr(manager, 'commit_products', broken)
    with warnings.catch_warnings(record=True):
        with pytest.raises(RuntimeError, match='injected cosmics'):
            instrument.remove_cosmic_rays(manager)
    assert manager.current_infiles == inputs


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
