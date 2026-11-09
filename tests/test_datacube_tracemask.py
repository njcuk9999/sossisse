#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trace-mask parity with bounded DataCube frame reads and medians."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits
from astropy.table import Table

from aperocore.constants.param_functions import ParamDict

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from sossisse.instruments.jwst_niriss import JWST_NIRISS_SOSS
from sossisse.instruments.jwst_nirspec import JWST_NIRSPEC
from test_datacube_fits import collect
from test_datacube_loader import make_loader


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_tracemask'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
TraceCase = tuple[DataCube, tuple[FloatArray, ...]]
TINY_BUDGET = 0.02 / 1024
NFRAMES = 7
YSIZE = 12
XSIZE = 20


# =============================================================================
# Define classes
# =============================================================================
class FixedTraceInstrument(Instrument):
    """Default instrument returning a fixed map and recording inputs."""

    def get_trace_positions(self, med: Any = None, cube: Any = None,
                            log: bool = True) -> NDArray[np.bool_]:
        """Record the forwarded inputs and return the fixed trace map.

        :param med: forwarded median image or None
        :param cube: forwarded DataCube or None
        :param log: forwarded logging switch
        :return: copy of the fixed Boolean trace map
        """
        self.position_calls.append(dict(med=med, cube=cube, log=log))
        return self.fixed_mask.copy()


# =============================================================================
# Define functions
# =============================================================================
def make_trace_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                    limit: int = 2, jump: tuple[int, int] | None = None,
                    zero_frame: int | None = None) -> TraceCase:
    """Write unequal segments containing a tilted trace and bad pixels.

    :param tmp_path: pytest directory for raw and working products
    :param monkeypatch: pytest patcher used by the loader fixture
    :param limit: integrations per normalized product block
    :param jump: optional active half-open integration selection
    :param zero_frame: optional raw frame index whose science is all zero
    :return: normalized manager and the selected SCI/ERR/DQ arrays
    """
    loader, _ = make_loader(tmp_path, monkeypatch, limit=limit)
    generator = np.random.default_rng(23)
    rows, cols = np.indices((YSIZE, XSIZE))
    centre = 4.0 + 0.15 * cols
    profile = 200 * np.exp(-0.5 * ((rows - centre) / 1.5) ** 2) + 10
    scale = 1 + 0.05 * np.sin(np.arange(NFRAMES))
    noise = generator.normal(0, 2, (NFRAMES, YSIZE, XSIZE))
    science = scale[:, None, None] * profile + noise
    # An all-NaN pixel makes every median warn about an All-NaN slice.
    science[:, 0, 9] = np.nan
    science[2, 4, 6] = np.nan
    science[5, 7, 3] = np.inf
    if zero_frame is not None:
        science[zero_frame] = 0.0
    error = generator.uniform(1, 3, science.shape)
    quality = np.arange(science.size, dtype=float).reshape(science.shape)
    files = []
    for start, stop in [(0, 3), (3, NFRAMES)]:
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=header)]
        for name, array in [('SCI', science), ('ERR', error),
                            ('DQ', quality)]:
            hdus.append(fits.ImageHDU(array[start:stop], name=name))
        filename = tmp_path / f'trace_segment{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        files.append(str(filename))
    loader.params.set('GENERAL.FILES', files[::-1], source='test')
    loader._jump_slice = jump
    manager = loader.load_data_with_dq()
    original = (science, error, quality)
    if jump is not None:
        original = tuple(array[slice(*jump)] for array in original)
    return manager, original


def make_trace_instrument(instrument_class: type, tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch,
                          name: str = 'pos.fits') -> Any:
    """Create a minimal instrument with trace and size settings.

    :param instrument_class: Instrument subclass to construct without init
    :param tmp_path: pytest directory for the position file
    :param monkeypatch: pytest patcher for variables and wave grids
    :param name: basename of the (possibly absent) POS_FILE
    :return: configured instrument instance
    """
    instrument = instrument_class.__new__(instrument_class)
    instrument.name = 'TEST'
    params = ParamDict()
    params.set('WLC.GENERAL.LINRECON_TRACE_WIDTH', 5, source='test')
    params.set('WLC.GENERAL.X_TRACE_OFFSET', 0, source='test')
    params.set('WLC.GENERAL.Y_TRACE_OFFSET', 0, source='test')
    params.set('WLC.GENERAL.TRACE_WIDTH_MASKING', 4, source='test')
    params.set('GENERAL.WLC_DOMAIN', None, source='test')
    params.set('GENERAL.POS_FILE', str(tmp_path / name), source='test')
    instrument.params = params
    variables = dict(DATA_X_SIZE=XSIZE, DATA_Y_SIZE=YSIZE)
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, func: variables[key])
    wavegrid = np.linspace(1.0, 5.0, XSIZE)

    def get_wavegrid(*unused_args: Any, return_xpix: bool = False,
                     **unused_kwargs: Any) -> Any:
        if return_xpix:
            return np.arange(XSIZE), wavegrid.copy()
        return wavegrid.copy()

    monkeypatch.setattr(instrument, 'get_wavegrid', get_wavegrid)
    instrument.position_calls = []
    instrument.fixed_mask = np.zeros((YSIZE, XSIZE), dtype=bool)
    instrument.fixed_mask[3:8, 2:17] = True
    return instrument


def reference_trace_mask(trace_mask: NDArray[np.bool_],
                         wavegrid: FloatArray | None,
                         domain: tuple[float, float] | None,
                         cube: FloatArray | None, med: FloatArray | None,
                         plot_frames: list[int] | None
                         ) -> tuple[NDArray[np.bool_], Any, Any]:
    """Reproduce the original full-cube get_trace_mask after positions.

    :param trace_mask: instrument trace position map
    :param wavegrid: wavelength per column, used only with a domain
    :param domain: optional (low, high) WLC wavelength domain
    :param cube: full active science cube, or None
    :param med: median image used when cube is None
    :param plot_frames: optional plotted integration indices
    :return: final mask, plotted images and labels
    """
    trace_mask = trace_mask.copy()
    if domain is not None:
        wavelow, wavehigh = domain
        trace_mask[:, wavegrid < wavelow] = False
        trace_mask[:, wavegrid > wavehigh] = False
        trace_mask[:, ~np.isfinite(wavegrid)] = False
    if plot_frames is not None:
        if cube is None:
            images = [med]
            labels = ['Median image']
        else:
            images = [cube[it] for it in plot_frames]
            labels = [f'Frame{it}' for it in plot_frames]
    else:
        images, labels = None, None
    return trace_mask, images, labels


def reference_nirspec_median(cube: FloatArray) -> FloatArray:
    """Reproduce the original NIRSpec full-cube median (warnings shown).

    :param cube: full active science cube
    :return: per-pixel NaN median over every integration
    """
    return np.nanmedian(cube, axis=0)


def record_trace_plots(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record trace-mask plot inputs without writing figures.

    :param monkeypatch: pytest patcher restoring the plot function
    :return: mutable list of recorded mask, images and labels
    """
    calls: list[tuple] = []

    def record_plot(inst: Any, trace_map: Any, images: Any,
                    labels: Any) -> None:
        copied = None if images is None else [np.array(image)
                                              for image in images]
        calls.append((np.array(trace_map), copied, labels))

    monkeypatch.setattr(plots, 'plot_trace_mask', record_plot)
    return calls


def spy_reads(monkeypatch: pytest.MonkeyPatch, manager: DataCube
              ) -> list[tuple[int, int]]:
    """Record explicit integration reads on one manager.

    :param monkeypatch: pytest patcher restoring the method
    :param manager: DataCube whose reads are recorded
    :return: mutable list of requested (start, stop) ranges
    """
    reads: list[tuple[int, int]] = []
    original_read = manager.read_integrations

    def record_read(start: Any = 0, stop: Any = None) -> Any:
        reads.append((start, stop))
        return original_read(start, stop)

    monkeypatch.setattr(manager, 'read_integrations', record_read)
    return reads


def forbid_math(monkeypatch: pytest.MonkeyPatch, manager: DataCube) -> None:
    """Fail if the manager performs block math or iterates its blocks.

    :param monkeypatch: pytest patcher restoring the methods
    :param manager: DataCube that must not be reduced
    """
    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        pytest.fail('Unexpected full-time-series DataCube access')

    monkeypatch.setattr(manager, 'block_math', forbidden)
    monkeypatch.setattr(manager, 'iter_integrations', forbidden)


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('jump', [None, (1, 6)])
@pytest.mark.parametrize('frames', [[0, -1], [1, -2], [-1, 0]])
def test_trace_mask_plot_frames_parity(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch,
                                       limit: int,
                                       jump: tuple[int, int] | None,
                                       frames: list[int]) -> None:
    """Plot only the requested frames, identical to cube indexing."""
    manager, original = make_trace_case(tmp_path, monkeypatch, limit, jump)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    calls = record_trace_plots(monkeypatch)
    reads = spy_reads(monkeypatch, manager)
    forbid_math(monkeypatch, manager)
    parents = manager.current_infiles.copy()
    mask = instrument.get_trace_mask(cube=manager, no_plot=False,
                                     plot_frames=frames)
    reference_args = [instrument.fixed_mask, None, None, original[0], None,
                      frames]
    expected = reference_trace_mask(*reference_args)
    np.testing.assert_equal(mask, expected[0])
    assert len(instrument.position_calls) == 1
    assert instrument.position_calls[0]['cube'] is manager
    assert instrument.position_calls[0]['med'] is None
    assert len(calls) == 1
    np.testing.assert_equal(calls[0][0], expected[0])
    assert calls[0][2] == expected[2]
    assert len(calls[0][1]) == len(expected[1])
    for actual_image, expected_image in zip(calls[0][1], expected[1]):
        np.testing.assert_equal(actual_image, expected_image)
    # Each plotted frame reads exactly one integration.
    nframes = len(manager)
    expected_reads = [(it % nframes, it % nframes + 1) for it in frames]
    assert reads == expected_reads
    assert manager.current_infiles == parents


def test_trace_mask_first_and_last_frames(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch
                                          ) -> None:
    """The pipeline's [0, -1] request plots the first and last frames."""
    manager, original = make_trace_case(tmp_path, monkeypatch, 2, (1, 6))
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    calls = record_trace_plots(monkeypatch)
    instrument.get_trace_mask(cube=manager, no_plot=False,
                              plot_frames=[0, -1])
    images, labels = calls[0][1], calls[0][2]
    assert labels == ['Frame0', 'Frame-1']
    np.testing.assert_equal(images[0], original[0][0])
    np.testing.assert_equal(images[1], original[0][-1])


def test_trace_mask_without_plot_reads_nothing(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch
                                               ) -> None:
    """No plot (default) or no plot frames never read the DataCube."""
    manager, _ = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    calls = record_trace_plots(monkeypatch)
    reads = spy_reads(monkeypatch, manager)
    forbid_math(monkeypatch, manager)
    instrument.get_trace_mask(cube=manager, plot_frames=[0, -1])
    assert not calls
    instrument.get_trace_mask(cube=manager, no_plot=False)
    assert calls[0][1] is None and calls[0][2] is None
    assert not reads


def test_trace_mask_median_image(tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A med-only call keeps plotting the median image."""
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    calls = record_trace_plots(monkeypatch)
    med = np.arange(YSIZE * XSIZE, dtype=float).reshape(YSIZE, XSIZE)
    instrument.get_trace_mask(med=med, no_plot=False, plot_frames=[0, -1])
    assert instrument.position_calls[0]['med'] is med
    assert instrument.position_calls[0]['cube'] is None
    assert calls[0][2] == ['Median image']
    np.testing.assert_equal(calls[0][1][0], med)


def test_trace_mask_wavelength_domain(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch
                                      ) -> None:
    """The WLC wavelength cut matches the original full-cube function."""
    manager, original = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    instrument.params['GENERAL.WLC_DOMAIN'] = [2.0, 4.0]
    wavegrid = instrument.get_wavegrid()
    wavegrid[5] = np.nan
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: wavegrid)
    calls = record_trace_plots(monkeypatch)
    mask = instrument.get_trace_mask(cube=manager, no_plot=False,
                                     plot_frames=[0, -1])
    reference_args = [instrument.fixed_mask, wavegrid, (2.0, 4.0),
                      original[0], None, [0, -1]]
    expected = reference_trace_mask(*reference_args)
    np.testing.assert_equal(mask, expected[0])
    assert not mask[:, 5].any()
    np.testing.assert_equal(calls[0][1][1], expected[1][1])


def test_trace_width_disabled(tmp_path: Path,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A width below one returns ones before touching positions or data."""
    manager, _ = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    instrument.params['WLC.GENERAL.LINRECON_TRACE_WIDTH'] = 0
    calls = record_trace_plots(monkeypatch)
    reads = spy_reads(monkeypatch, manager)
    mask = instrument.get_trace_mask(cube=manager, no_plot=False,
                                     plot_frames=[0, -1])
    np.testing.assert_equal(mask, np.ones((YSIZE, XSIZE), dtype=bool))
    assert not instrument.position_calls and not calls and not reads


@pytest.mark.parametrize('frame', [7, -8])
def test_trace_mask_out_of_range_frame(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch,
                                       frame: int) -> None:
    """Invalid plot frames raise IndexError like cube indexing did."""
    manager, _ = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    calls = record_trace_plots(monkeypatch)
    with pytest.raises(IndexError):
        instrument.get_trace_mask(cube=manager, no_plot=False,
                                  plot_frames=[0, frame])
    assert not calls


def test_trace_mask_plot_failure_keeps_manager(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch
                                               ) -> None:
    """A plot error propagates and leaves the active stage unchanged."""
    manager, original = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(FixedTraceInstrument, tmp_path,
                                       monkeypatch)
    parents = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> None:
        raise RuntimeError('injected trace plot failure')

    monkeypatch.setattr(plots, 'plot_trace_mask', broken)
    with pytest.raises(RuntimeError, match='injected trace plot'):
        instrument.get_trace_mask(cube=manager, no_plot=False,
                                  plot_frames=[0, -1])
    assert manager.current_infiles == parents
    np.testing.assert_equal(collect(manager)[0], original[0])


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('budget', [TINY_BUDGET, 256.0 / 1024])
@pytest.mark.parametrize('jump', [None, (1, 6)])
def test_nirspec_positions_parity(tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch,
                                  limit: int, budget: float,
                                  jump: tuple[int, int] | None) -> None:
    """NIRSpec automatic positions match the full-cube median reference."""
    manager, original = make_trace_case(tmp_path, monkeypatch, limit, jump)
    manager.max_math_memory_gb = budget
    parents = manager.current_infiles.copy()
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    reference = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch,
                                      name='reference_pos.fits')
    tiles = []
    original_tile = manager._math_tile

    def record_tile(*tile_args: Any) -> Any:
        tiles.append(tile_args[1])
        return original_tile(*tile_args)

    monkeypatch.setattr(manager, '_math_tile', record_tile)
    with warnings.catch_warnings(record=True):
        mask = instrument.get_trace_positions(cube=manager)
        med = reference_nirspec_median(original[0])
        expected = reference.get_trace_positions(med=med)
    if budget == TINY_BUDGET:
        # The tight budget must force several spatial median tiles.
        assert len(tiles) > 1
    else:
        assert tiles
    np.testing.assert_equal(mask, expected)
    actual_table = Table.read(instrument.params['GENERAL.POS_FILE'])
    expected_table = Table.read(reference.params['GENERAL.POS_FILE'])
    for column in ('X', 'Y', 'WAVELENGTH'):
        np.testing.assert_equal(np.array(actual_table[column]),
                                np.array(expected_table[column]))
    assert manager.current_infiles == parents


def test_nirspec_trace_mask_plots_frames(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch
                                         ) -> None:
    """The NIRSpec pipeline call plots the first and last integrations."""
    manager, original = make_trace_case(tmp_path, monkeypatch, 2, (1, 6))
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    calls = record_trace_plots(monkeypatch)
    with warnings.catch_warnings(record=True):
        mask = instrument.get_trace_mask(cube=manager, no_plot=False,
                                         plot_frames=[0, -1])
    assert mask.any()
    assert calls[0][2] == ['Frame0', 'Frame-1']
    np.testing.assert_equal(calls[0][1][0], original[0][0])
    np.testing.assert_equal(calls[0][1][1], original[0][-1])


def test_nirspec_med_or_existing_pos_skip_cube(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch
                                               ) -> None:
    """A supplied median or existing POS_FILE never reduces the cube."""
    manager, original = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    forbid_math(monkeypatch, manager)
    reads = spy_reads(monkeypatch, manager)
    with warnings.catch_warnings(record=True):
        med = reference_nirspec_median(original[0])
    first = instrument.get_trace_positions(med=med, cube=manager)
    # The position file now exists, so neither input is required.
    second = instrument.get_trace_positions(cube=manager)
    np.testing.assert_equal(first, second)
    assert not reads


def test_nirspec_requires_med_or_cube(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch
                                      ) -> None:
    """Without a position file, a median or DataCube is required."""
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    with pytest.raises(exceptions.SossisseException, match='med/cube'):
        instrument.get_trace_positions()


def test_nirspec_warning_handling(tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The NIRSpec median keeps the original unsuppressed warnings."""
    manager, _ = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        instrument.get_trace_positions(cube=manager)
    messages = [str(warning.message) for warning in caught]
    assert any('All-NaN' in message for message in messages)


def test_nirspec_plot_failure_keeps_manager(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch
                                            ) -> None:
    """The position file is written first, as before; the stage is kept."""
    manager, _ = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(JWST_NIRSPEC, tmp_path, monkeypatch)
    parents = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> None:
        raise RuntimeError('injected trace plot failure')

    monkeypatch.setattr(plots, 'plot_trace_mask', broken)
    with warnings.catch_warnings(record=True):
        with pytest.raises(RuntimeError, match='injected trace plot'):
            instrument.get_trace_mask(cube=manager, no_plot=False,
                                      plot_frames=[0, -1])
    assert Path(instrument.params['GENERAL.POS_FILE']).exists()
    assert manager.current_infiles == parents


def test_niriss_positions_ignore_datacube(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch
                                          ) -> None:
    """NIRISS SOSS positions come from POS_FILE orders, never the cube."""
    manager, original = make_trace_case(tmp_path, monkeypatch)
    instrument = make_trace_instrument(JWST_NIRISS_SOSS, tmp_path,
                                       monkeypatch)
    order1 = np.zeros((YSIZE, XSIZE), dtype=bool)
    order1[2:5] = True
    order2 = np.zeros((YSIZE, XSIZE), dtype=bool)
    order2[7:9, :10] = True
    orders = {1: order1, 2: order2}
    monkeypatch.setattr(instrument, 'get_trace_pos',
                        lambda map2d, order_num: (orders[order_num], None))
    calls = record_trace_plots(monkeypatch)
    reads = spy_reads(monkeypatch, manager)
    forbid_math(monkeypatch, manager)
    mask = instrument.get_trace_mask(cube=manager, no_plot=False,
                                     plot_frames=[0, -1])
    np.testing.assert_equal(mask, order1 | order2)
    assert reads == [(0, 1), (len(manager) - 1, len(manager))]
    np.testing.assert_equal(calls[0][1][1], original[0][-1])


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
