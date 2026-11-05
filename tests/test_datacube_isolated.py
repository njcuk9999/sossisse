#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isolated bad-pixel patch parity and transactional stage persistence."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits
from scipy.signal import convolve2d

from aperocore.constants.param_functions import ParamDict

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_loader import make_loader


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_isolated'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
PatchCase = tuple[Instrument, DataCube, tuple[FloatArray, ...]]


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def make_patch_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                    limit: int = 2, jump: tuple[int, int] | None = None
                    ) -> PatchCase:
    """Create unequal segments with isolated NaNs, clusters and spikes.

    :param tmp_path: pytest directory for raw and working products
    :param monkeypatch: pytest patcher used by the loader fixture
    :param limit: integrations per normalized product block
    :param jump: optional active half-open integration selection
    :return: instrument, normalized manager and selected SCI/ERR/DQ arrays
    """
    instrument, _ = make_loader(tmp_path, monkeypatch, limit=limit)
    generator = np.random.default_rng(11)
    science = generator.normal(100, 5, (7, 12, 20))
    error = generator.uniform(1, 3, science.shape)
    for index in range(7):
        # Every frame needs two stamps: the legacy grid divides by columns.
        science[index, 3 + index % 4, 4 + index] = np.nan
        science[index, 8, 15 - index] = np.nan
        # Positive and negative spikes exercise only the upper sigma cut.
        science[index, 5 + index % 2, 12 + index % 3] += 80 * (index + 1)
        science[index, 10, 5 + index] -= 90
    # An infinite neighbouring error propagates through the plain mean.
    error[2, 3 + 2, 4 + 3] = np.inf
    # Adjacent and edge NaNs are not isolated and stay NaN.
    science[:, 10, 12:14] = np.nan
    science[:, 0, 9] = np.nan
    # Constant pixels make the percentile width zero.
    science[:, 11, 19] = 100.0
    quality = np.arange(science.size, dtype=float).reshape(science.shape)
    quality[0, 0, 0] = np.nan
    files = []
    for start, stop in [(0, 3), (3, 7)]:
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=header)]
        for name, array in [('SCI', science), ('ERR', error),
                            ('DQ', quality)]:
            hdus.append(fits.ImageHDU(array[start:stop], name=name))
        filename = tmp_path / f'nrs1_jump001_segment{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        files.append(str(filename))
    instrument.params.set('GENERAL.FILES', files[::-1], source='test')
    instrument.params.set('WLC.GENERAL.PATCH_ISOLATED_BADS', True,
                          source='test')
    instrument.params.set('WLC.GENERAL.PATCH_IBADS_SSIZE', 5, source='test')
    instrument.params.set('WLC.GENERAL.REMOVE_COSMIC_RAYS', True,
                          source='test')
    instrument.params.set('WLC.GENERAL.COSMIC_RAY_SIGMA', 3.0, source='test')
    instrument._jump_slice = jump
    manager = instrument.load_data_with_dq()
    original = (science, error, quality)
    if jump is not None:
        original = tuple(array[slice(*jump)] for array in original)
    return instrument, manager, original


def reference_patch(instrument: Instrument, science: FloatArray,
                    error: FloatArray, size: int = 5) -> dict[str, Any]:
    """Reproduce the original full-cube isolated-NaN patch and diagnostics.

    :param instrument: owner of the unchanged stamp helper
    :param science: full active science cube, copied before patching
    :param error: full active error cube, copied before patching
    :param size: odd stamp width
    :return: patched arrays, heat map, first frame and stamp dictionaries
    """
    cube = science.copy()
    err = error.copy()
    heat_map = np.zeros_like(cube[0], dtype=float)
    bad_pixels_before = dict()
    bad_pixels_after = dict()
    bpixel_ssize = (size - 1) // 2
    iframe_before = np.array(cube[0])
    badpix_pos = dict()
    for it, iframe in enumerate(range(cube.shape[0])):
        cframe = np.array(cube[iframe, :, :])
        ecframe = np.array(err[iframe, :, :])
        mframe = np.isfinite(cframe)
        kernel = np.zeros([3, 3], dtype=float)
        kernel[1, :] = 1
        kernel[:, 1] = 1
        n_bad = convolve2d(np.array(mframe, dtype=float), kernel,
                           mode='same')
        isolated_bad = (n_bad == 4) & ~mframe
        heat_map += isolated_bad.astype(int)
        ypix, xpix = np.where(isolated_bad)
        mean_vals = np.mean([cframe[ypix - 1, xpix],
                             cframe[ypix + 1, xpix],
                             cframe[ypix, xpix - 1],
                             cframe[ypix, xpix + 1]], axis=0)
        mean_err_vals = np.mean([ecframe[ypix - 1, xpix],
                                 ecframe[ypix + 1, xpix],
                                 ecframe[ypix, xpix - 1],
                                 ecframe[ypix, xpix + 1]], axis=0)
        bad_pixels_before[it] = instrument.create_bad_pixel_stamps(
            ypix, xpix, cframe, bpixel_ssize)
        cframe[ypix, xpix] = mean_vals
        ecframe[ypix, xpix] = mean_err_vals
        bad_pixels_after[it] = instrument.create_bad_pixel_stamps(
            ypix, xpix, cframe, bpixel_ssize)
        cube[iframe, :, :] = cframe
        err[iframe, :, :] = ecframe
        badpix_pos[iframe] = [ypix, xpix]
    return dict(cube=cube, err=err, heat_map=heat_map,
                before=iframe_before, before_stamps=bad_pixels_before,
                after_stamps=bad_pixels_after, positions=badpix_pos)


def record_diagnostics(instrument: Instrument,
                       monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Record plot, stamp and cutout inputs without writing figures.

    :param instrument: instrument whose stamp/cutout methods are wrapped
    :param monkeypatch: pytest patcher restoring the originals
    :return: mutable mapping of diagnostic name to recorded calls
    """
    calls: dict[str, list] = dict(heatmap=[], pixels=[], stamps=[],
                                  cutouts=[])
    original_stamps = instrument.create_stamp_images

    def record_heatmap(inst: Any, heat: Any, before: Any, after: Any,
                       *labels: Any) -> None:
        calls['heatmap'].append((heat.copy(), before.copy(), after.copy(),
                                 labels))

    def record_pixels(inst: Any, pixel_dict: Any) -> None:
        calls['pixels'].append(pixel_dict)

    def record_stamps(*stamp_args: Any, **stamp_kwargs: Any) -> Any:
        calls['stamps'].append((stamp_args, stamp_kwargs))
        return original_stamps(*stamp_args, **stamp_kwargs)

    def record_cutouts(cutouts: Any, table: Any) -> None:
        calls['cutouts'].append((cutouts, table))

    monkeypatch.setattr(plots, 'plot_heatmap', record_heatmap)
    monkeypatch.setattr(plots, 'plot_pixels', record_pixels)
    monkeypatch.setattr(instrument, 'create_stamp_images', record_stamps)
    monkeypatch.setattr(instrument, 'save_cutouts', record_cutouts)
    return calls


def assert_diagnostics(instrument: Instrument, calls: dict[str, list],
                       expected: dict[str, Any]) -> None:
    """Compare globally indexed diagnostics against the full reference.

    :param instrument: owner of the unchanged stamp image helper
    :param calls: latest recorded diagnostic calls
    :param expected: reference_patch output for the active selection
    """
    heat, before, after, labels = calls['heatmap'][-1]
    np.testing.assert_equal(heat, expected['heat_map'])
    np.testing.assert_equal(before, expected['before'])
    np.testing.assert_allclose(after, expected['cube'][0])
    assert labels == ('isolated bad pixels', 'isolated_badpixels',
                      'Number across all integrations')
    stamp_args, stamp_kwargs = calls['stamps'][-1]
    before_dict, after_dict, positions = stamp_args
    assert stamp_kwargs == dict(mode='seperate')
    assert calls['stamps'][-2][1] == dict(num=100)
    frames = list(range(expected['cube'].shape[0]))
    assert list(before_dict) == list(after_dict) == frames
    assert list(positions) == frames
    for frame in frames:
        for name, actual_dict in (('before_stamps', before_dict),
                                  ('after_stamps', after_dict)):
            assert len(actual_dict[frame]) == len(expected[name][frame])
            for actual, wanted in zip(actual_dict[frame],
                                      expected[name][frame]):
                np.testing.assert_allclose(actual, wanted)
        np.testing.assert_equal(positions[frame],
                                expected['positions'][frame])
    stamp_images = instrument.create_stamp_images(
        expected['before_stamps'], expected['after_stamps'],
        expected['positions'], mode='seperate')
    cutouts, table = calls['cutouts'][-1]
    assert list(cutouts) == frames
    for frame in frames:
        np.testing.assert_allclose(cutouts[frame][0],
                                   stamp_images[0][frame][0])
        np.testing.assert_allclose(cutouts[frame][1],
                                   stamp_images[0][frame][1])
        assert table[frame].as_array().tolist() == (
            stamp_images[1][frame].as_array().tolist())
    assert list(calls['pixels'][-1]) == frames


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('jump', [None, (1, 6)])
def test_patch_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                      limit: int, jump: tuple[int, int] | None) -> None:
    """Match the original across blocks, unequal segments and jumps."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch,
                                                    limit, jump)
    science, error, quality = original
    expected = reference_patch(instrument, science, error)
    calls = record_diagnostics(instrument, monkeypatch)
    parents = manager.current_infiles.copy()
    reads = []
    original_read = manager.read_integrations

    def bounded_read(start: int = 0, stop: int | None = None) -> Any:
        arrays = original_read(start, stop)
        assert arrays['CUBE'].shape[0] <= limit
        reads.append(arrays['CUBE'].shape[0])
        return arrays

    monkeypatch.setattr(manager, 'read_integrations', bounded_read)
    assert instrument.patch_isolated_bads(manager) is manager
    assert not set(parents).intersection(manager.current_infiles)
    assert all('_isolated_' in name for name in manager.current_infiles)
    actual, actual_error, actual_quality = collect(manager)
    np.testing.assert_allclose(actual, expected['cube'])
    np.testing.assert_allclose(actual_error, expected['err'])
    np.testing.assert_equal(actual_quality, quality)
    # Isolated NaNs were really patched and non-isolated ones remain.
    assert np.isfinite(actual[:, 8, :]).all()
    assert np.isnan(actual[:, 10, 12:14]).all()
    assert len(calls['heatmap']) == len(calls['pixels']) == 1
    assert len(calls['cutouts']) == 1
    assert_diagnostics(instrument, calls, expected)
    assert reads and max(reads) <= limit


def test_infinite_error_neighbour_matches_legacy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ERR is the plain neighbour mean, propagating infinite neighbours."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch)
    record_diagnostics(instrument, monkeypatch)
    instrument.patch_isolated_bads(manager)
    actual_error = collect(manager)[1]
    assert np.isinf(actual_error[2, 3 + 2 % 4, 4 + 2])
    expected = reference_patch(instrument, original[0], original[1])
    np.testing.assert_equal(actual_error, expected['err'])


def test_disabled_patch_reads_only_enable() -> None:
    """A disabled stage requires no files, stamp size or reuse settings."""
    instrument = Instrument.__new__(Instrument)
    instrument.params = ParamDict()
    instrument.params.set('WLC.GENERAL.PATCH_ISOLATED_BADS', False,
                          source='test')
    manager = DataCube(max_integrations=2)
    assert instrument.patch_isolated_bads(manager) is manager
    assert manager.current_infiles == []


@pytest.mark.parametrize('size', [4, 0])
def test_even_stamp_size_rejected(tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch,
                                  size: int) -> None:
    """Reject even stamp sizes before reading or writing any block."""
    instrument, manager, _ = make_patch_case(tmp_path, monkeypatch)
    inputs = manager.current_infiles.copy()
    instrument.params.set('WLC.GENERAL.PATCH_IBADS_SSIZE', size,
                          source='test')

    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        pytest.fail('Validation must precede block access')

    monkeypatch.setattr(manager, 'iter_integrations', forbidden)
    monkeypatch.setattr(manager, 'write_product', forbidden)
    with pytest.raises(exceptions.SossisseConstantException, match='odd'):
        instrument.patch_isolated_bads(manager)
    assert manager.current_infiles == inputs


@pytest.mark.parametrize('reuse,missing', [(False, False), (True, False),
                                          (True, True)])
def test_patch_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     reuse: bool, missing: bool) -> None:
    """Reuse only a complete stage and still regenerate diagnostics."""
    instrument, manager, original = make_patch_case(tmp_path, monkeypatch)
    calls = record_diagnostics(instrument, monkeypatch)
    inputs = manager.current_infiles.copy()
    instrument.patch_isolated_bads(manager)
    outputs = manager.current_infiles.copy()
    if missing:
        Path(outputs[-1]).unlink()
    reopened = DataCube(inputs, max_integrations=2)
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    original_write = reopened.write_product
    writes = []

    def record_write(*write_args: Any) -> None:
        writes.append(write_args[0].filename)
        original_write(*write_args)

    monkeypatch.setattr(reopened, 'write_product', record_write)
    instrument.patch_isolated_bads(reopened)
    assert bool(writes) is (not reuse or missing)
    assert reopened.current_infiles == outputs
    expected = reference_patch(instrument, original[0], original[1])
    np.testing.assert_allclose(collect(reopened)[0], expected['cube'])
    assert len(calls['heatmap']) == len(calls['cutouts']) == 2
    assert_diagnostics(instrument, calls, expected)


def test_parent_change_invalidates_reuse(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch
                                         ) -> None:
    """Unchanged parents reuse products; changed parents force new ones."""
    instrument, manager, _ = make_patch_case(tmp_path, monkeypatch)
    record_diagnostics(instrument, monkeypatch)
    inputs = manager.current_infiles.copy()
    instrument.patch_isolated_bads(manager)
    outputs = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    same = DataCube(inputs, max_integrations=2)
    instrument.patch_isolated_bads(same)
    assert same.current_infiles == outputs
    # An identical-valued but distinct parent stage changes provenance.
    other = DataCube(inputs, max_integrations=2)
    parent_products = other.plan_stage('parent', dict(test='copy'))
    for product, (_, arrays) in zip(parent_products,
                                    other.iter_integrations()):
        other.write_product(product, arrays)
    other.commit_products(parent_products)
    instrument.patch_isolated_bads(other)
    assert not set(other.current_infiles).intersection(outputs)


@pytest.mark.parametrize('failure', ['write', 'heatmap', 'pixels',
                                     'cutouts', 'validate'])
def test_failed_patch_not_committed(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    failure: str) -> None:
    """Do not advance the active stage after any write or diagnostic error."""
    instrument, manager, _ = make_patch_case(tmp_path, monkeypatch)
    record_diagnostics(instrument, monkeypatch)
    inputs = manager.current_infiles.copy()

    def broken(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise RuntimeError('injected patch failure')

    if failure == 'write':
        original_write = manager.write_product
        writes = []

        def partial_write(*write_args: Any) -> None:
            if writes:
                raise RuntimeError('injected patch failure')
            original_write(*write_args)
            writes.append(write_args[0].filename)

        monkeypatch.setattr(manager, 'write_product', partial_write)
    elif failure == 'heatmap':
        monkeypatch.setattr(plots, 'plot_heatmap', broken)
    elif failure == 'pixels':
        monkeypatch.setattr(plots, 'plot_pixels', broken)
    elif failure == 'cutouts':
        monkeypatch.setattr(instrument, 'save_cutouts', broken)
    else:
        monkeypatch.setattr(manager, 'commit_products', broken)
    with pytest.raises(RuntimeError, match='injected patch'):
        instrument.patch_isolated_bads(manager)
    assert manager.current_infiles == inputs


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================
