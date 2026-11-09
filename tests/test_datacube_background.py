#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bounded background-stage scientific and transactional regressions."""

import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.hdu.image import Section
from numpy.typing import NDArray
from scipy.ndimage import shift

from aperocore import math as mp
from aperocore.constants.param_functions import ParamDict

from sossisse.core import base
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.instruments.default import Instrument
from test_datacube_fits import collect
from test_datacube_loader import make_loader


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'tests.test_datacube_background'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
FloatArray = NDArray[np.float64]
BackgroundCase = tuple[Instrument, DataCube, tuple[FloatArray, ...],
                       FloatArray, list[tuple[FloatArray, FloatArray]]]


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def make_background_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                         limit: int = 2, jump: tuple[int, int] | None = None,
                         variant: str | None = None) -> BackgroundCase:
    """Create unequal segments with spatial structure and uneven NaN counts."""
    instrument, _ = make_loader(tmp_path, monkeypatch, limit=limit)
    detector_y, detector_x = np.indices((12, 20), dtype=float)
    calibration = (10 + np.sin(detector_x / 2) * 3
                   + np.cos(detector_y / 3) * 2
                   + detector_x * detector_y / 40)
    generator = np.random.default_rng(42)
    science = np.stack([
        1.7 * shift(calibration, (0, 0.35)) + 4
        + index * np.sin(detector_x / 4) / 5
        + generator.normal(0, 0.03, calibration.shape)
        for index in range(7)])
    science[:3, 3, 4] = np.nan
    science[3:6, 6, 10] = np.nan
    science[:, 0, 0] = np.nan
    if variant == 'zero':
        science[:] = 0
    elif variant == 'allnan':
        science[:] = np.nan
    elif variant == 'infinite':
        science[0, 4, 5] = np.inf
        science[1, 5, 6] = -np.inf
        science[0, 6, 7] = np.inf
        science[-1, 6, 7] = -np.inf
    error = generator.uniform(1, 3, science.shape)
    error[2, 1, 1] = np.inf
    quality = np.arange(science.size, dtype=float).reshape(science.shape)
    quality[0, 0, 0] = np.nan
    quality[-1, 0, 1] = 2 ** 31 + 1
    files = []
    for start, stop in [(0, 3), (3, 7)]:
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=header)]
        for name, array in [('SCI', science), ('ERR', error), ('DQ', quality)]:
            hdus.append(fits.ImageHDU(array[start:stop], name=name))
        filename = tmp_path / f'nrs1_jump001_segment{start}.fits'
        fits.HDUList(hdus).writeto(filename)
        files.append(str(filename))
    calibration_file = tmp_path / 'background.fits'
    fits.writeto(calibration_file, calibration)
    instrument.params.set('GENERAL.FILES', files[::-1], source='test')
    instrument.params.set('GENERAL.DO_BACKGROUND', True, source='test')
    instrument.params.set('GENERAL.BKGFILE', str(calibration_file),
                          source='test')
    instrument.params.set('WLC.INPUTS.BACKGROUND_GLITCH_BOX', [2, 18, 2, 10],
                          source='test')
    instrument.params.set('WLC.INPUTS.BACKGROUND_SHIFTS', [-1., 1.1, 0.5],
                          source='test')
    monkeypatch.setattr(instrument, 'load_data', fits.getdata)
    instrument._jump_slice = jump
    manager = instrument.load_data_with_dq()
    images = []

    def record_plot(inst, before, after):
        images.append((before.copy(), after.copy()))

    monkeypatch.setattr(plots, 'plot_background', record_plot)
    original = (science, error, quality)
    if jump is not None:
        original = tuple(array[slice(*jump)] for array in original)
    return instrument, manager, original, calibration, images


def reference_background(science: NDArray[np.float64],
                         calibration: NDArray[np.float64],
                         box: list[int],
                         shifts: list[float]) -> NDArray[np.float64]:
    """Reproduce the original full-cube background fitting calculation."""
    with warnings.catch_warnings(record=True):
        mean = np.nanmean(science, axis=0)
    region = (slice(box[2], box[3]), slice(box[0], box[1]))
    offsets = np.arange(*shifts)
    residual_rms = np.zeros_like(offsets)
    for index, offset in enumerate(offsets):
        shifted = shift(calibration, (0, offset))
        coefficients, _ = mp.robust_polyfit(
            shifted[region].ravel(), mean[region].ravel(), 1, 5)
        residuals = mean - np.polyval(coefficients, shifted)
        residual_rms[index] = np.std(np.nanmedian(residuals[region], axis=0))
    minimum = np.argmin(residual_rms)
    optimal = 0.
    if minimum not in (0, len(offsets) - 1):
        local = slice(minimum - 1, minimum + 2)
        coefficients = np.polyfit(offsets[local], residual_rms[local], 2)
        optimal = -0.5 * coefficients[1] / coefficients[0]
    shifted = shift(calibration, (0, optimal))
    coefficients, _ = mp.robust_polyfit(
        shifted[region].ravel(), mean[region].ravel(), 1, 5)
    return np.polyval(coefficients, shifted)


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('jump', [None, (2, 6)])
def test_background_parity_and_bounds(tmp_path, monkeypatch, limit, jump):
    outputs = make_background_case(tmp_path, monkeypatch, limit, jump)
    instrument, manager, original, calibration, images = outputs
    before = manager.current_infiles.copy()
    fit_args = [original[0], calibration, [2, 18, 2, 10], [-1., 1.1, 0.5]]
    expected = original[0] - reference_background(*fit_args)
    reads = []
    original_read = manager.read_integrations

    def bounded_read(start=0, stop=None):
        arrays = original_read(start, stop)
        assert arrays['FLUX'].shape[0] <= limit
        reads.append(arrays['FLUX'].shape[0])
        return arrays

    monkeypatch.setattr(manager, 'read_integrations', bounded_read)
    assert instrument.remove_background(manager) is manager
    assert manager.current_infiles != before
    assert all('_background_' in name for name in manager.current_infiles)
    assert all('TEMP_INI_CUBE' in name for name in manager.current_infiles)
    for actual, wanted in zip(collect(manager),
                              (expected, original[1], original[2])):
        np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-10)
        assert actual.dtype == np.float64
    np.testing.assert_equal(images[0][0], original[0][0])
    np.testing.assert_allclose(images[0][1], expected[0], atol=1e-10)
    assert reads and max(reads) <= limit


def test_disabled_background_reads_only_enable():
    """A disabled stage requires no files, calibration or reuse settings."""
    instrument = Instrument.__new__(Instrument)
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.DO_BACKGROUND', False, source='test')
    manager = DataCube(max_integrations=2)
    assert instrument.remove_background(manager) is manager
    assert manager.current_infiles == []


@pytest.mark.parametrize('reuse', [False, True])
def test_background_complete_reuse(tmp_path, monkeypatch, reuse):
    instrument, manager, _, _, images = make_background_case(tmp_path,
                                                             monkeypatch)
    instrument.remove_background(manager)
    paths = manager.current_infiles.copy()
    instrument.params.set('GENERAL.USE_TEMPORARY', reuse, source='test')
    manager = instrument.load_data_with_dq()
    writes = []
    original_write = manager.write_product

    def tracked_write(product, arrays):
        writes.append(product.filename)
        return original_write(product, arrays)

    monkeypatch.setattr(manager, 'write_product', tracked_write)
    if reuse:
        def forbidden(*unused_args, **unused_kwargs):
            pytest.fail('A complete cache must not accumulate or refit')

        monkeypatch.setattr(manager, 'iter_integrations', forbidden)
        monkeypatch.setattr(mp, 'robust_polyfit', forbidden)
    instrument.remove_background(manager)
    assert writes == ([] if reuse else paths)
    assert manager.current_infiles == paths
    assert len(images) == 2
    for first, second in zip(images[0], images[1]):
        np.testing.assert_equal(first, second)


@pytest.mark.parametrize('change', ['missing', 'header', 'shape',
                                    'calibration', 'box', 'shifts', 'parents'])
def test_background_invalidates_reuse(tmp_path, monkeypatch, change):
    outputs = make_background_case(tmp_path, monkeypatch)
    instrument, manager, original, calibration, _ = outputs
    instrument.remove_background(manager)
    paths = manager.current_infiles.copy()
    if change == 'missing':
        Path(paths[-1]).unlink()
    elif change in ('header', 'shape'):
        with fits.open(paths[-1], mode='update') as hdul:
            if change == 'header':
                primary_hdu = cast(fits.PrimaryHDU, hdul[0])
                primary_hdu.header['PRODUCT'] = 'wrong-identity'
            else:
                hdul['DQ'] = fits.ImageHDU(np.zeros((1, 1, 1)), name='DQ')
    elif change == 'calibration':
        calibration = shift(calibration, (0, 0.2))
        fits.writeto(instrument.params['GENERAL.BKGFILE'], calibration,
                     overwrite=True)
    elif change == 'box':
        instrument.params.set('WLC.INPUTS.BACKGROUND_GLITCH_BOX',
                              [3, 17, 3, 9], source='test')
    elif change == 'shifts':
        instrument.params.set('WLC.INPUTS.BACKGROUND_SHIFTS',
                              [-0.5, 1.1, 0.5], source='test')
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    manager = instrument.load_data_with_dq()
    if change == 'parents':
        with fits.open(manager.current_infiles[-1], mode='update') as hdul:
            primary_hdu = cast(fits.PrimaryHDU, hdul[0])
            primary_hdu.header['PRODUCT'] = 'changed-last-parent'
        manager = DataCube(manager.current_infiles,
                           max_integrations=manager.max_integrations)
    writes = []
    original_write = manager.write_product

    def tracked_write(product, arrays):
        writes.append(product.filename)
        return original_write(product, arrays)

    monkeypatch.setattr(manager, 'write_product', tracked_write)
    instrument.remove_background(manager)
    assert writes
    if change in ('missing', 'header', 'shape'):
        assert writes == paths
    else:
        assert not set(paths).intersection(manager.current_infiles)
    box = instrument.params['WLC.INPUTS.BACKGROUND_GLITCH_BOX']
    shifts = instrument.params['WLC.INPUTS.BACKGROUND_SHIFTS']
    science = original[0][:len(manager)]
    expected = science - reference_background(science, calibration, box,
                                               shifts)
    np.testing.assert_allclose(collect(manager)[0], expected, atol=1e-10)


@pytest.mark.parametrize('failure', ['write', 'plot', 'read', 'fit',
                                     'validation', 'calibration'])
def test_background_failure_does_not_commit(tmp_path, monkeypatch, failure):
    instrument, manager, original, _, _ = make_background_case(tmp_path,
                                                               monkeypatch)
    before = manager.current_infiles.copy()
    original_write = manager.write_product
    calls = []

    def broken(*unused_args, **unused_kwargs):
        raise OSError('simulated background failure')

    def broken_write(product, arrays):
        calls.append(product.filename)
        if len(calls) == 2:
            broken()
        return original_write(product, arrays)

    with monkeypatch.context() as patch:
        if failure == 'write':
            patch.setattr(manager, 'write_product', broken_write)
        elif failure == 'plot':
            patch.setattr(plots, 'plot_background', broken)
        elif failure == 'read':
            patch.setattr(manager, 'iter_integrations', broken)
        elif failure == 'fit':
            patch.setattr(mp, 'robust_polyfit', broken)
        elif failure == 'calibration':
            patch.setattr(instrument, 'load_data', broken)
        else:
            patch.setattr(manager, 'products_valid', lambda products: False)
        with pytest.raises((OSError, ValueError)):
            instrument.remove_background(manager)
    assert manager.current_infiles == before
    for actual, expected in zip(collect(manager), original):
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize('variant', ['zero', 'allnan'])
def test_background_degenerate_mean_matches_legacy(tmp_path, monkeypatch,
                                                  variant):
    outputs = make_background_case(tmp_path, monkeypatch, variant=variant)
    instrument, manager, original, calibration, _ = outputs
    before = manager.current_infiles.copy()
    fit_args = [original[0], calibration, [2, 18, 2, 10], [-1., 1.1, 0.5]]
    with warnings.catch_warnings(record=True):
        if variant == 'allnan':
            with pytest.raises(Exception) as legacy:
                reference_background(*fit_args)
            with pytest.raises(legacy.type, match='Not enough xvector'):
                instrument.remove_background(manager)
            assert manager.current_infiles == before
        else:
            expected = original[0] - reference_background(*fit_args)
            instrument.remove_background(manager)
            np.testing.assert_equal(collect(manager)[0], expected)


@pytest.mark.parametrize('limit', [1, 2, 100])
def test_background_mean_counts_nan_not_infinity(tmp_path, monkeypatch,
                                                 limit):
    outputs = make_background_case(tmp_path, monkeypatch, limit=limit,
                                    variant='infinite')
    instrument, manager, original, _, _ = outputs
    before = manager.current_infiles.copy()
    with warnings.catch_warnings(record=True):
        expected = np.nanmean(original[0], axis=0)[2:10, 2:18].ravel()

    def inspect_fit(background, mean, degree, sigma):
        np.testing.assert_equal(mean, expected)
        raise RuntimeError('mean verified before fitting')

    monkeypatch.setattr(mp, 'robust_polyfit', inspect_fit)
    with pytest.raises(RuntimeError, match='mean verified'):
        instrument.remove_background(manager)
    assert manager.current_infiles == before


@pytest.mark.parametrize('reuse', [False, True])
def test_background_plot_failure_keeps_parent_stage(tmp_path, monkeypatch,
                                                    reuse):
    instrument, manager, original, _, _ = make_background_case(tmp_path,
                                                               monkeypatch)
    if reuse:
        instrument.remove_background(manager)
        instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
        manager = instrument.load_data_with_dq()
    before = manager.current_infiles.copy()

    def broken_plot(*unused_args):
        raise RuntimeError('plot failed')

    monkeypatch.setattr(plots, 'plot_background', broken_plot)
    with pytest.raises(RuntimeError, match='plot failed'):
        instrument.remove_background(manager)
    assert manager.current_infiles == before
    for actual, expected in zip(collect(manager), original):
        np.testing.assert_equal(actual, expected)


def test_background_cache_reads_only_first_science_frame(tmp_path,
                                                        monkeypatch):
    instrument, manager, _, _, _ = make_background_case(tmp_path, monkeypatch)
    instrument.remove_background(manager)
    instrument.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    manager = instrument.load_data_with_dq()
    section_read = Section.__getitem__
    reads = []

    def checked_section(section, selection):
        result = section_read(section, selection)
        reads.append(result.shape)
        return result

    monkeypatch.setattr(Section, '__getitem__', checked_section)
    instrument.remove_background(manager)
    assert reads == [(1, 12, 20)] * 3 + [(12, 20)]


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================