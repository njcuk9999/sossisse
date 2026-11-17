#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bounded coefficient fits, reconstruction publication and reopening."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from aperocore.constants.param_functions import ParamDict
from sossisse.general import general
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.general.reconstruction import ReconstructionProducts
from sossisse.instruments.default import Instrument


def make_case(tmp_path, monkeypatch, limit=2, selection=None, zpoint=False):
    instrument = Instrument.__new__(Instrument)
    instrument.name = 'test'
    instrument._trace_suffix = '_nrs1'
    instrument.params = ParamDict()
    instrument.params.set('WLC.LMODEL.FIT_ZERO_POINT_OFFSET', zpoint,
                          source='test')
    names = ['amplitude', 'dx'] + (['zeropoint'] if zpoint else [])
    med = np.linspace(20, 40, 48).reshape(6, 8)
    derivative = np.random.default_rng(42).normal(size=med.shape)
    basis = np.array([med.ravel(), derivative.ravel()]
                     + ([np.ones(med.size)] if zpoint else []))
    flux = np.array([med * (1 + index / 20) + derivative * index / 30
                     + (0.25 if zpoint else 0) for index in range(7)])
    error = np.full(flux.shape, 2.)
    quality = np.zeros_like(flux)
    flux[2, 1, 1] = np.nan
    flux[3, 2, 2] += 100
    paths = []
    for start, stop in [(0, 3), (3, 7)]:
        header = fits.Header(dict(INTSTART=start + 1, INTEND=stop))
        hdus = fits.HDUList([fits.PrimaryHDU(header=header)])
        for name, array in [('SCI', flux), ('ERR', error), ('DQ', quality)]:
            hdus.append(fits.ImageHDU(array[start:stop], name=name))
        path = tmp_path / f'source{start}.fits'
        hdus.writeto(path)
        paths.append(str(path))
    manager = DataCube(max_integrations=limit)
    manager.normalize(paths, tmp_path / 'ini.fits',
                      integration_slice=selection)
    chosen = slice(*selection) if selection else slice(None)
    variables = dict()
    variables['OUTPUT_NAMES'] = names
    variables['DATA_N_FRAMES'] = len(manager)
    variables['META'] = dict()
    variables['WLC_RECON_FILE'] = str(tmp_path / 'recon_nrs1.fits')
    variables['WLC_LTBL_FILE'] = str(tmp_path / 'coefficients.csv')
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, *unused: variables[key])
    monkeypatch.setattr(instrument, 'set_variable',
                        lambda key, value, *unused: variables.update(
                            {key: value}))
    monkeypatch.setattr(instrument, 'update_meta_data', lambda: None)
    monkeypatch.setattr(plots, 'trace_correction_sample',
                        lambda *unused: None)
    monkeypatch.setattr(plots, 'aperture_correction_plot',
                        lambda *unused: None)
    return instrument, manager, med, basis, flux[chosen], error[chosen]


def run_fit(instrument, manager, med, basis):
    empty = np.array([])
    fit_args = [manager, med, np.ones(med.shape), basis,
                empty, empty, empty, empty]
    return instrument.get_linear_coeffs(*fit_args)


def collect_products(products):
    blocks = list(products.iter_integrations())
    return {name: np.concatenate([arrays[name] for _, arrays in blocks])
            for name in ('RECON', 'RESIDUAL', 'VALID', 'ERR')}


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('selection', [None, (1, 6)])
@pytest.mark.parametrize('zpoint', [False, True])
def test_coefficients_and_persisted_planes(tmp_path, monkeypatch, limit,
                                           selection, zpoint):
    inst, manager, med, basis, flux, error = make_case(
        tmp_path, monkeypatch, limit, selection, zpoint)
    original_paths = manager.current_infiles.copy()
    empty = np.array([])
    ref_args = [flux, error, med, np.ones(med.shape), basis,
                empty, empty, empty, empty]
    reference, recon, valid, residual = inst._get_linear_coeffs_block(
        *ref_args)
    correction = np.array(reference['aperture_correction'])
    correction /= np.nanmedian(correction)
    reference['aperture_correction'] = correction
    reference['amplitude'] *= correction
    table, products = run_fit(inst, manager, med, basis)
    chosen = slice(*selection) if selection else slice(None)
    expected_amplitude = (1 + np.arange(7) / 20)[chosen]
    np.testing.assert_allclose(np.asarray(table['amplitude_uncorrected']),
                               expected_amplitude, rtol=1e-10)
    assert manager.current_infiles == original_paths
    for name in reference.colnames:
        np.testing.assert_allclose(np.asarray(table[name]),
                       np.asarray(reference[name]),
                                   equal_nan=True)
    actual = collect_products(products)
    for name, wanted in [('RECON', recon), ('VALID', valid),
                         ('RESIDUAL', residual), ('ERR', error)]:
        np.testing.assert_allclose(actual[name], wanted, equal_nan=True)
    assert max(shape[0] for shape in products.shapes) <= limit
    inst.save_wlc_results(products, table)
    reopened, restored = inst.load_wlc_results()
    assert restored.colnames == table.colnames
    for name in actual:
        np.testing.assert_equal(collect_products(reopened)[name], actual[name])
    assert not Path(inst.get_variable('WLC_RECON_FILE', 'test')).exists()


@pytest.mark.parametrize('damage', ['missing', 'table', 'plane'])
def test_manifest_rejects_incomplete_outputs(tmp_path, monkeypatch, damage):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    table, products = run_fit(inst, manager, med, basis)
    inst.save_wlc_results(products, table)
    if damage == 'missing':
        Path(products.current_infiles[-1]).unlink()
    elif damage == 'table':
        path = Path(inst.get_variable('WLC_LTBL_FILE', 'test'))
        path.write_text(path.read_text() + '\n')
    else:
        with fits.open(products.current_infiles[-1], mode='update') as hdul:
            hdul['ERR'] = fits.ImageHDU(np.zeros((1, 1, 1)), name='ERR')
    with pytest.raises((OSError, ValueError, KeyError)):
        inst.load_wlc_results()


def test_manifest_merges_jump_references_without_pixel_reads(tmp_path,
                                                            monkeypatch):
    cases = []
    for index, selection in enumerate([(0, 3), (3, 7)]):
        directory = tmp_path / str(index)
        directory.mkdir()
        inst, manager, med, basis, _, _ = make_case(
            directory, monkeypatch, selection=selection)
        table, products = run_fit(inst, manager, med, basis)
        inst.save_wlc_results(products, table)
        cases.append(dict(manifest=inst.get_wlc_manifest()))
    monkeypatch.setattr(inst, 'set_jump_context', lambda **unused: None)
    monkeypatch.setattr(inst, 'define_filenames', lambda: None)
    monkeypatch.setattr(plots, 'plot_stability', lambda *unused: None)
    original = fits.ImageHDU.data

    def forbidden(*unused):
        raise AssertionError('No full image read during jump merge')

    monkeypatch.setattr(fits.ImageHDU, 'data', property(forbidden))
    general.merge_linear_recon_chunks(inst, cases, [2])
    merged, table = inst.load_wlc_results()
    assert len(merged) == len(table) == 7
    assert len(merged.current_infiles) == 4
    monkeypatch.setattr(fits.ImageHDU, 'data', original)


def test_spectral_extraction_requires_complete_manifest(tmp_path, monkeypatch):
    inst, _, _, _, _, _ = make_case(tmp_path, monkeypatch)
    inst.params.set('INPUTS.OBJECTNAME', 'test', source='test')
    monkeypatch.setattr(inst, 'define_filenames', lambda: None)
    monkeypatch.setattr(general.misc, 'sossart', lambda: None)
    with pytest.raises(FileNotFoundError):
        general.spectral_extraction_chunk(inst)


@pytest.mark.parametrize('enabled', [False, True])
def test_pca_bridge_and_context_cache(tmp_path, monkeypatch, enabled):
    from sossisse.instruments import default

    inst, manager, med, _, _, _ = make_case(tmp_path, monkeypatch)
    inst.params.set('WLC.LMODEL.FIT_PCA', enabled, source='test')
    inst.params.set('WLC.LMODEL.FIT_N_PCA', 1, source='test')
    inst.params.set('GENERAL.USE_TEMPORARY', False, source='test')
    extras = dict()
    extras['DATA_X_SIZE'] = 8
    extras['DATA_Y_SIZE'] = 6
    extras['HAS_BASELINE'] = True
    extras['BASELINE_DOMAIN'] = np.arange(len(manager))
    cache = str(tmp_path / 'pca_nrs1_jump000.fits')
    extras['TEMP_PCA_FILE'] = cache
    original_get = inst.get_variable

    def get_variable(key, *unused):
        return extras[key] if key in extras else original_get(key, 'test')

    class FakePCA:
        def __init__(self, n_components):
            self.explained_variance_ratio_ = np.ones(n_components)

        def fit(self, flux, weights):
            assert flux.shape == weights.shape
            self.components_ = np.ones((1, flux.shape[1]))
            return self

    monkeypatch.setattr(inst, 'get_variable', get_variable)
    monkeypatch.setattr(inst, 'get_baseline_params', lambda: None)
    monkeypatch.setattr(default, 'EMPCA', FakePCA)
    monkeypatch.setattr(plots, 'pca_plot', lambda *unused: None)
    before = manager.current_infiles.copy()
    if not enabled:
        monkeypatch.setattr(manager, 'iter_integrations',
                            lambda: pytest.fail('Disabled PCA read pixels'))
        assert inst.fit_pca(manager, med, np.ones(med.shape, bool)) is None
        return
    result = inst.fit_pca(manager, med, np.ones(med.shape, bool))
    assert result is not None
    assert result.shape == (1, 6, 8)
    assert Path(cache).exists()
    assert manager.current_infiles == before
    inst.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    monkeypatch.setattr(manager, 'iter_integrations',
                        lambda: pytest.fail('Cached PCA read pixels'))
    np.testing.assert_equal(inst.fit_pca(manager, med,
                                         np.ones(med.shape, bool)), result)


@pytest.mark.parametrize('failure', ['write', 'plot', 'table'])
def test_failed_fit_does_not_publish(tmp_path, monkeypatch, failure):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    inputs = manager.current_infiles.copy()

    def broken(*unused, **unused_kwargs):
        raise OSError('injected reconstruction failure')

    if failure == 'write':
        monkeypatch.setattr(ReconstructionProducts, 'write_block', broken)
    elif failure == 'plot':
        monkeypatch.setattr(plots, 'aperture_correction_plot', broken)
    else:
        from sossisse.core import io
        monkeypatch.setattr(io, 'save_table', broken)
    with pytest.raises(OSError, match='injected'):
        table, products = run_fit(inst, manager, med, basis)
        inst.save_wlc_results(products, table)
    assert manager.current_infiles == inputs
    assert not Path(inst.get_wlc_manifest()).exists()


def test_pixel_baseline_fits_whole_history(tmp_path, monkeypatch):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    _, products = run_fit(inst, manager, med, basis)
    extras = dict(HAS_BASELINE=True, BASELINE_DOMAIN=np.arange(len(manager)))
    original_get = inst.get_variable
    monkeypatch.setattr(inst, 'get_variable', lambda key, *unused:
                        extras[key] if key in extras
                        else original_get(key, 'test'))
    monkeypatch.setattr(inst, 'get_baseline_params', lambda: None)
    inst.params.set('WLC.GENERAL.TRANSIT_BASELINE_POLYORD', 1, source='test')
    original = collect_products(products)
    expected = original['RESIDUAL'].copy()
    from aperocore import math as mp
    frames = np.arange(len(products), dtype=float)
    for row in range(expected.shape[1]):
        for column in range(expected.shape[2]):
            if not np.any(np.isfinite(original['VALID'][:, row, column])):
                continue
            sample = expected[:, row, column]
            valid = np.isfinite(sample)
            if np.count_nonzero(valid) > 1:
                fit, _ = mp.robust_polyfit(frames[valid], sample[valid], 1, 5)
                expected[:, row, column] -= np.polyval(fit, frames)
    corrected = inst.per_pixel_baseline(products)
    actual = collect_products(corrected)
    np.testing.assert_allclose(actual['RESIDUAL'], expected, equal_nan=True)
    for name in ('RECON', 'ERR', 'VALID'):
        np.testing.assert_equal(actual[name], original[name])


def test_linear_recon_chunk_completes_real_fit_and_save(tmp_path, monkeypatch):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    monkeypatch.setattr(general, 'linear_recon_init', lambda inst: False)
    monkeypatch.setattr(inst, 'load_data_with_dq', lambda: manager)
    for name in ('apply_flat_field', 'apply_dq', 'remove_background',
                 'low_pass_filter', 'patch_isolated_bads',
                 'remove_cosmic_rays'):
        monkeypatch.setattr(inst, name, lambda data: data)
    mask = np.ones(med.shape, dtype=bool)
    empty = np.array([])
    monkeypatch.setattr(inst, 'get_trace_mask', lambda **unused: mask)
    monkeypatch.setattr(inst, 'create_median_stack',
                        lambda data: [data, med, np.ones(len(data))])
    monkeypatch.setattr(inst, 'clean_residual_1f',
                        lambda data, *unused: data)
    monkeypatch.setattr(inst, 'recenter_trace_position',
                        lambda trace, med: trace)
    monkeypatch.setattr(inst, 'fit_pca', lambda *unused: None)
    monkeypatch.setattr(inst, 'get_fit_params', lambda med: [med] * 5)
    monkeypatch.setattr(inst, 'get_linear_recon_mask',
                        lambda *unused: (mask, empty, empty, empty, empty))
    monkeypatch.setattr(inst, 'setup_linear_reconstruction',
                        lambda *unused: basis)
    monkeypatch.setattr(inst, 'get_int_times',
                        lambda: np.arange(len(manager), dtype=float))
    original_get = inst.get_variable

    def get_variable(key, *unused):
        if key == 'BASELINE_DOMAIN':
            return np.arange(len(manager))
        return original_get(key, 'test')

    monkeypatch.setattr(inst, 'get_variable', get_variable)
    monkeypatch.setattr(inst, 'get_baseline_params', lambda: None)
    monkeypatch.setattr(inst, 'rms_baselines', lambda: [])
    monkeypatch.setattr(inst, 'get_effective_wavelength', lambda med: None)
    monkeypatch.setattr(plots, 'plot_stability', lambda *unused: None)
    monkeypatch.setattr(general, 'write_sossisse_summary', lambda inst: None)
    assert general.linear_recon_chunk(inst) is inst
    products, table = inst.load_wlc_results()
    assert len(products) == len(table) == len(manager)
    assert 'bjd' in table.colnames
    assert set(collect_products(products)) == {'ERR', 'RECON', 'RESIDUAL',
                                              'VALID'}


def test_reconstruction_reuse_requires_complete_outputs(tmp_path, monkeypatch):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    table, products = run_fit(inst, manager, med, basis)
    inst.save_wlc_results(products, table)
    inst.params.set('INPUTS.OBJECTNAME', 'test', source='test')
    inst.params.set('GENERAL.USE_TEMPORARY', True, source='test')
    monkeypatch.setattr(inst, 'define_filenames', lambda: None)
    restored = []
    monkeypatch.setattr(inst, 'set_linear_output_metadata',
                        lambda: restored.append(True))
    monkeypatch.setattr(general.misc, 'sossart', lambda: None)
    assert general.linear_recon_init(inst) is True
    assert restored == [True]
    Path(products.current_infiles[-1]).unlink()
    assert general.linear_recon_init(inst) is False
    assert restored == [True]


def test_mixed_detector_products_are_rejected(tmp_path, monkeypatch):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    _, products = run_fit(inst, manager, med, basis)
    with fits.open(products.current_infiles[-1], mode='update') as hdul:
        cast(fits.PrimaryHDU, hdul[0]).header['RCDET'] = '_nrs2'
    with pytest.raises(ValueError, match='Mixed.*detector'):
        ReconstructionProducts(products.current_infiles, 2)


def test_incomplete_jump_selection_rejected(tmp_path, monkeypatch):
    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch)
    _, products = run_fit(inst, manager, med, basis)
    for paths in (products.current_infiles[1:],
                  products.current_infiles[:-1]):
        with pytest.raises(ValueError, match='coverage'):
            ReconstructionProducts(paths, 2)