#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral geometry, convergence and exact block/full-cube parity."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from aperocore.constants.param_functions import ParamDict
from sossisse.core import exceptions
from sossisse.core import io
from sossisse.general import general
from sossisse.general import plots
from sossisse.general.datacube import DataCube
from sossisse.general.reconstruction import ReconstructionProducts
from sossisse.instruments import default
from sossisse.instruments.default import Instrument


def extraction_instrument(monkeypatch, nframes=2, width=7):
    """Set up only ratio settings and legacy dimension metadata."""
    inst = Instrument.__new__(Instrument)
    inst.name = 'test'
    inst.params = ParamDict()
    inst.params.set('WLC.GENERAL.LINRECON_TRACE_WIDTH', 2, source='test')
    variables = dict(DATA_N_FRAMES=nframes, DATA_Y_SIZE=9, DATA_X_SIZE=width)
    monkeypatch.setattr(inst, 'get_variable',
                        lambda key, *unused: variables[key])
    return inst


def test_extract_original_detector_columns(monkeypatch):
    """Leading exclusions and gaps must not shift sampled detector columns."""
    inst = extraction_instrument(monkeypatch)
    model = np.broadcast_to(np.arange(1., 8.)[None, None, :], (2, 9, 7))
    ratio = np.arange(7.) / 10
    residual = model * ratio
    error = np.ones_like(residual)
    positions = np.array([np.nan, 4., 4., np.nan, 4., 0., 4.])
    spec, spec_err = inst.ratio_residual_to_trace(model, error, residual,
                                                  positions)
    columns = np.array([1, 2, 4, 6])
    np.testing.assert_allclose(spec[:, columns], np.tile(ratio[columns],
                                                         (2, 1)))
    assert np.isnan(spec[:, [0, 3, 5]]).all()
    probability = 1 / (1 + 1e-4)
    expected_error = 1 / (np.arange(1., 8.)[columns]
                          * np.sqrt(3 * probability))
    np.testing.assert_allclose(spec_err[:, columns],
                               np.tile(expected_error, (2, 1)))


@pytest.mark.parametrize('converges_on_last', [False, True])
def test_extraction_iteration_limit(monkeypatch, converges_on_last):
    """Enforce ten updates, but accept a stable tenth update."""
    inst = extraction_instrument(monkeypatch, nframes=1, width=5)
    model = np.ones((1, 9, 5))
    updates = []

    def evaluate(expression):
        if expression == 'exp(-0.5*nsig**2)':
            return np.ones((3, 5))
        step = len(updates) + 1
        value = 0.2 if step % 2 else 0.8
        if converges_on_last and step == 10:
            value = updates[-1]
        updates.append(value)
        if len(updates) > 10:
            pytest.fail('Convergence loop exceeded ten iterations')
        return np.full((3, 5), value)

    monkeypatch.setattr(default.ne, 'evaluate', evaluate)
    ratio_args = [model, model.copy(), model / 10, np.full(5, 4.)]
    if converges_on_last:
        spec, _ = inst.ratio_residual_to_trace(*ratio_args)
        np.testing.assert_allclose(spec, 0.1)
    else:
        with pytest.raises(exceptions.SossisseException, match='converge'):
            inst.ratio_residual_to_trace(*ratio_args)
    assert len(updates) == 10


@pytest.mark.parametrize('positions', [np.full(7, np.nan), np.zeros(7)])
def test_no_accepted_trace_columns(monkeypatch, positions):
    """An excluded trace returns aligned NaN spectra without a fit."""
    inst = extraction_instrument(monkeypatch)
    model = np.ones((2, 9, 7))
    monkeypatch.setattr(default.ne, 'evaluate',
                        lambda *unused: pytest.fail('Excluded trace fitted'))
    spec, error = inst.ratio_residual_to_trace(model, model, model, positions)
    assert spec.shape == error.shape == (2, 7)
    assert np.isnan(spec).all() and np.isnan(error).all()


def reference_ratio(model, err, residual, posmax, width=1):
    """Freeze full-cube equations with the column/counter fixes."""
    nframes, nrows, ncols = residual.shape
    spec = np.full((nframes, ncols), np.nan)
    spec_err = np.full_like(spec, np.nan)
    bmask = np.isfinite(posmax) & (posmax > width)
    bmask &= posmax < nrows - width
    rows = np.array([posmax[bmask] + offset
                     for offset in range(-width, width + 1)]).astype(int)
    columns = np.array([np.flatnonzero(bmask)
                        for _ in range(-width, width + 1)]).astype(int)
    if not bmask.any():
        return spec, spec_err
    model_cut = np.array(model[:, rows, columns])
    residual_cut = np.array(residual[:, rows, columns])
    err_cut = np.array(err[:, rows, columns])
    nans = ~np.isfinite(residual_cut + err_cut)
    residual_cut[nans] = 0
    err_cut[nans] = np.inf
    for iframe in range(nframes):
        tmp_model = np.array(model_cut[iframe])
        tmp_residual = np.array(residual_cut[iframe])
        tmp_err = np.array(err_cut[iframe])
        ratio1 = tmp_residual / tmp_model
        noise_ratio1 = tmp_err / tmp_model
        p_valid = np.ones_like(tmp_model)
        max_dp_valid = 1
        counter = 0
        while max_dp_valid > 1e-4 and counter < 10:
            counter += 1
            with np.errstate(all='ignore'):
                weight = p_valid / noise_ratio1 ** 2
                denom = np.nansum(weight, axis=0)
                mean1 = np.nansum(ratio1 * weight, axis=0) / denom
                model2 = np.tile(mean1, (rows.shape[0], 1)) * tmp_model
                nsig = (tmp_residual - model2) / tmp_err
                p1 = default.ne.evaluate('exp(-0.5*nsig**2)')
                p2 = 1e-4
                p_valid_prev = p_valid.copy()
                p_valid = default.ne.evaluate('p1/(p1+p2)')
                dp_valid = np.abs(p_valid - p_valid_prev)
                max_dp_valid = np.max(dp_valid)
                sigsum = np.sum(p_valid / noise_ratio1 ** 2, axis=0)
                sig1 = np.sqrt(1 / sigsum)
        if counter >= 10 and max_dp_valid > 1e-4:
            raise exceptions.SossisseException('Reference did not converge')
        spec[iframe, bmask] = mean1
        spec_err[iframe, bmask] = sig1
    return spec, spec_err


def make_spectral_case(tmp_path, monkeypatch, limit=2, selection=None,
                        traces=(1,)):
    """Write real reconstruction blocks and independent trace calibrations."""
    inst = Instrument.__new__(Instrument)
    inst.name = 'test'
    inst.params = ParamDict()
    settings = dict()
    settings['INPUTS.OBJECTNAME'] = 'target'
    settings['GENERAL.TRACES'] = None if traces == (None,) else list(traces)
    settings['GENERAL.SAVE_RESULTS'] = True
    settings['WLC.GENERAL.LINRECON_TRACE_WIDTH'] = 2
    for key, value in settings.items():
        inst.params.set(key, value, source='test')
    generator = np.random.default_rng(42)
    model = generator.uniform(3., 12., (7, 9, 7))
    residual = generator.normal(0., 0.1, model.shape)
    err = generator.uniform(1., 2., model.shape)
    residual[2, 4, 2] = np.nan
    err[3, 5, 4] = np.inf
    residual[5, :, 6] = np.nan
    valid = np.ones_like(model)
    sources = []
    for begin, end in [(0, 3), (3, 7)]:
        path = tmp_path / f'raw{begin}.fits'
        header = fits.Header(dict(INTSTART=begin + 1, INTEND=end))
        fits.HDUList([fits.PrimaryHDU(header=header),
                      fits.ImageHDU(model[begin:end], name='SCI'),
                      fits.ImageHDU(err[begin:end], name='ERR'),
                      fits.ImageHDU(valid[begin:end],
                                    name='DQ')]).writeto(path)
        sources.append(str(path))
    manager = DataCube(max_integrations=limit)
    manager.normalize(sources, tmp_path / 'initial.fits',
                      integration_slice=selection)
    plans = manager.plan_stage('linrecon', 'spectral-test')
    chosen = slice(*selection) if selection else slice(None)
    originals = dict(RECON=model[chosen], RESIDUAL=residual[chosen],
                     ERR=err[chosen], VALID=valid[chosen])
    offset = 0
    for product in plans:
        stop = offset + product.shape[0]
        arrays = {key: value[offset:stop] for key, value in originals.items()}
        ReconstructionProducts.write_block(product, arrays, '_nrs1')
        offset = stop
    products = ReconstructionProducts([part.filename for part in plans],
                                       limit)
    variables = dict()
    variables['WLC_RECON_FILE'] = str(tmp_path / 'recon_nrs1_jump000.fits')
    variables['WLC_LTBL_FILE'] = str(tmp_path / 'coefficients.csv')
    variables['MEDIAN_IMAGE_FILE'] = str(tmp_path / 'median.fits')
    variables['META'] = dict()
    patterns = dict(SPE_SED_TBL='sed_{objname}_{trace}.csv',
                    RES_NO_GREY_ORD='nogrey_{trace}.fits',
                    RES_GREY_ORD='grey_{trace}.fits',
                    SPECTRA_ORD='spectra_{trace}.fits',
                    WAVE_ORD='wave_{trace}.fits')
    for key, pattern in patterns.items():
        variables[key] = str(tmp_path / pattern)
    monkeypatch.setattr(inst, 'get_variable',
                        lambda key, *unused: variables[key])
    monkeypatch.setattr(inst, 'set_variable', lambda key, value, *unused:
                        variables.update({key: value}))
    monkeypatch.setattr(inst, 'update_meta_data', lambda: None)
    monkeypatch.setattr(inst, 'define_filenames', lambda: None)
    med = np.nanmedian(model, axis=0)
    fits.writeto(variables['MEDIAN_IMAGE_FILE'], med)
    monkeypatch.setattr(inst, 'get_fit_params',
                        lambda median, no_plot: [median] * 5)
    positions = dict()
    for trace in traces:
        centre = 4. if trace in (None, 1) else 5.
        positions[trace] = np.array([np.nan, centre, centre, np.nan,
                                     centre, 0., centre])
    monkeypatch.setattr(inst, 'get_trace_pos', lambda order_num:
                        (positions[order_num], np.ones(7)))
    monkeypatch.setattr(inst, 'get_wavegrid', lambda order_num:
                        np.linspace(1., 2., 7))
    table = Table()
    table['amplitude'] = np.linspace(0.9, 1.1, 7)[chosen]
    table['bjd'] = (2450000. + np.arange(7))[chosen]
    inst.save_wlc_results(products, table)
    monkeypatch.setattr(general.misc, 'sossart', lambda: None)
    monkeypatch.setattr(plots, 'plot_sed', lambda *unused: None)
    monkeypatch.setattr(plots, 'plot_spectral_timeseries',
                        lambda *unused: None)
    return inst, products, table, originals, positions, variables


@pytest.mark.parametrize('limit', [1, 2, 100])
@pytest.mark.parametrize('selection', [None, (1, 6)])
@pytest.mark.parametrize('traces', [(None,), (1,), (1, 2)])
def test_spectral_chunk_matches_full_cube_exactly(tmp_path, monkeypatch,
                                                  limit, selection, traces):
    """Require bit-identical results across blocks, traces and jumps."""
    inst, products, table, original, positions, variables = make_spectral_case(
        tmp_path, monkeypatch, limit, selection, traces)
    expected = dict()
    for trace in traces:
        ratio_args = [original['RECON'], original['ERR'],
                      original['RESIDUAL'], positions[trace]]
        spec, error = reference_ratio(*ratio_args)
        amplitude = np.repeat(np.array(table['amplitude']), spec.shape[1])
        expected[trace] = (spec, error,
                           spec + amplitude.reshape(spec.shape))
    old_load = io.load_fits
    loads = []

    def bounded_load(filename, *load_args, **load_kwargs):
        assert str(filename) == variables['MEDIAN_IMAGE_FILE']
        loads.append(str(filename))
        return old_load(filename, *load_args, **load_kwargs)

    monkeypatch.setattr(io, 'load_fits', bounded_load)
    monkeypatch.setattr(inst, 'load_model',
                        lambda *unused: pytest.fail('Full model copy'))
    original_read = ReconstructionProducts.read_integrations
    reads = []

    def bounded_read(self, start=0, stop=None):
        arrays = original_read(self, start, stop)
        assert arrays['RECON'].shape[0] <= limit
        reads.append(arrays['RECON'].shape[0])
        return arrays

    monkeypatch.setattr(ReconstructionProducts, 'read_integrations',
                        bounded_read)
    storage = general.spectral_extraction_chunk(inst)
    assert len(loads) == len(traces)
    assert len(reads) == len(products.current_infiles) * len(traces)
    assert variables['DATA_N_FRAMES'] == len(products)
    for trace in traces:
        for key, wanted in zip(('spec', 'spec_err', 'spec2'), expected[trace]):
            np.testing.assert_array_equal(storage[trace][key].view(np.uint64),
                                           wanted.view(np.uint64))
        np.testing.assert_array_equal(storage[trace]['ltable']['bjd'],
                                       table['bjd'])
        label = '' if trace is None else trace
        path = variables['SPECTRA_ORD'].format(trace=label)
        np.testing.assert_array_equal(fits.getdata(path, 'RELFLUX'),
                                       expected[trace][2])
        np.testing.assert_array_equal(fits.getdata(path, 'RELFLUX_ERROR'),
                                       expected[trace][1])
        np.testing.assert_array_equal(fits.getdata(path, 'BJD'), table['bjd'])


def test_block_dimensions_do_not_change_instrument_metadata(monkeypatch):
    """Extraction uses its block shape even with full-observation metadata."""
    inst = extraction_instrument(monkeypatch, nframes=1000, width=7)
    model = np.ones((1, 9, 7))
    residual = np.full_like(model, 0.1)
    error = model.copy()
    positions = np.full(7, 4.)
    original = [array.copy() for array in (model, residual, error)]
    expected = reference_ratio(model, error, residual, positions)
    actual = inst.ratio_residual_to_trace(model, error, residual, positions)
    for array, wanted in zip(actual, expected):
        np.testing.assert_array_equal(array.view(np.uint64),
                                       wanted.view(np.uint64))
    assert inst.get_variable('DATA_N_FRAMES', 'test') == 1000
    for array, wanted in zip((model, residual, error), original):
        np.testing.assert_array_equal(array, wanted)


def test_missing_reconstruction_fails_before_extraction(tmp_path, monkeypatch):
    from pathlib import Path

    inst, products, _, _, _, _ = make_spectral_case(tmp_path, monkeypatch)
    Path(products.current_infiles[-1]).unlink()
    monkeypatch.setattr(inst, 'ratio_residual_to_trace',
                        lambda *unused: pytest.fail('Incomplete inputs read'))
    with pytest.raises(OSError):
        general.spectral_extraction_chunk(inst)


def test_failed_extraction_does_not_save_spectra(tmp_path, monkeypatch):
    from pathlib import Path

    inst, products, _, _, _, variables = make_spectral_case(
        tmp_path, monkeypatch)
    inputs = products.current_infiles.copy()

    def broken(*unused):
        raise exceptions.SossisseException('injected non-convergence')

    monkeypatch.setattr(inst, 'ratio_residual_to_trace', broken)
    with pytest.raises(exceptions.SossisseException, match='injected'):
        general.spectral_extraction_chunk(inst)
    assert products.current_infiles == inputs
    assert not Path(variables['SPECTRA_ORD'].format(trace=1)).exists()


def test_real_linear_fit_to_spectral_extraction_exact(tmp_path, monkeypatch):
    """Use actual fitted RECON/RESIDUAL/ERR outputs as extraction inputs."""
    from test_datacube_reconstruction import make_case, run_fit
    from test_datacube_reconstruction import collect_products

    inst, manager, med, basis, _, _ = make_case(tmp_path, monkeypatch, limit=2)
    table, products = run_fit(inst, manager, med, basis)
    table['bjd'] = 2450000. + np.arange(len(table))
    inst.save_wlc_results(products, table)
    inst.params.set('INPUTS.OBJECTNAME', 'target', source='test')
    inst.params.set('GENERAL.TRACES', [1], source='test')
    inst.params.set('GENERAL.SAVE_RESULTS', False, source='test')
    inst.params.set('WLC.GENERAL.LINRECON_TRACE_WIDTH', 2, source='test')
    med_file = str(tmp_path / 'median.fits')
    fits.writeto(med_file, med)
    inst.set_variable('MEDIAN_IMAGE_FILE', med_file)
    posmax = np.array([np.nan, 3., 3., np.nan, 3., 3., 0., 3.])
    monkeypatch.setattr(inst, 'get_trace_pos', lambda order_num:
                        (posmax, np.ones(8)))
    monkeypatch.setattr(inst, 'get_wavegrid', lambda order_num:
                        np.linspace(1., 2., 8))
    monkeypatch.setattr(inst, 'get_fit_params',
                        lambda median, no_plot: [median] * 5)
    monkeypatch.setattr(inst, 'define_filenames', lambda: None)
    monkeypatch.setattr(general.misc, 'sossart', lambda: None)
    monkeypatch.setattr(plots, 'plot_sed', lambda *unused: None)
    monkeypatch.setattr(plots, 'plot_spectral_timeseries',
                        lambda *unused: None)
    full = collect_products(products)
    ratio_args = [full['RECON'], full['ERR'], full['RESIDUAL'], posmax]
    expected, expected_err = reference_ratio(*ratio_args)
    amplitudes = np.repeat(np.array(table['amplitude']), expected.shape[1])
    expected_total = expected + amplitudes.reshape(expected.shape)
    storage = general.spectral_extraction_chunk(inst)[1]
    for key, wanted in [('spec', expected), ('spec_err', expected_err),
                         ('spec2', expected_total)]:
        np.testing.assert_array_equal(storage[key].view(np.uint64),
                                       wanted.view(np.uint64))