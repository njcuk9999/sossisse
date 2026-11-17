import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from aperocore.constants.param_functions import ParamDict
from sossisse.instruments.jwst_nirspec import JWST_NIRSPEC


@pytest.mark.parametrize('detector, width', [('NRS1', 6), ('NRS2', 4)])
@pytest.mark.parametrize('flag_cds', [False, True])
@pytest.mark.parametrize('bin_size', [1, 2])
@pytest.mark.parametrize('image_width', [4, 8])
@pytest.mark.parametrize('data_width', [None, 3, 4, 8])
@pytest.mark.parametrize('force_pixel_grid', [False, True])
def test_load_cube_pixel_grid(tmp_path, monkeypatch, detector, width,
                              flag_cds, bin_size, image_width, data_width,
                              force_pixel_grid):
    subsize = width
    width = subsize if data_width is None else data_width
    bounds = (1, 5) if subsize == 6 else (0, 3)
    pixel_start, pixel_end = bounds if force_pixel_grid else (0, width)
    output_width = len(range(width)[pixel_start:pixel_end])
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('WLC.INPUTS', {'CDS_IDS': [0, 1], 'CDS_RON': 4},
                          source='test')
    monkeypatch.setattr(instrument, '_force_pixel_grid',
                        lambda filename: bounds)
    filenames = []
    expected = [[], [], []]
    bin_calls = []

    def bin_cube(data, bin_type='Flux'):
        # Check that pixel selection happens before binning.
        assert data.shape[-1] == output_width
        bin_calls.append(bin_type)
        grouped = data.reshape((-1, bin_size) + data.shape[1:])
        if bin_type == 'Error':
            return np.sqrt(np.sum(grouped ** 2, axis=1))
        if bin_type == 'DQ':
            return np.max(grouped, axis=1)
        return np.sum(grouped, axis=1)

    monkeypatch.setattr(instrument, 'bin_cube', bin_cube)
    for segment, frames in enumerate([2, 4]):
        shape = (frames, 2, 2, width) if flag_cds else (frames, 2, width)
        data = np.arange(np.prod(shape), dtype=float).reshape(shape) + segment
        header = fits.Header({'DETECTOR': detector, 'SUBSIZE1': subsize})
        hdus = [fits.PrimaryHDU(header=header), fits.ImageHDU(data)]
        selected = data[..., pixel_start:pixel_end]
        binned = selected.reshape((-1, bin_size) + selected.shape[1:]).sum(axis=1)
        if flag_cds:
            science = binned[:, 1] - binned[:, 0]
            outputs = (science, np.sqrt(np.abs(science) + 4),
                       ~np.isfinite(science))
        else:
            error = data + 1
            dq = (data % 4).astype(np.uint32)
            hdus.extend([fits.ImageHDU(error), fits.ImageHDU(dq)])
            error_bins = error[..., pixel_start:pixel_end].reshape(
                (-1, bin_size, 2, output_width))
            dq_bins = dq[..., pixel_start:pixel_end].reshape(
                (-1, bin_size, 2, output_width))
            outputs = (binned, np.sqrt(np.sum(error_bins ** 2, axis=1)),
                       np.max(dq_bins, axis=1))
        for container, output in zip(expected, outputs):
            container.append(output)
        filename = tmp_path / f'segment-{segment}.fits'
        fits.HDUList(hdus).writeto(filename)
        filenames.append(str(filename))
    instrument.params.set('GENERAL.FILES', filenames, source='test')

    result = instrument.load_cube(6 // bin_size, [2, image_width], flag_cds,
                                  force_pixel_grid=force_pixel_grid)

    for actual, segments in zip(result, expected):
        np.testing.assert_allclose(actual, np.concatenate(segments))
    assert bin_calls == (['Flux'] if flag_cds else ['Flux', 'Error', 'DQ']) * 2


@pytest.mark.parametrize('ndim', [1, 2, 3, 4])
@pytest.mark.parametrize('width', [3, 4, 6, 8])
@pytest.mark.parametrize('subsize, force_pixel_grid',
                         [(None, False), (6, False), (6, True)])
def test_load_data_pixel_grid(tmp_path, monkeypatch, ndim, width, subsize,
                              force_pixel_grid):
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    monkeypatch.setattr(instrument, '_force_pixel_grid', lambda filename: (1, 5))
    data = np.arange(2 ** (ndim - 1) * width).reshape(
        (2,) * (ndim - 1) + (width,))
    header = fits.Header()
    if subsize is not None:
        header['DETECTOR'] = 'NRS1'
        header['SUBSIZE1'] = subsize
    filename = tmp_path / 'data.fits'
    fits.PrimaryHDU(data, header=header).writeto(filename)

    actual = instrument.load_data(str(filename),
                                  force_pixel_grid=force_pixel_grid)

    expected = data[..., 1:5] if force_pixel_grid else data
    np.testing.assert_array_equal(actual, expected)


def test_load_data_preserves_table(tmp_path):
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    header = fits.Header({'DETECTOR': 'NRS1', 'SUBSIZE1': 6})
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='time', format='D', array=np.arange(6.0))])
    filename = tmp_path / 'table.fits'
    fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(filename)

    actual = instrument.load_data(str(filename), ext=1)

    np.testing.assert_array_equal(actual['time'], np.arange(6.0))


def test_load_data_default_preserves_detector_width(tmp_path, monkeypatch):
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    monkeypatch.setattr(instrument, 'get_wavegrid',
                        lambda: pytest.fail('Default loading must not select a grid'))
    data = np.arange(4096).reshape(2, 2048)
    header = fits.Header({'DETECTOR': 'NRS1', 'SUBSIZE1': 2048})
    filename = tmp_path / 'processed.fits'
    fits.PrimaryHDU(data, header=header).writeto(filename)

    actual = instrument.load_data(str(filename))

    np.testing.assert_array_equal(actual, data)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('limit', [1, 1000])
def test_load_data_with_dq_pixel_grid(tmp_path, monkeypatch, cached, limit):
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.USE_TEMPORARY', cached, source='test')
    instrument.params.set('GENERAL.MAX_INTEGRATIONS', limit, source='test')
    instrument.params.set('GENERAL.MAX_MATH_MEMORY_GB', 8.0, source='test')
    instrument.params.set('WLC.LMODEL.DATA_BIN_TIME', False, source='test')
    instrument._jump_slice = None
    variables = {}
    monkeypatch.setattr(instrument, 'set_variable', variables.__setitem__)
    monkeypatch.setattr(instrument, 'update_meta_data', lambda: None)
    monkeypatch.setattr(instrument, '_force_pixel_grid', lambda filename: (1, 5))
    monkeypatch.setattr(instrument, 'bin_cube',
                        lambda data, bin_type='Flux', get_shape=False:
                        list(data.shape) if get_shape else data)
    data = np.arange(24.0).reshape(2, 2, 6)
    error = data + 1
    dq = (data % 4).astype(np.uint32)
    header = fits.Header(dict(DETECTOR='NRS1', SUBSIZE1=6,
                             INTSTART=1, INTEND=2))
    filename = tmp_path / 'raw.fits'
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.ImageHDU(data, name='SCI'),
                  fits.ImageHDU(error), fits.ImageHDU(dq)]).writeto(filename)
    instrument.params.set('GENERAL.FILES', [str(filename)], source='test')
    cache_paths = {}
    for key, array in zip(['TEMP_INI_CUBE', 'TEMP_INI_ERR', 'TEMP_INI_DQ'],
                          [data, error, dq]):
        path = tmp_path / f'{key}.fits'
        fits.PrimaryHDU(array, header=header).writeto(path)
        cache_paths[key] = str(path)
    monkeypatch.setattr(instrument, 'get_variable',
                        lambda key, func_name: cache_paths[key])

    datacube = instrument.load_data_with_dq()

    assert datacube.current_infiles
    assert not hasattr(datacube, 'in_memory')
    blocks = list(datacube.iter_integrations())
    actual = tuple(np.concatenate([arrays[name] for _, arrays in blocks])
                   for name in ('FLUX', 'ERR', 'DQ'))
    for result, original in zip(actual, [data, error, dq]):
        expected = original[..., 1:5]
        np.testing.assert_array_equal(result, expected)
    assert variables['DATA_X_SIZE'] == 4
    initial_files = datacube.current_infiles.copy()
    datacube = instrument.load_data_with_dq()
    assert datacube.current_infiles == initial_files


@pytest.mark.parametrize('detector', ['NRS1', 'NRS2'])
def test_force_pixel_grid_uses_slit_cutout(monkeypatch, caplog, detector):
    from jwst import datamodels
    from jwst.assign_wcs import AssignWcsStep
    from jwst.extract_2d import nirspec

    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.WAVE_FILE', 'wave.fits', source='test')
    instrument.params.set('GENERAL.WAVE_FILE_TYPE', 'fits', source='test')
    instrument.params.set('PATHS.CALIBPATH', '.', source='test')
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: np.arange(4))
    cube = MagicMock()
    result = MagicMock()
    cube.__enter__.return_value = cube
    result.__enter__.return_value = result
    slit = object()
    slit_model = MagicMock()
    cube.meta.instrument.detector = detector
    result.meta.wcs.get_transform.return_value.slits = [slit]
    open_model = MagicMock(return_value=cube)
    logger = logging.getLogger('jwst.assign_wcs.test')
    caplog.set_level(logging.INFO)

    def run_assign_wcs(cube, configure_log):
        logger.info('pipeline progress')
        logger.warning('pipeline warning')
        return result

    assign_wcs = MagicMock(side_effect=run_assign_wcs)
    process_slit = MagicMock(return_value=(slit_model, 1, 5, 0, 2))
    monkeypatch.setattr(datamodels, 'open', open_model)
    monkeypatch.setattr(AssignWcsStep, 'call', assign_wcs)
    monkeypatch.setattr(nirspec, 'process_slit', process_slit)
    monkeypatch.setattr(fits, 'getheader',
                        lambda filename, ext=0: {'DETECTOR': detector})

    previous_disable = logging.root.manager.disable
    assert instrument._force_pixel_grid('raw.fits') == (1, 5)
    open_model.assert_called_once_with('raw.fits')
    assign_wcs.assert_called_once_with(cube, configure_log=False)
    result.meta.wcs.get_transform.assert_called_once_with('gwa', 'slit_frame')
    process_slit.assert_called_once_with(result, slit)
    slit_model.close.assert_called_once_with()
    assert 'pipeline progress' not in caplog.text
    assert 'pipeline warning' in caplog.text
    assert logging.root.manager.disable == previous_disable


def test_force_pixel_grid_cache_is_per_detector_and_wave_solution(
        tmp_path, monkeypatch):
    from jwst import datamodels
    from jwst.assign_wcs import AssignWcsStep
    from jwst.extract_2d import nirspec

    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.WAVE_FILE', 'shared-wave.fits',
                          source='test')
    instrument.params.set('GENERAL.WAVE_FILE_TYPE', 'fits', source='test')
    instrument.params.set('PATHS.CALIBPATH', str(tmp_path), source='test')
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: np.arange(4))

    cubes = {}
    results = {}
    slit_models = {}
    bounds = {'NRS1': (1, 5), 'NRS2': (2, 6)}
    for detector in bounds:
        cube = MagicMock()
        cube.__enter__.return_value = cube
        cube.meta.instrument.detector = detector
        result = MagicMock()
        result.__enter__.return_value = result
        slit = object()
        result.meta.wcs.get_transform.return_value.slits = [slit]
        cubes[detector] = cube
        results[detector] = result
        slit_models[detector] = MagicMock()

    open_model = MagicMock(side_effect=lambda filename: cubes[filename])
    assign_wcs = MagicMock(
        side_effect=lambda cube, configure_log:
        results[cube.meta.instrument.detector])

    def get_slit_bounds(result, slit):
        detector = next(detector for detector, item in results.items()
                        if item is result)
        return (slit_models[detector], *bounds[detector], 0, 2)

    process_slit = MagicMock(side_effect=get_slit_bounds)
    monkeypatch.setattr(datamodels, 'open', open_model)
    monkeypatch.setattr(AssignWcsStep, 'call', assign_wcs)
    monkeypatch.setattr(nirspec, 'process_slit', process_slit)
    monkeypatch.setattr(fits, 'getheader',
                        lambda filename, ext=0: {'DETECTOR': filename})

    assert instrument._force_pixel_grid('NRS1') == bounds['NRS1']
    assert instrument._force_pixel_grid('NRS1') == bounds['NRS1']
    assert instrument._force_pixel_grid('NRS2') == bounds['NRS2']
    instrument.params.set('GENERAL.WAVE_FILE', 'other-wave.fits',
                          source='test')
    assert instrument._force_pixel_grid('NRS1') == bounds['NRS1']

    assert assign_wcs.call_count == 3
    assert process_slit.call_count == 3
    assert open_model.call_count == 3


@pytest.mark.parametrize('disabled_level', [logging.NOTSET, logging.WARNING])
def test_force_pixel_grid_restores_logging_on_error(monkeypatch, disabled_level):
    from jwst import datamodels

    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('GENERAL.WAVE_FILE', 'wave.fits', source='test')
    instrument.params.set('GENERAL.WAVE_FILE_TYPE', 'fits', source='test')
    instrument.params.set('PATHS.CALIBPATH', '.', source='test')
    previous_disable = logging.root.manager.disable
    monkeypatch.setattr(datamodels, 'open', MagicMock(side_effect=ValueError))
    monkeypatch.setattr(fits, 'getheader',
                        lambda filename, ext=0: {'DETECTOR': 'NRS1'})

    logging.disable(disabled_level)
    try:
        with pytest.raises(ValueError):
            instrument._force_pixel_grid('raw.fits')
        assert logging.root.manager.disable == disabled_level
    finally:
        logging.disable(previous_disable)
