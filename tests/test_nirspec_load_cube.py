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
    pixel_start, pixel_end = (subsize - 4, subsize) if force_pixel_grid else (0, width)
    output_width = len(range(width)[pixel_start:pixel_end])
    instrument = JWST_NIRSPEC.__new__(JWST_NIRSPEC)
    instrument.params = ParamDict()
    instrument.params.set('WLC.INPUTS', {'CDS_IDS': [0, 1], 'CDS_RON': 4},
                          source='test')
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: np.arange(4))
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
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: np.arange(4))
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

    expected = data[..., 2:6] if force_pixel_grid else data
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
    monkeypatch.setattr(instrument, 'get_wavegrid', lambda: np.arange(4))
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
                   for name in ('CUBE', 'ERR', 'DQ'))
    for result, original in zip(actual, [data, error, dq]):
        expected = original[..., 2:6]
        np.testing.assert_array_equal(result, expected)
    assert variables['DATA_X_SIZE'] == 4
    initial_files = datacube.current_infiles.copy()
    datacube = instrument.load_data_with_dq()
    assert datacube.current_infiles == initial_files
