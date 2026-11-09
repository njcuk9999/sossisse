"""Normalized FITS products preserve values and bound section reads."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.hdu.image import Section

from sossisse.general.datacube import DataCube


def normalize(filenames, max_integrations=1000, **normalize_kwargs):
    """Use the direct manager API with a test-local product prefix."""
    manager = DataCube(max_integrations=max_integrations)
    directory = Path(filenames[0]).parent if filenames else Path('.')
    prefix = directory / 'initial.fits'
    return manager.normalize(filenames, prefix, **normalize_kwargs)


def write_raw(tmp_path, name='raw.fits', start=7, count=5, cds=False,
              named=True, synthetic=False, width=4):
    shape = (count, 2, 2, width) if cds else (count, 2, width)
    science = np.arange(np.prod(shape), dtype=float).reshape(shape) - 3
    error = np.abs(science) + 2
    dq = (np.arange(np.prod(shape)).reshape(shape) % 4).astype(np.uint32)
    dq.flat[0] = 2 ** 31 + 1
    header = fits.Header({'INTSTART': start, 'INTEND': start + count - 1})
    hdus = fits.HDUList([fits.PrimaryHDU(header=header)])
    products = [('SCI', science)]
    if not cds and not synthetic:
        products.extend([('ERR', error), ('DQ', dq)])
    for label, data in products:
        hdus.append(fits.ImageHDU(data, name=label if named else None))
    filename = tmp_path / name
    hdus.writeto(filename)
    return filename, (science, error, dq)


@pytest.fixture
def section_reads(monkeypatch):
    reads = []
    original = Section.__getitem__

    def bounded_section(section, selection):
        integration = selection[0]
        if isinstance(integration, (int, np.integer)):
            reads.append((integration, integration + 1))
        else:
            reads.append((integration.start, integration.stop))
        return original(section, selection)

    monkeypatch.setattr(Section, '__getitem__', bounded_section)
    return reads


def collect(cube):
    """Collect named blocks into a test-only SCI/ERR/DQ comparison tuple."""
    blocks = list(cube.iter_integrations())
    return tuple(np.concatenate([arrays[name] for _, arrays in blocks])
                 for name in ('FLUX', 'ERR', 'DQ'))


@pytest.mark.parametrize('limit', [2, 20])
@pytest.mark.parametrize('named', [False, True])
def test_order_values_and_header_only_count(tmp_path, section_reads, limit,
                                            named):
    first, first_arrays = write_raw(tmp_path, 'z.fits', named=named)
    second_outputs = write_raw(tmp_path, 'a.fits', start=12, named=named)
    second, second_arrays = second_outputs
    filenames = [second, first]
    assert DataCube.input_n_integrations(filenames) == 10
    assert DataCube.input_n_integrations(filenames, bin_size=2) == 4
    assert section_reads == []
    cube = normalize(filenames, max_integrations=limit)
    assert cube.filenames == (first, second)
    assert cube.integration_ranges == ((7, 11), (12, 16))
    assert not cube.flag_cds
    assert cube.current_infiles
    assert section_reads
    for actual, left, right in zip(collect(cube), first_arrays, second_arrays):
        assert actual.dtype == float
        np.testing.assert_equal(actual, np.concatenate([left, right]))
    assert max(stop - start for start, stop in section_reads) <= limit


@pytest.mark.parametrize('limit', [2, 20])
@pytest.mark.parametrize('bin_size', [1, 2, 5])
def test_binning_crop_jump_and_bounded_reads(tmp_path, section_reads, limit,
                                            bin_size):
    first, arrays = write_raw(tmp_path, count=11)
    second, _ = write_raw(tmp_path, 'second.fits', count=11, start=18)
    expected = []
    for index, array in enumerate(arrays):
        selected = array[..., 1:3][:11 // bin_size * bin_size].astype(float)
        grouped = selected.reshape((-1, bin_size, 2, 2))
        if index == 1:
            binned = np.sqrt(np.sum(grouped ** 2, axis=1))
        elif index == 2:
            binned = np.nanmax(grouped, axis=1)
        else:
            binned = np.sum(grouped, axis=1)
        expected.append(np.concatenate([binned, binned])[1:-1])
    cube = normalize(
        [second, first], max_integrations=limit, bin_size=bin_size,
        pixel_bounds=lambda filename: (1, 3),
        integration_slice=(1, 2 * (11 // bin_size) - 1))
    for actual, wanted in zip(collect(cube), expected):
        np.testing.assert_allclose(actual, wanted)
    assert max(stop - start for start, stop in section_reads) <= limit
    last_complete_bin = (11 // bin_size) * bin_size
    assert all(stop <= last_complete_bin for _, stop in section_reads)


@pytest.mark.parametrize('limit', [1, 20])
def test_cds_conversion_after_binning(tmp_path, section_reads, limit):
    filename, arrays = write_raw(tmp_path, count=9, cds=True)
    calls = []

    def convert(data):
        calls.append(data.copy())
        science = data[:, 1] - data[:, 0]
        return science, np.sqrt(np.abs(science) + 4), ~np.isfinite(science)

    cube = normalize([filename], max_integrations=limit, bin_size=4,
                               from_cds=convert)
    assert cube.flag_cds
    actual = collect(cube)
    binned = arrays[0][:8].reshape(2, 4, 2, 2, 4).sum(axis=1)
    np.testing.assert_equal(np.concatenate(calls), binned)
    science = binned[:, 1] - binned[:, 0]
    for result, expected in zip(actual, (science, np.sqrt(abs(science) + 4),
                                         np.zeros_like(science))):
        np.testing.assert_allclose(result, expected)
    assert max(stop - start for start, stop in section_reads) <= limit


@pytest.mark.parametrize('limit', [1, 20])
def test_synthetic_fgs_errors(tmp_path, limit):
    filename, arrays = write_raw(tmp_path, synthetic=True)
    cube = normalize([filename], max_integrations=limit,
                               synthetic_errors=True, bin_size=2)
    science = arrays[0][:4].reshape(2, 2, 2, 4).sum(axis=1)
    for result, expected in zip(collect(cube), (science, np.sqrt(abs(science)),
                                               np.zeros_like(science))):
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('limit', [2, 20])
def test_cache(tmp_path, section_reads, limit):
    filename, arrays = write_raw(tmp_path)
    initial = normalize([filename], max_integrations=limit)
    section_reads.clear()
    cube = DataCube(initial.current_infiles, max_integrations=limit)
    assert cube.current_infiles == initial.current_infiles
    assert not cube.flag_cds
    assert cube.integration_ranges == ()
    assert section_reads == []
    for actual, expected in zip(collect(cube), arrays):
        np.testing.assert_equal(actual, expected)
    assert max(stop - start for start, stop in section_reads) <= limit


@pytest.mark.parametrize('key,value', [('INTSTART', None), ('INTEND', None),
                                      ('INTSTART', 'bad'), ('INTSTART', 0),
                                      ('INTSTART', 1.5), ('INTEND', 6),
                                      ('INTEND', 99)])
def test_invalid_integration_headers(tmp_path, section_reads, key, value):
    filename, _ = write_raw(tmp_path)
    with fits.open(filename, mode='update') as hdul:
        header = cast(fits.PrimaryHDU, hdul[0]).header
        if value is None:
            del header[key]
        else:
            header[key] = value
    for factory in (normalize, DataCube.input_n_integrations):
        with pytest.raises(ValueError, match=str(filename)):
            factory([filename])
    assert section_reads == []


@pytest.mark.parametrize('start,match', [(11, 'overlap'), (13, 'gap')])
def test_discontinuous_ranges(tmp_path, section_reads, start, match):
    first, _ = write_raw(tmp_path)
    second, _ = write_raw(tmp_path, 'second.fits', start=start)
    for factory in (normalize, DataCube.input_n_integrations):
        with pytest.raises(ValueError, match=match):
            factory([second, first])
    assert section_reads == []


@pytest.mark.parametrize('selection', [(-1, 2), (0, 6), (2, 2), (3, 1),
                                      (0, 1.5), (True, 2), (1,), 'bad'])
def test_invalid_jump(tmp_path, section_reads, selection):
    filename, _ = write_raw(tmp_path)
    with pytest.raises(ValueError, match='integration_slice'):
        normalize([filename], integration_slice=selection)
    assert section_reads == []


def test_jump_selected_before_normalizing(tmp_path, section_reads):
    filename, arrays = write_raw(tmp_path, count=12)
    cube = normalize([filename], max_integrations=2,
                               integration_slice=(7, 9))
    assert all(7 <= start < stop <= 9 for start, stop in section_reads)
    section_reads.clear()
    np.testing.assert_equal(collect(cube)[0], arrays[0][7:9])
    assert all(0 <= start < stop <= 2 for start, stop in section_reads)


def test_inconsistent_shapes_and_mixed_cds(tmp_path, section_reads):
    first, _ = write_raw(tmp_path)
    wrong, _ = write_raw(tmp_path, 'wrong.fits', start=12, width=3)
    cds, _ = write_raw(tmp_path, 'cds.fits', start=12, cds=True)
    for second, match in [(wrong, 'shapes'), (cds, 'mixed CDS')]:
        with pytest.raises(ValueError, match=match):
            normalize([first, second])
    with pytest.raises(ValueError, match='from_cds'):
        normalize([cds])
    assert section_reads == []


def test_invalid_cache(tmp_path, section_reads):
    filenames = []
    for index, shape in enumerate([(5, 2, 4), (5, 2, 4), (6, 2, 4)]):
        filename = tmp_path / f'{index}.fits'
        fits.writeto(filename, np.zeros(shape))
        filenames.append(filename)
    with pytest.raises(ValueError, match='normalized product header'):
        DataCube(filenames)
    assert section_reads == []


@pytest.mark.parametrize('limit', [2, 20])
def test_no_data_access_and_open_options(tmp_path, monkeypatch, section_reads,
                                          limit):
    filename, expected = write_raw(tmp_path)
    initial = normalize([filename], max_integrations=limit)
    section_reads.clear()
    original_open = fits.open

    def checked_open(*filenames, **options):
        assert options['memmap'] is False
        assert options['lazy_load_hdus'] is True
        return original_open(*filenames, **options)

    def forbidden_data(hdu):
        pytest.fail('Full image .data access is forbidden')

    monkeypatch.setattr(fits, 'open', checked_open)
    monkeypatch.setattr(fits.PrimaryHDU, 'data', property(forbidden_data))
    monkeypatch.setattr(fits.ImageHDU, 'data', property(forbidden_data))
    assert DataCube.input_n_integrations([filename]) == 5
    assert section_reads == []
    cube = DataCube(initial.current_infiles, max_integrations=limit)
    assert section_reads == []
    for actual, wanted in zip(collect(cube), expected):
        np.testing.assert_equal(actual, wanted)


def test_named_extensions_override_numeric_positions(tmp_path):
    filename, expected = write_raw(tmp_path)
    with fits.open(filename, mode='update') as hdul:
        hdul.insert(1, fits.ImageHDU(np.zeros((1, 1)), name='UNRELATED'))
    cube = normalize([filename])
    for actual, wanted in zip(collect(cube), expected):
        np.testing.assert_equal(actual, wanted)


@pytest.mark.parametrize('ndim', [2, 5])
def test_invalid_image_dimensions(tmp_path, section_reads, ndim):
    filename = tmp_path / 'dimensions.fits'
    header = fits.Header({'INTSTART': 1, 'INTEND': 2})
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.ImageHDU(np.zeros((2,) * ndim))]).writeto(filename)
    with pytest.raises(ValueError, match='dimensions'):
        normalize([filename])
    assert section_reads == []


def test_invalid_companion_extension_shape(tmp_path, section_reads):
    filename, _ = write_raw(tmp_path)
    with fits.open(filename, mode='update') as hdul:
        hdul[2] = fits.ImageHDU(np.ones((4, 2, 4)), name='ERR')
    with pytest.raises(ValueError, match='extension shapes'):
        normalize([filename], max_integrations=1)
    assert section_reads == []


@pytest.mark.parametrize('bounds', [(2, 2), (8, 10), (1,), (1, 2.5), None])
def test_invalid_pixel_bounds(tmp_path, section_reads, bounds):
    filename, _ = write_raw(tmp_path)
    with pytest.raises(ValueError, match='pixel_bounds'):
        normalize([filename], pixel_bounds=lambda filename: bounds)
    assert section_reads == []


@pytest.mark.parametrize('bin_size', [0, -1, True, 1.5])
def test_invalid_bin_size(tmp_path, bin_size):
    filename, _ = write_raw(tmp_path)
    for factory in (normalize, DataCube.input_n_integrations):
        with pytest.raises(ValueError, match='bin_size'):
            factory([filename], bin_size=bin_size)


def test_empty_input_and_no_complete_bins(tmp_path, section_reads):
    for factory in (normalize, DataCube.input_n_integrations):
        with pytest.raises(ValueError, match='No input'):
            factory([])
    filename, _ = write_raw(tmp_path)
    assert DataCube.input_n_integrations([filename], bin_size=6) == 0
    with pytest.raises(ValueError, match='nonempty'):
        normalize([filename], bin_size=6)
    assert section_reads == []


@pytest.mark.parametrize('limit', [1, 20])
def test_nanmax_dq_across_streamed_chunks(tmp_path, limit):
    filename, _ = write_raw(tmp_path, count=6)
    quality = np.arange(48., dtype=float).reshape(6, 2, 4)
    quality[::2] = np.nan
    with fits.open(filename, mode='update') as hdul:
        hdul[3] = fits.ImageHDU(quality, name='DQ')
    with np.errstate(invalid='ignore'):
        if limit == 1:
            with pytest.warns(RuntimeWarning, match='All-NaN slice'):
                cube = normalize([filename], max_integrations=limit,
                                 bin_size=3)
                actual = collect(cube)[2]
        else:
            cube = normalize([filename], max_integrations=limit, bin_size=3)
            actual = collect(cube)[2]
    expected = np.nanmax(quality.reshape(2, 3, 2, 4), axis=1)
    np.testing.assert_equal(actual, expected)