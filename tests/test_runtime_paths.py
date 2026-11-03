from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from aperocore.constants.param_functions import ParamDict
from sossisse.core import const_funcs
from sossisse.core import constants
from sossisse.core import exceptions
from sossisse.core import io


PATH_DEFAULTS = {
    'RUN_PATH': None,
    'YAMLPATH': None,
    'RAWPATH': None,
    'CALIBPATH': None,
    'TEMP_PATH': None,
    'PLOT_PATH': None,
    'OTHER_PATH': None,
    'OUT_PATH': None,
}
CHILD_PATHS = {
    'RAWPATH': 'inputs',
    'CALIBPATH': 'calibration',
    'TEMP_PATH': 'temporary',
    'PLOT_PATH': 'plots',
    'OTHER_PATH': 'other',
    'OUT_PATH': 'outputs',
}


def make_params(root, source='SOSSISSE', project='visit-one', **overrides):
    params = ParamDict()
    values = {
        '__SOURCE__': source,
        'INPUTS.SOSSIOPATH': str(root),
        'INPUTS.PROJECT_NAME': project,
        'INPUTS.INSTRUMENTMODE': 'JWST.NIRISS.SOSS',
        'INPUTS.OBJECTNAME': 'target',
        'INPUTS.PARAM_FILE': None,
        'GENERAL.FILES': None,
        'PLOTS.SHOW': True,
    }
    path_values = {f'PATHS.{key}': value
                   for key, value in PATH_DEFAULTS.items()}
    values.update(path_values)
    values.update(overrides)
    for key, value in values.items():
        params.set(key, value, source='test_runtime_paths')
    return params


@pytest.mark.parametrize('source', ['SOSSISSE', 'POGOS'])
def test_project_runtime_paths(tmp_path, source):
    params = make_params(tmp_path, source)
    result = const_funcs.run_time_params(params, only_create=True)
    run_path = tmp_path / 'projects' / 'visit-one' / 'sossisse'
    assert Path(result['PATHS.RUN_PATH']) == run_path
    for key, directory in CHILD_PATHS.items():
        assert Path(result[f'PATHS.{key}']) == run_path / directory
        assert (run_path / directory).is_dir()
    if source == 'SOSSISSE':
        assert Path(result['PATHS.YAMLPATH']) == run_path / 'yamls'
        assert (run_path / 'yamls').is_dir()
    else:
        assert result['PATHS.YAMLPATH'] is None
        assert not (run_path / 'yamls').exists()


@pytest.mark.parametrize('source', ['SOSSISSE', 'POGOS'])
def test_repeat_runs_reuse_project(tmp_path, source):
    first = const_funcs.run_time_params(make_params(tmp_path, source), True)
    marker = Path(first['PATHS.OUT_PATH']) / 'existing.txt'
    marker.write_text('previous output')
    for params in [first, make_params(tmp_path, source)]:
        repeated = const_funcs.run_time_params(params, True)
        for key in CHILD_PATHS:
            assert repeated[f'PATHS.{key}'] == first[f'PATHS.{key}']
    assert marker.read_text() == 'previous output'


@pytest.mark.parametrize('source', ['SOSSISSE', 'POGOS'])
def test_absolute_run_override(tmp_path, source):
    custom_run = tmp_path / 'custom-run'
    params = make_params(tmp_path / 'root', source,
                         **{'PATHS.RUN_PATH': str(custom_run)})
    result = const_funcs.run_time_params(params, True)
    assert result['PATHS.RUN_PATH'] == str(custom_run)
    for key, directory in CHILD_PATHS.items():
        assert Path(result[f'PATHS.{key}']) == custom_run / directory


@pytest.mark.parametrize('key', [*CHILD_PATHS, 'YAMLPATH'])
@pytest.mark.parametrize('source', ['SOSSISSE', 'POGOS'])
def test_absolute_child_override(tmp_path, key, source):
    override = tmp_path / 'shared' / key.lower()
    params = make_params(tmp_path / 'root', source,
                         **{f'PATHS.{key}': str(override)})
    result = const_funcs.run_time_params(params, True)
    assert result[f'PATHS.{key}'] == str(override)
    if key != 'YAMLPATH' or source == 'SOSSISSE':
        assert override.is_dir()


@pytest.mark.parametrize('project', [None, '', '  ', '.', '..', '../other',
                                   '/absolute', 'nested/project'])
def test_project_name_required(tmp_path, project):
    params = make_params(tmp_path, project=project)
    with pytest.raises(exceptions.SossisseConstantException,
                       match='PROJECT_NAME'):
        const_funcs.run_time_params(params, True)


def test_projects_are_isolated(tmp_path):
    first = const_funcs.run_time_params(make_params(tmp_path), True)
    second = const_funcs.run_time_params(
        make_params(tmp_path, project='visit-two'), True)
    assert first['PATHS.RUN_PATH'] != second['PATHS.RUN_PATH']
    assert Path(second['PATHS.RUN_PATH']) == (
        tmp_path / 'projects' / 'visit-two' / 'sossisse')


def test_load_fits_logs_numeric_extension_precedence(tmp_path, monkeypatch):
    primary = fits.PrimaryHDU(np.array([0]))
    science = fits.ImageHDU(np.array([1]), name='SCI')
    error = fits.ImageHDU(np.array([2]), name='ERR')
    filename = tmp_path / 'input.fits'
    fits.HDUList([primary, science, error]).writeto(filename)
    messages = []
    monkeypatch.setattr(
        io.misc, 'printc', lambda message, msg_type: messages.append(
            (message, msg_type)))

    data = io.load_fits(str(filename), ext=1, extname='ERR')

    np.testing.assert_array_equal(data, [1])
    assert messages == [
        (f'Loaded data from {filename} (ext=1)', 'info')]


@pytest.mark.parametrize('source', ['SOSSISSE', 'POGOS'])
def test_file_loading_with_shared_calibrations(tmp_path, source, monkeypatch):
    shared = tmp_path / 'shared-calibrations'
    shared.mkdir()
    background = shared / 'background.fits'
    background.touch()
    external_flat = tmp_path / 'external-flat.fits'
    external_flat.touch()
    external_input = tmp_path / 'external-input.fits'
    external_input.touch()
    params = make_params(tmp_path / 'root', source, **{
        'PATHS.CALIBPATH': str(shared),
        'GENERAL.BKGFILE': background.name,
        'GENERAL.FLATFILE': str(external_flat),
        'GENERAL.POS_FILE': None,
    })
    params = const_funcs.run_time_params(params, True)
    raw_file = Path(params['PATHS.RAWPATH']) / 'raw.fits'
    raw_file.touch()
    params['GENERAL.FILES'] = [raw_file.name, str(external_input)]
    checked_headers = []
    monkeypatch.setattr(
        const_funcs.input_validation, 'check_input_data_headers',
        lambda mode, files: checked_headers.append((mode, list(files))))
    for iteration in range(2):
        params = const_funcs.run_time_params(params, False)
        assert params['GENERAL.FILES'] == [str(raw_file), str(external_input)]
        assert params['GENERAL.BKGFILE'] == str(background)
        assert params['GENERAL.FLATFILE'] == str(external_flat)
        assert params['GENERAL.POS_FILE'] == str(shared / 'pos_file.fits')
        assert params['PATHS.CALIBPATH'] == str(shared)
    assert len(checked_headers) == 2
    assert checked_headers[0] == (
        'JWST.NIRISS.SOSS', [str(raw_file), str(external_input)])


def test_project_constant_schema():
    schema = constants.CDict.storage
    project = schema['INPUTS.PROJECT_NAME']
    assert project.not_none
    assert project.cmd_arg == 'project_name'
    assert 'PATHS.RUN_PATH' in schema
    assert 'INPUTS.SUBDIRECTORY' not in schema
    assert 'PATHS.SUBDIRECTORY_PATH' not in schema
    setup_inputs = const_funcs.INPUTARGS['sossisse.recipes.run_setup']
    assert 'INPUTS.PROJECT_NAME' in setup_inputs


def test_summary_uses_project_name(tmp_path):
    html = io.info_html(make_params(tmp_path), 'INPUTS.PROJECT_NAME')
    assert '<b>PROJECT_NAME</b>: visit-one' in html