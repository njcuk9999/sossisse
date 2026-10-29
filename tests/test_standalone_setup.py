import os
import subprocess
import sys
import textwrap

import pytest


STANDALONE_WORKFLOW = textwrap.dedent("""
    import importlib.abc
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    class BlockPogos(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'pogos' or fullname.startswith('pogos.'):
                raise AssertionError(f'Unexpected POGOS import: {fullname}')
            return None

    sys.meta_path.insert(0, BlockPogos())
    entrypoint, workspace = sys.argv[1:]
    sys.argv = ['standalone-sossisse']

    import yaml
    import sossisse
    from aperocore.constants.param_functions import ParamDict
    from sossisse.core import const_funcs, constants, io
    from sossisse.recipes import run_setup

    workspace = Path(workspace)
    root = workspace / 'root'
    project = 'standalone-visit'
    run_path = root / 'projects' / project / 'sossisse'
    child_paths = {
        'YAMLPATH': 'yamls',
        'RAWPATH': 'inputs',
        'CALIBPATH': 'calibration',
        'TEMP_PATH': 'temporary',
        'PLOT_PATH': 'plots',
        'OTHER_PATH': 'other',
        'OUT_PATH': 'outputs',
    }
    prompts = []
    defaults = const_funcs.load_functions.load_parameters([constants.CDict])
    assert isinstance(defaults, ParamDict)
    assert defaults['INPUTS.PROJECT_NAME'] is None
    assert defaults.instances['INPUTS.PROJECT_NAME'].not_none is True

    def answer_prompt(question, **options):
        prompts.append(question)
        assert 'PROJECT_NAME' in question, question
        assert 'SUBDIRECTORY' not in question, question
        assert options['required'] is True
        return project

    def scientific_instrument(params):
        assert isinstance(params, ParamDict)
        assert params['CONFIG_LIST'] == [constants.CDict]
        assert params['WLC.LMODEL.FIT_QUAD_TERM'] == (
            defaults['WLC.LMODEL.FIT_QUAD_TERM'])
        return SimpleNamespace(params=params)

    setup_options = dict(
        sossiopath=str(root), objname='standalone-target',
        instmode='JWST.NIRISS.SOSS', yaml_name='standalone.yaml')
    with patch.object(const_funcs, 'load_instrument',
                      side_effect=scientific_instrument) as instrument_loader:
        with patch.object(const_funcs.load_functions, 'starting_point',
                          side_effect=lambda params, *values: params):
            with patch.object(const_funcs.load_functions, 'download_data') as download:
                with patch.object(const_funcs.load_functions.drs_text,
                                  'user_input', side_effect=answer_prompt):
                    if entrypoint == 'recipe':
                        instrument = run_setup.main(**setup_options)
                    else:
                        instrument = sossisse.get_parameters(
                            __NAME__=run_setup.__NAME__, setup_mode=True,
                            **setup_options)
                download.assert_called_once()

        assert instrument is not None
        params = instrument.params
        assert len(prompts) == 1, prompts
        assert params['__SOURCE__'] == 'SOSSISSE'
        assert params['INPUTS.PROJECT_NAME'] == project
        assert params['INPUTS.OBJECTNAME'] == 'standalone-target'
        assert Path(params['PATHS.RUN_PATH']) == run_path
        for key, directory in child_paths.items():
            assert Path(params[f'PATHS.{key}']) == run_path / directory
            assert (run_path / directory).is_dir()

        yaml_path = Path(params['INPUTS.PARAM_FILE'])
        assert yaml_path == run_path / 'yamls' / 'standalone.yaml'
        assert yaml_path.is_file()
        yaml_bytes = yaml_path.read_bytes()
        setup_hash = io.get_hash(yaml_bytes.decode('utf-8'))
        assert len(setup_hash) == 64
        saved = yaml.safe_load(yaml_bytes)
        assert saved['INPUTS']['PROJECT_NAME'] == project
        assert 'SUBDIRECTORY' not in saved['INPUTS']
        assert 'POGOS' not in saved
        assert not list((Path.home() / '.sossisse').glob('*.yaml'))

        with patch.object(const_funcs.load_functions.drs_text, 'user_input',
                          side_effect=AssertionError('Reload must not prompt')):
            reloaded = sossisse.get_parameters(
                __NAME__='sossisse.recipes.run_sossisse',
                param_file=str(yaml_path), only_create=True)

        assert instrument_loader.call_count == 2
        assert reloaded.params['__SOURCE__'] == 'SOSSISSE'
        assert reloaded.params['INPUTS.PROJECT_NAME'] == project
        assert reloaded.params['INPUTS.OBJECTNAME'] == 'standalone-target'
        assert reloaded.params['INPUTS.INSTRUMENTMODE'] == 'JWST.NIRISS.SOSS'
        assert reloaded.params['INPUTS.PARAM_FILE'] == str(yaml_path)
        assert reloaded.params['PATHS.RUN_PATH'] == str(run_path)
        for key in child_paths:
            assert reloaded.params[f'PATHS.{key}'] == params[f'PATHS.{key}']
        assert yaml_path.read_bytes() == yaml_bytes
        assert io.get_hash(yaml_path.read_text()) == setup_hash
        assert not any(name == 'pogos' or name.startswith('pogos.')
                       for name in sys.modules)
""")


@pytest.mark.parametrize('entrypoint', ['parameters', 'recipe'])
def test_standalone_setup_and_yaml_reload(tmp_path, entrypoint):
    home = tmp_path / 'home'
    home.mkdir()
    environment = dict(os.environ, HOME=str(home), MPLBACKEND='Agg')
    result = subprocess.run(
        [sys.executable, '-c', STANDALONE_WORKFLOW, entrypoint, str(tmp_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr