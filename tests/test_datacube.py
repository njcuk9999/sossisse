"""Checks for bounded on-disk integration containers."""

import numpy as np
import pytest

from sossisse.general.datacube import DataCube
from test_datacube_fits import normalize
from test_datacube_fits import write_raw


@pytest.mark.parametrize('limit', [2, 5, 8])
def test_datacube_storage(tmp_path, limit):
    filename, arrays = write_raw(tmp_path)
    datacube = normalize([filename], max_integrations=limit)
    assert not hasattr(datacube, 'in_memory')
    assert not hasattr(datacube, 'cube')
    assert not hasattr(datacube, 'err')
    assert not hasattr(datacube, 'dq')
    assert not hasattr(DataCube, 'from_files')
    assert not hasattr(DataCube, 'from_cache')
    assert datacube.filenames == (filename,)
    blocks = list(datacube.iter_integrations())
    for name, expected in zip(('FLUX', 'ERR', 'DQ'), arrays):
        actual = np.concatenate([block[name] for _, block in blocks])
        np.testing.assert_equal(actual, expected)
    assert max(part.stop - part.start for part, _ in blocks) <= limit
    if limit < len(datacube):
        with pytest.raises(ValueError, match='exceed'):
            datacube.read_integrations(0, len(datacube))


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError, match='positive integer'):
        DataCube(max_integrations=limit)