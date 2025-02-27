# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch

from climagent.tools.json_tools import _get_dataset_attrs, _get_dataset_coords, _get_dataset_vars, _create_json_spec

@pytest.fixture
def sample_dataset():
    x_coords = np.array([1, 2, 3])  # Definisci i valori per la coordinata x
    return xr.Dataset({
        'temperature': ('x', [20, 22, 25]),
        'pressure': ('x', [1000, 1010, 1020])
    }, coords={'x': x_coords}, attrs={'units': 'metric'})

def test_get_dataset_attrs(sample_dataset):
    attrs = _get_dataset_attrs(sample_dataset)
    assert attrs == {'units': 'metric'}

def test_get_dataset_coords(sample_dataset):
    coords = _get_dataset_coords(sample_dataset)
    assert 'x' in coords

def test_get_dataset_vars(sample_dataset):
    vars = _get_dataset_vars(sample_dataset)
    assert 'temperature' in vars and 'pressure' in vars

def test_create_json_spec(sample_dataset):
    json_spec = _create_json_spec(sample_dataset)
    assert 'attrs' in json_spec.dict_
    assert 'coords' in json_spec.dict_
    assert 'data_vars' in json_spec.dict_