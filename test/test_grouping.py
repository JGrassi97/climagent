# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

import pytest
import xarray as xr
import pandas as pd
from unittest.mock import MagicMock
from climagent.tools.xarray_tools_grouping import ResampleTimeTool
from climagent.state.dataset_state import DatasetMemory
from climagent.state.json_state import JsonMemory

@pytest.fixture
def sample_dataset():
    time = pd.date_range('2020-01-01', periods=10, freq='D')
    return xr.Dataset({
        'temperature': ('time', [20, 22, 25, 23, 21, 19, 18, 17, 16, 15]),
        'pressure': ('time', [1000, 1010, 1020, 1015, 1005, 995, 990, 985, 980, 975])
    }, coords={'time': time})

@pytest.fixture
def dataset_memory(sample_dataset):
    memory = MagicMock(spec=DatasetMemory)
    memory.dataset = sample_dataset
    return memory

@pytest.fixture
def json_memory():
    return MagicMock(spec=JsonMemory)

@pytest.fixture
def resample_tool(dataset_memory, json_memory):
    return ResampleTimeTool(dataset_memory=dataset_memory, json_memory=json_memory)

def test_run_success(resample_tool):
    result = resample_tool._run(coordinate_name='time', frequency='2D')
    assert "Subset executed successfully" in result

def test_run_unsupported_frequency(resample_tool):
    result = resample_tool._run(coordinate_name='time', frequency='unsupported')
    assert "Error in dataset resampling" in result

def test_run_exception_handling(resample_tool, dataset_memory):
    dataset_memory.dataset = None  # Force an exception
    result = resample_tool._run(coordinate_name='time', frequency='2D')
    assert "Error in dataset resampling" in result