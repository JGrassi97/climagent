# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain_community.tools.json.tool import (
    JsonGetValueTool,   # subclass of BaseTool
    JsonListKeysTool,   # subclass of BaseTool
    JsonSpec,           # subclass of BaseModel
)

import xarray as xr

import logging
_logger = logging.getLogger(__name__)



def _get_dataset_attrs(dataset : xr.Dataset) -> dict :
    """Returns dataset attributes."""
    _logger.info("Getting dataset attributes")
    return dataset.attrs

def _get_dataset_coords(dataset : xr.Dataset) -> dict : 
    """Returns dataset coordinates."""
    _logger.info("Getting dataset coordinates")
    all_coords = list(dataset.coords.keys())
    coord_info = {coord: dataset.coords[coord].to_dict() for coord in all_coords}
    return coord_info

def _get_dataset_vars(dataset : xr.Dataset) -> dict :
    """Returns dataset variables."""
    _logger.info("Getting dataset variables")
    all_vars = list(dataset.data_vars.keys())
    var_info = {var: dataset[var].attrs for var in all_vars}
    return var_info

def _create_json_spec(dataset : xr.Dataset, max_value_lenght : int = 1000) -> JsonSpec :
    """Creates JsonSpec from dataset."""
    data = {
        'attrs': _get_dataset_attrs(dataset), 
        'coords': _get_dataset_coords(dataset), 
        'data_vars': _get_dataset_vars(dataset)
        }

    json_spec = JsonSpec(
        dict_=data,
        max_value_length=max_value_lenght
    )

    return json_spec
