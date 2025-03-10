# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

import xarray as xr
from langchain_community.tools.json.tool import JsonSpec



class JsonState:
    def __init__(self, dataset : xr.Dataset):
        self.json_spec_original = self._create_json_spec(dataset)
        self.json_spec = self._create_json_spec(dataset)
        self.history = []

    def update_json_spec(self, new_dataset, operation):
        """Update the json_spec."""
        self.json_spec = self._create_json_spec(new_dataset)
        self.history.append(operation)

    def get_spec(self):
        """Get updated version in json_spec."""
        return self.json_spec

    def get_history(self):
        """Get history of operations on json_spec."""
        return "\n".join(self.history)
    

    def _get_dataset_attrs(self, dataset : xr.Dataset) -> dict :
        """Returns dataset attributes."""
        return dataset.attrs

    def _get_dataset_coords(self, dataset : xr.Dataset) -> dict : 
        """Returns dataset coordinates."""
        all_coords = list(dataset.coords.keys())
        coord_info = {coord: dataset.coords[coord].to_dict() for coord in all_coords}
        return coord_info

    def _get_dataset_vars(self, dataset : xr.Dataset) -> dict :
        """Returns dataset variables."""
        all_vars = list(dataset.data_vars.keys())
        var_info = {var: dataset[var].attrs for var in all_vars}
        return var_info

    def _create_json_spec(self, dataset : xr.Dataset, max_value_lenght : int = 1000) -> JsonSpec :
        """Creates JsonSpec from dataset."""
        data = {
            'attrs': self._get_dataset_attrs(dataset), 
            'coords': self._get_dataset_coords(dataset), 
            'data_vars': self._get_dataset_vars(dataset)
            }

        json_spec = JsonSpec(
            dict_=data,
            max_value_length=max_value_lenght
        )

        return json_spec
