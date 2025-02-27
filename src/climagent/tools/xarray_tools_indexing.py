# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain.tools import BaseTool
import xarray as xr
from typing import List, Type
from pydantic import BaseModel, Field
from climagent.state.dataset_memory import DatasetMemory



class SubsetDatasetInput(BaseModel):
    coordinate_name: str = Field(description="The name of the coordinate to subset the dataset on.")
    values: List[str] = Field(description="A list of values to subset the dataset on. Use one value for nearest selection or two values for a slice.")

class SubsetDatasetTool(BaseTool):
    name: str = "subset_dataset"
    description: str = "Subset an xarray dataset based on the provided coordinate name and values. Use one value for nearest selection or two values for a slice. Be sure to provide values consistent with the coordinate type (numeric, datetime, etc.)."
    args_schema: Type[SubsetDatasetInput] = SubsetDatasetInput
    dataset_memory: DatasetMemory

    def __init__(self, dataset_memory: DatasetMemory, **kwargs):
        kwargs["dataset_memory"] = dataset_memory 
        super().__init__(**kwargs)

    def _run(self, coordinate_name: str, values: List[str]) -> str:

        subset_dat = self.dataset_memory.dataset.copy()

        try:
            try:
                if coordinate_name not in ['time', 'datetime', 'timedelta', 'valid_time', 'scenario', 'model']:
                    values = [float(v) if '.' in v else int(v) for v in values]
            except ValueError:
                return "Error: Values must be numeric strings."

            if len(values) == 1:

                try:
                    subset_dat = subset_dat.sel({coordinate_name: values[0]})
                
                except:
                    subset_dat = subset_dat.sel({coordinate_name: values[0]}, method="nearest")

                operation = f"Subset on {coordinate_name}({values[0]})"

            elif len(values) == 2:
                subset_dat = subset_dat.sel({coordinate_name: slice(values[0], values[1])})
                operation = f"Subset on {coordinate_name}({values[0]}:{values[1]})"

            else:
                return "Error: Invalid number of values provided for subsetting. Provide one or two numeric values."

            self.dataset_memory.update_dataset(subset_dat, operation)
            return f"Subset executed successfully: {operation}"

        except Exception as e:
            return f"Error in dataset slicing: {e}"

    async def _arun(self, coordinate_name: str, values: List[str]) -> str:
        raise NotImplementedError("subset_dataset does not support async")