# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain.tools import BaseTool
import xarray as xr
from typing import List, Type
from pydantic import BaseModel, Field
from climagent.state.dataset_memory import DatasetMemory



class ResampleTimeDatasetInput(BaseModel):
    coordinate_name: str = Field(description="The name of the coordinate to group the dataset on.")
    frequency: str = Field(description="The frequency of the resampling operation. Use '*YE', '*M', '*W', '*D', '*H', '*T', '*S' for yearly, monthly, weekly, daily, hourly, minutely, and secondly grouping, respectively. Substitute '*' with an integer for a custom frequency.")

class ResampleTimeTool(BaseTool):
    name: str = "resampletime_dataset"
    description: str = "Resample a dataset on a time coordinate"
    args_schema: Type[ResampleTimeDatasetInput] = ResampleTimeDatasetInput
    dataset_memory: DatasetMemory

    def __init__(self, dataset_memory: DatasetMemory, **kwargs):
        kwargs["dataset_memory"] = dataset_memory 
        super().__init__(**kwargs)

    def _run(self, coordinate_name: str, frequency: str) -> str:

        subset_dat = self.dataset_memory.dataset.copy()

        try:
            
            subset_dat = subset_dat.resample({coordinate_name : frequency}).mean()
            operation = f"Group on {coordinate_name}({frequency})"

            self.dataset_memory.update_dataset(subset_dat, operation)
            return f"Subset executed successfully: {operation}"

        except Exception as e:
            return f"Error in dataset resampling: {e}"

    async def _arun(self, coordinate_name: str, values: List[str]) -> str:
        raise NotImplementedError("resampletime_dataset does not support async")