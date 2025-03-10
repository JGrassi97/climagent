# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain.tools import BaseTool
import xarray as xr
from typing import List, Type, Literal
from pydantic import BaseModel, Field
from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState
import numpy as np


# https://docs.xarray.dev/en/stable/api.html#aggregation

class AggregateDatasetInput(BaseModel):
    func: Literal["mean", "max", "min", "sum", "std", "var"] = Field(escription="The aggregation function to apply to the dataset.")
    dims: List[str] = Field(description="A list of coordinate names along which to aggregate the dataset.")


class AggregateDatasetTool(BaseTool):
    name: str = "aggregate_dataset"
    description: str = "Aggregate the dataset on one or more coordinates."
    args_schema: Type[AggregateDatasetInput] = AggregateDatasetInput
    dataset_state: DatasetState
    json_memory: JsonState  

    XARRAY_FUNCTIONS : dict = {
        "mean": "mean",
        "max": "max",
        "min": "min",
        "median" : "median",
        "prod" : "prod",
        "sum": "sum",
        "std": "std",
        "var": "var",
        "cumsum": "cumsum",
        "cumprod": "cumprod",
    }

    def __init__(self, dataset_state: DatasetState, json_memory: JsonState, **kwargs):
        kwargs["dataset_state"] = dataset_state 
        kwargs["json_memory"] = json_memory
        super().__init__(**kwargs)

    def _run(self, func: str, dims: List[str]) -> str:

        reduced_dat = self.dataset_state.dataset.copy()

        if func not in self.XARRAY_FUNCTIONS:
            return f"Error: Unsupported function '{func}'. Choose from {list(self.XARRAY_FUNCTIONS.keys())}"

        try:
            reduced_dat = getattr(reduced_dat, self.XARRAY_FUNCTIONS[func])(dim=dims)
            operation = f"Aggregated on {dims} using {func}"

            self.dataset_state.update_dataset(reduced_dat, operation)
            self.json_memory.update_json_spec(reduced_dat, operation)

            return f"Aggregation executed successfully: {operation}"

        except Exception as e:
            return f"Error in dataset aggregation: {e}"

    async def _arun(self, func: str, dims: List[str]) -> str:
        raise NotImplementedError("aggregate_dataset does not support async")