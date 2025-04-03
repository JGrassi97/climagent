# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain.tools import BaseTool
import xarray as xr
from typing import List, Type
from pydantic import BaseModel, Field
from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState
# JSON functions developed by langchain_community


# class LookDatasetInput(BaseModel):
#     coordinate_name: str = Field(description="The name of the coordinate to subset the dataset on.")
#     values: List[str] = Field(description="A list of values to subset the dataset on. Use one value for nearest selection or two values for a slice.")

class LookDatasetTool(BaseTool):
    name: str = "look_dataset"
    description: str = "Look at the numerical data inside the dataset. Use this tool when you have performed all the operations you need and want to see the final result."
    #args_schema: Type[LookDatasetInput] = LookDatasetInput
    dataset_state: DatasetState
    json_state: JsonState  

    def __init__(self, dataset_state: DatasetState, json_state: JsonState, **kwargs):
        kwargs["dataset_state"] = dataset_state 
        kwargs["json_state"] = json_state
        super().__init__(**kwargs)

    def _run(self) -> str:

        dat = self.dataset_state.dataset.copy()

        try:
            return f"Dataset content: {dat.to_dict()}"

        except Exception as e:
            return f"Cannot look at the dataset: {e}"

    async def _arun(self) -> str:
        raise NotImplementedError("look_dataset does not support async")