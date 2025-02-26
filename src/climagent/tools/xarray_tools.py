from langchain.tools import BaseTool
import xarray as xr
from typing import List, Type
from pydantic import BaseModel, Field

class SubsetDatasetInput(BaseModel):
    coordinate_name: str = Field(description="The name of the coordinate to subset the dataset on.")
    values: List[str] = Field(description="A list of values to subset the dataset on. Use one value for nearest selection or two values for a slice.")

class SubsetDatasetTool(BaseTool):
    name: str = "subset_dataset"
    description: str = "Subset an xarray dataset based on the provided coordinate name and values. Use one value for nearest selection or two values for a slice."
    args_schema: Type[SubsetDatasetInput] = SubsetDatasetInput
    dataset: xr.Dataset

    def __init__(self, dataset: xr.Dataset, **kwargs):
        kwargs["dataset"] = dataset #aggiungo dataset ai kwargs.
        super().__init__(**kwargs)  # Passa tutti i kwargs, incluso dataset
        #self.dataset = dataset #non serve più

    def _run(self, coordinate_name: str, values: List[str]) -> str:
        """Use the tool."""
        subset_dat = self.dataset.copy()

        try:
            # Converte i valori delle stringhe in float o int se possibile
            try:
                values = [float(v) if '.' in v else int(v) for v in values]
            except ValueError:
                return "Error: Values must be numeric strings."

            if len(values) == 1:
                subset_dat = subset_dat.sel({coordinate_name: values[0]}, method="nearest")
                self.dataset = subset_dat
                return f"Subset executed successfully for {coordinate_name}({values[0]})"

            elif len(values) == 2:
                subset_dat = subset_dat.sel({coordinate_name: slice(values[0], values[1])})
                self.dataset = subset_dat
                return f"Subset executed successfully for {coordinate_name}({values[0]}:{values[1]})"

            else:
                return "Error: Invalid number of values provided for subsetting. Provide one or two numeric values."

        except Exception as e:
            return f"Error in dataset slicing: {e}"

    async def _arun(self, coordinate_name: str, values: List[str]) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("subset_dataset does not support async")