# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from climagent.state.json_state import JsonState
from langchain.tools import BaseTool
from typing import Optional
#from langchain.tools import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from langchain_community.tools.json.tool import (
    JsonGetValueTool,   # subclass of BaseTool
    JsonListKeysTool,   # subclass of BaseTool
    JsonSpec,           # subclass of BaseModel
)

import xarray as xr

import logging
_logger = logging.getLogger(__name__)




class JsonGetValueTool_custom(BaseTool):
    """Tool for getting a value in a JSON spec."""

    name: str = "json_spec_get_value"
    description: str = """
    Can be used to see value in string format at a given path.
    Before calling this you should be SURE that the path to this exists.
    The input is a text representation of the path to the dict in Python syntax (e.g. data["key1"][0]["key2"]).
    """
    json_memory: JsonState 

    def __init__(self, json_memory: JsonState, **kwargs):
        kwargs["json_memory"] = json_memory
        super().__init__(**kwargs)

    def _run(
        self,
        tool_input: str,
        #run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.json_memory.get_spec().value(tool_input)

    async def _arun(
        self,
        tool_input: str,
        #run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)
    



class JsonListKeysTool_custom(BaseTool):
    """Tool for listing keys in a dynamically updated JSON spec."""

    name: str = "json_spec_list_keys"
    description: str = """
    Can be used to list all keys at a given path in the JSON structure.
    The input should be a text representation of the path in Python syntax 
    (e.g., data["key1"][0]["key2"]). Make sure the path exists before calling.
    """
    json_memory: JsonState  # Use JsonState instead of a static JsonSpec

    def __init__(self, json_memory: JsonState, **kwargs):
        kwargs["json_memory"] = json_memory
        super().__init__(**kwargs)

    def _run(
        self,
        tool_input: str,
        #run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Returns the list of keys at the specified JSON path."""
        try:
            json_spec = self.json_memory.get_spec()  # Get the latest json_spec
            return str(json_spec.keys(tool_input))  # Convert list to string
        except KeyError:
            return f"Error: Invalid path '{tool_input}', key not found."
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        tool_input: str,
        #run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)
