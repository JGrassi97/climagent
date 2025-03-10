# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

import os
import xarray as xr
from langgraph.graph import StateGraph, END

from typing_extensions import TypedDict
from typing import Annotated
import operator
from langchain_core.messages import AnyMessage, ToolMessage


# Import tools and memory modules
from climagent.tools.json_tools import JsonGetValueTool_custom, JsonListKeysTool_custom
from climagent.tools.xarray_tools_indexing import SubsetDatasetTool
from climagent.tools.xarray_tools_grouping import ResampleTimeTool
from climagent.tools.xarray_tools_aggregating import AggregateDatasetTool
from climagent.agent.prefix import PREFIX

from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class ClimAgent:
    def __init__(self, dataset_path: str, llm, llm_temperature: float = 0.0):
        
    
        # Load dataset
        self.dataset = xr.open_mfdataset(dataset_path)
        self.dataset_state = DatasetState(self.dataset)
        self.json_memory = JsonState(self.dataset)
        
        # Initialize tools
        self.tools = [
            JsonGetValueTool_custom(json_memory=self.json_memory),
            JsonListKeysTool_custom(json_memory=self.json_memory),
            SubsetDatasetTool(dataset_state=self.dataset_state, json_memory=self.json_memory),
            ResampleTimeTool(dataset_state=self.dataset_state, json_memory=self.json_memory),
            AggregateDatasetTool(dataset_state=self.dataset_state, json_memory=self.json_memory)
        ]
        self.tools_names = {t.name: t for t in self.tools}
        
        # Bind tools to LLM
        self.llm = llm.bind_tools(self.tools)
        
        # Create agent
        self.graph = self._build_agent()
    
    def _build_agent(self):
        def run_llm(state: State):
            messages = state['messages']
            message = self.llm.invoke(messages)
            return {'messages': [message]}

        def execute_tools(state: State):
            tool_calls = state['messages'][-1].tool_calls
            results = []
            for t in tool_calls:
                if t['name'] not in self.tools_names:
                    result = "Error: There's no such tool, please try again"
                else:
                    result = self.tools_names[t['name']].invoke(t['args'])

                results.append(
                    ToolMessage(
                        tool_call_id=t['id'],
                        name=t['name'],
                        content=str(result)
                    )
                )
            return {'messages': results}

        def tool_exists(state: State):
            result = state['messages'][-1]
            return len(result.tool_calls) > 0
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("llm", run_llm)
        graph_builder.add_node("tools", execute_tools)
        graph_builder.add_conditional_edges("llm", tool_exists, {True: "tools", False: END})
        graph_builder.add_edge("tools", "llm")
        graph_builder.set_entry_point("llm")
        
        return graph_builder.compile()
    
    def run(self, messages: list[AnyMessage]):
        return self.graph.invoke({'messages': [PREFIX] + messages},  config={"recursion_limit": 50}), self.dataset_state.dataset
