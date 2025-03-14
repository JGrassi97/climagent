# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

import os
import xarray as xr
from langgraph.graph import StateGraph, END


from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage


# Import tools and memory modules
from climagent.tools.json_tools import JsonGetValueTool_custom, JsonListKeysTool_custom
from climagent.tools.xarray_tools_indexing import SubsetDatasetTool
from climagent.tools.xarray_tools_grouping import ResampleTimeTool
from climagent.tools.xarray_tools_aggregating import AggregateDatasetTool
from climagent.agent.prefix import PREFIX
from climagent.agent.suffix import make_suffix

from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState
from climagent.state.agent_state import State



class ClimAgent:
    def __init__(self, dataset: xr.Dataset, llm, llm_temperature: float = 0.0):
        
    
        # Load dataset
        self.dataset = dataset
        self.dataset_state = DatasetState(self.dataset)
        self.json_state = JsonState(self.dataset)
        
        # Initialize tools
        self.tools = [
            JsonGetValueTool_custom(json_state=self.json_state),
            JsonListKeysTool_custom(json_state=self.json_state),
            SubsetDatasetTool(dataset_state=self.dataset_state, json_state=self.json_state),
            ResampleTimeTool(dataset_state=self.dataset_state, json_state=self.json_state),
            AggregateDatasetTool(dataset_state=self.dataset_state, json_state=self.json_state)
        ]
        self.tools_names = {t.name: t for t in self.tools}
        
        # Bind tools to LLM
        self.llm = llm.bind_tools(self.tools)
        self.llm_planner = llm
        
        # Create agent
        self.graph = self._build_agent()


    def _build_agent(self):
        
        def planner(state: State):
            messages = state['messages']


            plan_prompt = (
                "Please follow the steps below to create the analysis plan:\n"
                "1. Identify the key objectives of the analysis.\n"
                "2. Break the analysis into **clear sequential steps**.\n"
                "3. Specify which tools (e.g., `SubsetDatasetTool`, `ResampleTimeTool`, `AggregateDatasetTool`) should be used in each step.\n"
                "4. Define the **expected output** of each step.\n"
                "5. Ensure the plan is **logical, structured, and efficient**.\n"
                "\n"
                "Example Format:\n"
                "- **Step 1**: Select the geographic region → Use `SubsetDatasetTool` to filter dataset by latitude/longitude.\n"
                "- **Step 2**: Resample time series data → Use `ResampleTimeTool` to aggregate data into larger time intervals.\n"
                "- **Step 3**: Compute statistical metrics → Use `AggregateDatasetTool` to compute mean, min, max, etc.\n"
                "- **Final Output**: A cleaned and structured dataset ready for visualization or further analysis.\n"
                "\n"
                "Now, based on the user's request, generate a custom plan.\n"
                "DO NOT CALL ANY TOOLS. Just create a structured plan."
            )
            
            # Aggiungi il plan_prompt come SystemMessage
            plan_message = SystemMessage(content=plan_prompt)

            messages = [plan_message]
            
            # Chiediamo all'LLM di generare il piano di analisi
            plan_message_response = self.llm_planner.invoke(messages)

            # Assicuriamoci che il messaggio sia un oggetto valido di tipo AnyMessage
            if isinstance(plan_message_response, str):  
                plan_message_response = AnyMessage(content=plan_message_response)  

            return {'messages': messages + [plan_message_response] + [PREFIX] + [make_suffix(self.json_state)]}
        
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
        graph_builder.add_node("planner", planner)
        graph_builder.add_node("llm", run_llm)
        graph_builder.add_node("tools", execute_tools)

        # Collegamenti: planner viene eseguito una volta, poi passa il controllo a llm
        graph_builder.set_entry_point("planner")
        graph_builder.add_edge("planner", "llm")

        # llm chiama tool se necessario, altrimenti finisce
        graph_builder.add_conditional_edges("llm", tool_exists, {True: "tools", False: END})
        graph_builder.add_edge("tools", "llm")

        return graph_builder.compile()
    
    
    def run(self, messages: list[AnyMessage]):
        return self.graph.invoke({'messages': messages},  config={"recursion_limit": 50}), self.dataset_state.dataset
