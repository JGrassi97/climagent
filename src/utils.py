from climagent.tools.xarray_tools_indexing import SubsetDatasetTool
from climagent.tools.xarray_tools_grouping import ResampleTimeTool
from climagent.tools.xarray_tools_aggregating import AggregateDatasetTool
from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState


from langchain_openai import AzureChatOpenAI
import os
import json
import pandas as pd

def initialize_llm():
    return AzureChatOpenAI(
        deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        temperature=0,
    )



## UTILS FOR EVALUATION

def deterministic_process_dataset(dataset, functions):
    dataset_state_ref = DatasetState(dataset)
    json_state_ref = JsonState(dataset)
    
    tools = {
        "subset": SubsetDatasetTool(dataset_state=dataset_state_ref, json_state=json_state_ref),
        "resampletime_dataset": ResampleTimeTool(dataset_state=dataset_state_ref, json_state=json_state_ref),
        "aggregate_dataset": AggregateDatasetTool(dataset_state=dataset_state_ref, json_state=json_state_ref)
    }
    
    for function in functions:
        tool_name = function["name"]
        arguments = json.loads(function["arguments"])

        if tool_name not in tools:
            raise ValueError(f"Tool {tool_name} not recognized")

        tool = tools[tool_name]
        tool.invoke(arguments)
    
    return dataset_state_ref


def load_queries(query_file):
    with open(query_file, "r") as f:
        return json.load(f)

def save_statistics_to_excel(stats, filename="evaluation_results.xlsx"):
    df = pd.DataFrame(stats)
    df.to_excel(filename, index=False)