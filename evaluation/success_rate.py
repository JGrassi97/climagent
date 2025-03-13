# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

# The Success Rate metric (SR) has been proposed in various papers.

from climagent.agent.climagent import ClimAgent
from climagent.tools.xarray_tools_indexing import SubsetDatasetTool

from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import xarray as xr

from tqdm import tqdm

load_dotenv("../credentials.env")

if __name__ == "__main__":
    
    dataset_path = "../example_datasets/copernicus/data.nc"
    dataset = xr.open_dataset(dataset_path, decode_timedelta=True)
    
    # Initialize agent
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        temperature=0,
    )

    query: str = (
        "I would like you to perform the following operations on the input dataset:\n"
        "1. Select the coordinates of Rome in the dataset.\n"
        "2. Select the optimistic scenario.\n"
    )
    messages = [HumanMessage(query)]

    correct = 0
    runtime = 0
    failed = 0

    n_runs = 20

    for j in tqdm(range(n_runs)):

        try:
            dataset_state_ref = DatasetState(dataset)
            json_state_ref = JsonState(dataset)
            subset_tool = SubsetDatasetTool(dataset_state=dataset_state_ref, json_state=json_state_ref)
            subset_tool.invoke({"coordinate_name": "lat", "values": ["41.9028"]})
            subset_tool.invoke({"coordinate_name": "lon", "values": ["12.4964"]})
            subset_tool.invoke({"coordinate_name": "scenario", "values": ["ssp1_2_6"]})

            agent = ClimAgent(dataset, llm)
            
            response, mod_dataset = agent.run(messages)

            if dataset_state_ref.dataset.equals(mod_dataset):
                correct += 1
            else:
                failed += 1
        except Exception as e:
            runtime += 1
    
    print("\n")
    print(f"Correct: {correct} : {correct/n_runs * 100}%")   
    print(f"Runtime error: {runtime} : {runtime/n_runs * 100}%")
    print(f"Failed: {failed} : {failed/n_runs * 100}%")
    print("\n")


    #print(response['messages'][-1].content)