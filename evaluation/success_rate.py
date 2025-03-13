import os
import json
import xarray as xr
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from climagent.agent.climagent import ClimAgent
from climagent.tools.xarray_tools_indexing import SubsetDatasetTool
from climagent.state.dataset_state import DatasetState
from climagent.state.json_state import JsonState

from utils import initialize_llm

# Carica variabili d'ambiente
load_dotenv("../credentials.env")

def process_dataset(dataset, coordinates):
    dataset_state_ref = DatasetState(dataset)
    json_state_ref = JsonState(dataset)
    subset_tool = SubsetDatasetTool(dataset_state=dataset_state_ref, json_state=json_state_ref)
    
    for c_name, c_value in coordinates.items():
        subset_tool.invoke({"coordinate_name": c_name, "values": [c_value]})
    
    return dataset_state_ref

def evaluate_agent(dataset, llm, messages, coordinates, n_runs=20):
    correct, runtime, failed = 0, 0, 0
    
    for _ in tqdm(range(n_runs)):
        try:
            dataset_state_ref = process_dataset(dataset, coordinates)
            agent = ClimAgent(dataset, llm)
            response, mod_dataset = agent.run(messages)
            
            if dataset_state_ref.dataset.equals(mod_dataset):
                correct += 1
            else:
                failed += 1
        except Exception:
            runtime += 1
    
    return correct, runtime, failed, n_runs

def load_queries(query_file):
    with open(query_file, "r") as f:
        return json.load(f)

def save_statistics_to_excel(stats, filename="evaluation_results.xlsx"):
    df = pd.DataFrame(stats)
    df.to_excel(filename, index=False)

def main():
    dataset_path = "../example_datasets/copernicus/data.nc"
    dataset = xr.open_dataset(dataset_path, decode_timedelta=True)
    
    llm = initialize_llm()
    queries = load_queries("queries.json")
    
    total_correct, total_runtime, total_failed, total_runs = 0, 0, 0, 0
    stats = []
    
    for query_data in queries:
        query = query_data["query"]
        coordinates = query_data["coordinates"]
        messages = [HumanMessage(query)]
        
        correct, runtime, failed, n_runs = evaluate_agent(dataset, llm, messages, coordinates, n_runs=5)
        
        total_correct += correct
        total_runtime += runtime
        total_failed += failed
        total_runs += n_runs
        
        stats.append({
            "Query": query,
            "Correct": correct,
            "Runtime Error": runtime,
            "Failed": failed,
            "Success Rate (%)": (correct / n_runs) * 100
        })
        
        print(f"\nQuery: {query}")
        print(f"Correct: {correct} : {correct/n_runs * 100}%")   
        print(f"Runtime error: {runtime} : {runtime/n_runs * 100}%")
        print(f"Failed: {failed} : {failed/n_runs * 100}%\n")
    
    overall_stats = {
        "Query": "Overall Statistics",
        "Correct": total_correct,
        "Runtime Error": total_runtime,
        "Failed": total_failed,
        "Success Rate (%)": (total_correct / total_runs) * 100
    }
    stats.append(overall_stats)
    
    print("\n==== Overall Statistics ====")
    print(f"Total Correct: {total_correct} : {total_correct/total_runs * 100}%")   
    print(f"Total Runtime error: {total_runtime} : {total_runtime/total_runs * 100}%")
    print(f"Total Failed: {total_failed} : {total_failed/total_runs * 100}%\n")
    
    save_statistics_to_excel(stats)
    print(f"Statistics saved to evaluation_results.xlsx")

if __name__ == "__main__":
    main()
