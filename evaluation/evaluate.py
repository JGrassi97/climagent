import os
import json
import xarray as xr
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from climagent.agent.climagent import ClimAgent


from utils import initialize_llm, load_queries, save_statistics_to_excel, deterministic_process_dataset
from metrics import success_rate, repetition_rate

import argparse

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Carica variabili d'ambiente
load_dotenv("../credentials.env")

# Carica variabili d'ambiente

def main():

    # Parse number of iterations
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=10, help="Number of iterations to run the evaluation")
    args = parser.parse_args()
    n_runs = args.n_runs

    dataset_path = "../example_datasets/copernicus/data.nc"
    dataset = xr.open_dataset(dataset_path, decode_timedelta=True)
    
    llm = initialize_llm()
    queries = load_queries("queries.json")
    
    query_results = {}
    
    for query_data in queries:


        query = query_data["query"]
        functions = query_data["functions"]  
        dataset_state_ref = deterministic_process_dataset(dataset, functions)

        messages = [HumanMessage(query)]

        run_results = {}

        for idx_run in tqdm(range(n_runs)):
            try:
                agent = ClimAgent(dataset, llm)
                
                response, mod_dataset = agent.run(messages)
                
                sr = success_rate(dataset_state_ref, mod_dataset)
                rr = repetition_rate(response)

            except Exception:
                sr = 'Run failed'
                rr = 'Run failed'

            run_results[idx_run] = {"success_rate": sr,
                                    "repetition_rate": rr}
        
        query_results[query] = run_results
    

    with open("query_results.json", "w") as f:
        # Save the results to a json file in indent format
        json.dump(query_results, f, indent=4)

if __name__ == "__main__":
    main()