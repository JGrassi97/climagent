from climagent.agent.climagent import ClimAgent
import numpy as np

from utils import deterministic_process_dataset



def success_rate(dataset_ref, dataset_agent):

    correct, failed = 0, 0
    
    if dataset_ref.dataset.equals(dataset_agent):
        return 'correct'
    else:
        return 'failed'



def repetition_rate(response):
    # The Repetition Rate metric (RR) has been proposed in AgentQuest (Giocchini et al., 2024)

    all_tools = []
    results = {}
    for id, message in enumerate(response['messages']):
        
        if message.name is not None:
            all_tools.append(message.name)

        all_tools_unique = np.unique(all_tools)
        for tool in all_tools_unique:

            results[tool] = all_tools.count(tool)

    return results


def optimal_flow(functions):

    all_tools = []
    results = {}
    for id, func in enumerate(functions):
        
        if func['name'] is not None:
            all_tools.append(func['name'])

        all_tools_unique = np.unique(all_tools)
        for tool in all_tools_unique:

            results[f'{tool}_optimal'] = all_tools.count(tool)

    return results


