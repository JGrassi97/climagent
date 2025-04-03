# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain_core.messages import SystemMessage

def make_suffix(json_memory):
    suffix =  SystemMessage(
    content=(
        "I'll provide you some standard information about the dataset, to avoid a large number of calls to the json tools.\n"
        "I'm providing just the keys for the three main sections of the JSON: 'attrs', 'coords', and 'vars'.\n"
        f"attrs: {json_memory.get_spec().dict_['attrs'].keys()}\n"
        f"coords: {json_memory.get_spec().dict_['coords'].keys()}\n"
        f"data_vars: {json_memory.get_spec().dict_['data_vars'].keys()}\n"
        "Please, call the json_function only if you think you need more information."
        "Plase, keep in mind the operations you have performed, in order to avoid redoing them."
        )
    )
    return suffix