# Author: jacopo.grassi@polito.it
# Institute: Politecnico di Torino

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
