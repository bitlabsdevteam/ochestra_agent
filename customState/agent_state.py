from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# AgentState represents the state of an agent in a conversation
# It is implemented as a TypedDict for structured state management
class AgentState(TypedDict):
    # messages: A sequence of BaseMessage objects representing the conversation history
    # The operator.add annotation enables combining message sequences with the + operator
    # This allows for easy concatenation of message histories when merging agent states
    location: Annotated[Sequence[BaseMessage], operator.add]
