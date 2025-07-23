from pydantic import BaseModel, Field
from typing import Dict, List, Annotated, Sequence, Any, Optional
from langchain_core.messages import BaseMessage

class ToolParameters(BaseModel):
    """Model for tool parameters in a function call"""
    __root__: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameter name to parameter value")

class FunctionCall(BaseModel):
    """Model for a single function call"""
    name: str = Field(description="The name of the function to call")
    arguments: Dict[str, Any] = Field(description="The arguments to pass to the function")

class WeatherOutputParser(BaseModel):
    thinking: str = Field(description="String describing what the agent is doing and why")
    response: str = Field(description="String response to the user")
    function_calls: List[FunctionCall] = Field(default_factory=list, description="List of function calls to make")