"""Simplified Orchestra Agent module for the Sales Maker application.

A minimal implementation that maintains core functionality while being as simple as possible.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent

# Import LLM factory
from LLMs.llm_factory import LLMFactory

# Import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.weather_tool import WeatherTool
from tools.time_tool import TimeTool
from tools.city_facts_tool import CityFactsTool

# Import output parser
from outputParser.trip_output_parser import FunctionCall

class OrchestraAgent:
    """A simplified agent that processes queries using LLMs and tools.
    
    Supports basic conversation and travel planning functionality.
    """
    
    def __init__(self, llm_prefix: str = "deepseek", system_prompt: Optional[str] = None, temperature: float = 0.7, use_tools: bool = True):
        """Initialize the OrchestraAgent.
        
        Args:
            llm_prefix: LLM provider to use ('openai', 'gemini', 'deepseek').
            system_prompt: Optional system prompt.
            temperature: Temperature for the LLM.
            use_tools: Whether to use tools.
        """
        self.llm = LLMFactory.get_llm(llm_prefix, temperature=temperature)
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize tools
        self.tools: List[BaseTool] = []
        if use_tools:
            self.tools = [
                WeatherTool(),
                TimeTool(),
                CityFactsTool()
            ]
        
        # Set default system prompt
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer questions concisely and accurately."
        
        # Add system message to conversation history
        self.conversation_history.append({"role": "system", "content": system_prompt})
        
        # Create travel agent if tools are enabled
        if use_tools:
            self._initialize_travel_agent()
    
    def _initialize_travel_agent(self):
        """Initialize the travel agent with tools."""
        # Create a proper ReAct prompt with required variables
        travel_prompt = """You are a travel assistant. Help plan trips and provide information about destinations.
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}"""
        
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(travel_prompt)
        
        # Create the travel agent
        self.travel_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        self.travel_agent_executor = AgentExecutor(
            agent=self.travel_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_query(self, query: str, use_travel_agent: bool = False) -> Union[str, Dict[str, Any]]:
        """Process a user query and return a response.
        
        Args:
            query: The user's query.
            use_travel_agent: Whether to use travel agent mode.
            
        Returns:
            String response or dictionary with structured data.
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": query})
        
        if use_travel_agent and hasattr(self, 'travel_agent_executor'):
            # Use travel agent
            response = self.travel_agent_executor.invoke({"input": query})
            raw_response = response.get("output", "I couldn't process that request.")
            
            try:
                # Parse the response
                parsed_response = self._parse_travel_agent_response(raw_response)
                self.conversation_history.append({"role": "assistant", "content": parsed_response["response"]})
                return parsed_response
            except Exception as e:
                # Return raw response on error
                print(f"Error parsing response: {str(e)}")
                self.conversation_history.append({"role": "assistant", "content": raw_response})
                return raw_response
        else:
            # Use standard conversation
            messages = self._convert_history_to_messages()
            response = self.llm.invoke(messages)
            response_content = response.content
            self.conversation_history.append({"role": "assistant", "content": response_content})
            return response_content
    
    def _convert_history_to_messages(self):
        """Convert history to LangChain messages."""
        messages = []
        for message in self.conversation_history:
            if message["role"] == "system":
                messages.append(SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
        return messages
    
    def clear_history(self):
        """Clear history except system prompt."""
        system_prompt = next((msg for msg in self.conversation_history if msg["role"] == "system"), None)
        self.conversation_history = [system_prompt] if system_prompt else []
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
        
    def get_available_tools(self) -> List[BaseTool]:
        """Get available tools."""
        return self.tools.copy()
        
    def _parse_travel_agent_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse travel agent response into structured format."""
        import json
        
        # Default values
        thinking = ""
        response = raw_response
        function_calls = []
        
        # Extract thinking (last thought before final answer)
        if "Thought:" in raw_response:
            thought_parts = raw_response.split("Thought:")
            if len(thought_parts) > 1:
                thinking_part = thought_parts[-1].strip()
                if "Action:" in thinking_part:
                    thinking = thinking_part.split("Action:")[0].strip()
                elif "Final Answer:" in thinking_part:
                    thinking = thinking_part.split("Final Answer:")[0].strip()
                else:
                    thinking = thinking_part
        
        # Extract response (Final Answer)
        if "Final Answer:" in raw_response:
            response = raw_response.split("Final Answer:")[-1].strip()
        
        # Extract function calls
        if "Action:" in raw_response and "Action Input:" in raw_response:
            parts = raw_response.split("Action:")
            for i in range(1, len(parts)):
                action_part = parts[i]
                if "Action Input:" in action_part:
                    tool_name = action_part.split("Action Input:")[0].strip()
                    tool_input = action_part.split("Action Input:")[1].strip()
                    
                    # Cut off at next section
                    for delimiter in ["\nObservation:", "\nAction:", "\nThought:", "\nFinal Answer:"]:
                        if delimiter in tool_input:
                            tool_input = tool_input.split(delimiter)[0].strip()
                            break
                    
                    # Parse tool input
                    try:
                        tool_args = json.loads(tool_input)
                    except:
                        tool_args = {"input": tool_input}
                    
                    function_calls.append(FunctionCall(name=tool_name, arguments=tool_args))
        
        return {
            "thinking": thinking,
            "response": response,
            "function_calls": function_calls
        }