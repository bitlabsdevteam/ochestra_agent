"""Orchestra Agent module for the Sales Maker application.

This module provides an agent class that orchestrates various tasks using LangChain's ReAct agent pattern.
It integrates tools and memory for enhanced capabilities.
"""

from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from LLMs.llm_factory import LLMFactory

class OrchestraAgent:
    """An agent that orchestrates various tasks using OpenAI's LLM.
    
    This agent uses the LLMFactory to create an OpenAI LLM instance and provides
    methods for processing user queries and coordinating responses.
    """
    
    def __init__(self, system_prompt: Optional[str] = None, temperature: float = 0.7):
        """Initialize the OrchestraAgent.
        
        Args:
            system_prompt: An optional system prompt to guide the agent's behavior.
            temperature: The temperature parameter for the LLM (controls randomness).
        """
        self.llm = LLMFactory.get_llm("openai", temperature=temperature)
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Set default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an orchestration agent that helps coordinate tasks and provide helpful responses. "
                "Answer questions concisely and accurately."
            )
        
        # Add system message to conversation history
        self.conversation_history.append({"role": "system", "content": system_prompt})
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response.
        
        Args:
            query: The user's query string.
            
        Returns:
            The agent's response to the query.
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Convert conversation history to LangChain message format
        messages = self._convert_history_to_messages()
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        return response.content
    
    def _convert_history_to_messages(self):
        """Convert conversation history to LangChain message format.
        
        Returns:
            A list of LangChain message objects.
        """
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
        """Clear the conversation history except for the system prompt."""
        system_prompt = next((msg for msg in self.conversation_history if msg["role"] == "system"), None)
        self.conversation_history = [system_prompt] if system_prompt else []
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history.
        
        Returns:
            The conversation history as a list of message dictionaries.
        """
        return self.conversation_history.copy()