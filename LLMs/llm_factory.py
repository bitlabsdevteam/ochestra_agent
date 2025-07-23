"""LLM Factory module for the Sales Maker application.

This module provides a factory class for creating OpenAI LLM instances.
"""

from langchain_openai import ChatOpenAI

class LLMFactory:
    """A factory class for creating OpenAI LLM instances.
    
    This class provides methods to create and configure OpenAI LLM instances.
    """
    
    @staticmethod
    def get_llm(prefix: str = "openai", **kwargs) -> ChatOpenAI:
        """Get an OpenAI LLM instance.
        
        Args:
            prefix: The provider prefix (only 'openai' is supported).
            **kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.
            
        Returns:
            A ChatOpenAI instance configured with GPT-4o.
            
        Raises:
            ValueError: If the prefix is not 'openai'.
        """
        prefix = prefix.lower()
        
        if prefix == 'openai':
            return LLMFactory._get_openai_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {prefix}. Only 'openai' is supported.")
    
    @staticmethod
    def _get_openai_llm(**kwargs) -> ChatOpenAI:
        """Get an OpenAI LLM instance.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.
            
        Returns:
            A ChatOpenAI instance configured with GPT-4o.
        """
        # Always use GPT-4o model, ignoring any model specified in kwargs
        kwargs['model'] = 'gpt-4o'
            
        return ChatOpenAI(**kwargs)