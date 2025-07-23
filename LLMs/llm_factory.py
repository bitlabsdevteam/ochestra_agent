"""LLM Factory module for the Sales Maker application.

This module provides a factory class for creating LLM instances from different providers
including OpenAI, Google Gemini, and Groq.
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

class LLMFactory:
    """A factory class for creating LLM instances.
    
    This class provides methods to create and configure LLM instances from different providers
    including OpenAI, Google Gemini, and Groq DeepSeek.
    """
    
    @staticmethod
    def get_llm(prefix: str = "openai", **kwargs):
        """Get an LLM instance based on the specified provider.
        
        Args:
            prefix: The provider prefix. Currently supported: 'openai' (uses gpt-4o), 'gemini' (uses gemini-1.5-flash),
                   'deepseek' (uses deepseek-r1-distill-llama-70b).
            **kwargs: Additional keyword arguments to pass to the LLM constructor.
            
        Returns:
            BaseChatModel: An LLM instance from the specified provider.
            
        Raises:
            ValueError: If the provider is not supported.
        """
        prefix = prefix.lower()
        
        if prefix == 'openai':
            return LLMFactory._get_openai_llm(**kwargs)
        elif prefix == 'gemini':
            return LLMFactory._get_gemini_llm(**kwargs)
        elif prefix == 'deepseek':
            return LLMFactory._get_deepseek_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {prefix}. Supported providers: 'openai', 'gemini', 'deepseek'.")
    
    @staticmethod
    def _get_openai_llm(**kwargs) -> ChatOpenAI:
        """Get an OpenAI LLM instance.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.
            
        Returns:
            A ChatOpenAI instance configured with GPT-4o.
            
        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
        """
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your environment or .env file."
            )
            
        # Always use GPT-4o model, ignoring any model specified in kwargs
        kwargs['model'] = 'gpt-4o'
            
        return ChatOpenAI(**kwargs)
    
    @staticmethod
    def _get_gemini_llm(**kwargs) -> ChatGoogleGenerativeAI:
        """Get a Google Gemini LLM instance.
        
        This method creates a ChatGoogleGenerativeAI instance using the gemini-1.5-flash model.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the ChatGoogleGenerativeAI constructor.
            
        Returns:
            A ChatGoogleGenerativeAI instance configured with gemini-1.5-flash.
            
        Raises:
            ValueError: If the GEMINI_API_KEY environment variable is not set.
        """
        # Check if GEMINI_API_KEY is set
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your environment or .env file."
            )
            
        # Always use gemini-1.5-flash model, ignoring any model specified in kwargs
        kwargs['model'] = 'gemini-1.5-flash'
            
        # Ensure we're using the correct environment variable
        if not kwargs.get('google_api_key'):
            kwargs['google_api_key'] = os.getenv("GEMINI_API_KEY")
            
        return ChatGoogleGenerativeAI(**kwargs)
    
    @staticmethod
    def _get_deepseek_llm(**kwargs) -> ChatGroq:
        """Get a Deepseek LLM instance via Groq API.
        
        This method creates and returns a ChatGroq instance configured with the
        'deepseek-r1-distill-llama-70b' model.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the ChatGroq constructor.
            
        Returns:
            ChatGroq: A Deepseek LLM instance via Groq.
            
        Raises:
            ValueError: If the GROQ_API_KEY environment variable is not set.
        """
        # Check if GROQ_API_KEY is set
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it in your environment or .env file."
            )
            
        # Always use deepseek-r1-distill-llama-70b model, ignoring any model specified in kwargs
        kwargs['model'] = 'deepseek-r1-distill-llama-70b'
            
        return ChatGroq(**kwargs)