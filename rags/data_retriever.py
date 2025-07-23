import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataRetriever:
    def __init__(self, provider_name):
        """
        Initialize the DataRetriever with a provider name.
        
        Args:
            provider_name (str): The name of the provider (e.g., 'openAI')
        """
        self.provider_name = provider_name
        
        # Load API keys from environment variables
        self.api_keys = {
            'openai': os.environ.get('OPENAI_API_KEY'),
            'huggingface': os.environ.get('HUGGINGFACE_API_KEY'),
            'gemini': os.environ.get('GEMINI_API_KEY')
        }
    
    def get_embedding_model(self):
        """
        Returns the appropriate embedding model based on the provider name.
        
        Returns:
            str: The name of the embedding model
        """
        provider = self.provider_name.lower()
        
        if provider == 'openai':
            return "text-embedding-ada-002"  # OpenAI embedding model
        elif provider == 'huggingface':
            return "sentence-transformers/all-mpnet-base-v2"  # Hugging Face embedding model
        elif provider == 'gemini':
            return "embedding-001"  # Google Gemini embedding model
        else:
            return f"No embedding model available for {self.provider_name}"
    
    def get_api_key(self):
        """
        Returns the API key for the current provider.
        
        Returns:
            str: The API key or None if not available
        """
        provider = self.provider_name.lower()
        return self.api_keys.get(provider)
    
    def has_valid_api_key(self):
        """
        Checks if the current provider has a valid API key.
        
        Returns:
            bool: True if a valid API key exists, False otherwise
        """
        api_key = self.get_api_key()
        return api_key is not None and len(api_key) > 0
    
    def __str__(self):
        """
        String representation of the DataRetriever.
        
        Returns:
            str: A string representation of the DataRetriever
        """
        has_key = "with API key" if self.has_valid_api_key() else "without API key"
        return f"DataRetriever(provider_name={self.provider_name}, {has_key})"