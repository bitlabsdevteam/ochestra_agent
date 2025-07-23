import os
import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from typing import List, Dict, Any, Optional, Union

# Load environment variables
load_dotenv()

class VectorDBManager:
    """
    A class to manage Pinecone vector database operations including initialization,
    index creation, document addition, and retrieval.
    """
    
    def __init__(self, provider_name: str = "openai"):
        """
        Initialize the VectorDBManager with a provider name.
        
        Args:
            provider_name (str): The name of the embedding provider (e.g., 'openai', 'huggingface', 'gemini')
        """
        self.provider_name = provider_name.lower()
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY') or os.environ.get('PINECODE_API_KEY')
        self.pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT') or os.environ.get('PINECODE_ENVIRONMENT', 'gcp-starter')
        self.index_name = os.environ.get('PINECONE_INDEX_NAME') or os.environ.get('PINECODE_INDEX_NAME', 'sales-maker-index')
        self.dimension = self._get_dimension_for_provider()
        
        # Load API keys from environment variables
        self.api_keys = {
            'openai': os.environ.get('OPENAI_API_KEY'),
            'huggingface': os.environ.get('HUGGINGFACE_API_KEY'),
            'gemini': os.environ.get('GEMINI_API_KEY')
        }
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _get_dimension_for_provider(self) -> int:
        """
        Returns the appropriate dimension for the embedding model based on the provider name.
        
        Returns:
            int: The dimension of the embedding model
        """
        if self.provider_name == 'openai':
            return 1536  # OpenAI ada-002 dimension
        elif self.provider_name == 'huggingface':
            return 768   # Default for sentence-transformers/all-mpnet-base-v2
        elif self.provider_name == 'gemini':
            return 768   # Gemini embedding dimension
        else:
            return 1536  # Default to OpenAI dimension
    
    def _init_pinecone(self) -> None:
        """
        Initialize the Pinecone client.
        
        Raises:
            ValueError: If Pinecone API key is not set
        """
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY environment variable.")
        
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
    
    def create_index(self, index_name: Optional[str] = None, dimension: Optional[int] = None) -> str:
        """
        Create a new Pinecone index if it doesn't exist.
        
        Args:
            index_name (str, optional): Name of the index to create. Defaults to self.index_name.
            dimension (int, optional): Dimension of the vectors. Defaults to self.dimension.
            
        Returns:
            str: Name of the created or existing index
            
        Raises:
            ValueError: If index creation fails
        """
        index_name = index_name or self.index_name
        dimension = dimension or self.dimension
        
        # Check if index already exists
        existing_indexes = pinecone.list_indexes()
        
        if index_name not in existing_indexes:
            try:
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {index_name} with dimension {dimension}")
            except Exception as e:
                raise ValueError(f"Failed to create Pinecone index: {str(e)}")
        else:
            print(f"Using existing Pinecone index: {index_name}")
        
        return index_name
    
    def get_embedding_model(self):
        """
        Returns the appropriate embedding model based on the provider name.
        
        Returns:
            Embedding model instance
            
        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        if self.provider_name == 'openai':
            api_key = self.api_keys.get('openai')
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            return OpenAIEmbeddings(openai_api_key=api_key)
            
        elif self.provider_name == 'huggingface':
            api_key = self.api_keys.get('huggingface')
            model_name = "sentence-transformers/all-mpnet-base-v2"
            if api_key:
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    huggingfacehub_api_token=api_key
                )
            else:
                # Use local model if no API key
                return HuggingFaceEmbeddings(model_name=model_name)
                
        elif self.provider_name == 'gemini':
            api_key = self.api_keys.get('gemini')
            if not api_key:
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
            return GoogleGenerativeAIEmbeddings(
                model="embedding-001",
                google_api_key=api_key
            )
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
    
    def add_documents(self, documents: List[Union[Document, Dict[str, Any]]], namespace: str = "") -> None:
        """
        Add documents to the Pinecone index.
        
        Args:
            documents: List of documents to add. Can be langchain Document objects or dictionaries
                      with 'page_content' and 'metadata' keys.
            namespace: Namespace to add documents to (optional)
            
        Raises:
            ValueError: If documents cannot be added
        """
        try:
            # Ensure index exists
            self.create_index()
            
            # Convert dict documents to langchain Documents if needed
            processed_docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    processed_docs.append(Document(
                        page_content=doc.get('page_content', ''),
                        metadata=doc.get('metadata', {})
                    ))
                else:
                    processed_docs.append(doc)
            
            # Get embedding model
            embeddings = self.get_embedding_model()
            
            # Create vector store and add documents
            vector_store = PineconeVectorStore.from_documents(
                documents=processed_docs,
                embedding=embeddings,
                index_name=self.index_name,
                namespace=namespace
            )
            
            print(f"Added {len(processed_docs)} documents to Pinecone index {self.index_name}")
            return vector_store
            
        except Exception as e:
            raise ValueError(f"Failed to add documents to Pinecone: {str(e)}")
    
    def get_retriever(self, namespace: str = "", search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the Pinecone vector store.
        
        Args:
            namespace: Namespace to search in (optional)
            search_kwargs: Additional search parameters (optional)
            
        Returns:
            A retriever object that can be used to query the vector store
            
        Raises:
            ValueError: If retriever cannot be created
        """
        try:
            # Get embedding model
            embeddings = self.get_embedding_model()
            
            # Set default search kwargs if not provided
            if search_kwargs is None:
                search_kwargs = {"k": 4}  # Default to retrieving top 4 results
            
            # Create vector store
            vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=embeddings,
                namespace=namespace
            )
            
            # Return retriever
            return vector_store.as_retriever(search_kwargs=search_kwargs)
            
        except Exception as e:
            raise ValueError(f"Failed to create retriever: {str(e)}")
    
    def query(self, query_text: str, namespace: str = "", top_k: int = 4):
        """
        Query the vector store directly and return results.
        
        Args:
            query_text: The query text
            namespace: Namespace to search in (optional)
            top_k: Number of results to return
            
        Returns:
            List of documents that match the query
            
        Raises:
            ValueError: If query fails
        """
        try:
            retriever = self.get_retriever(namespace=namespace, search_kwargs={"k": top_k})
            return retriever.get_relevant_documents(query_text)
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")
    
    def delete_index(self, index_name: Optional[str] = None) -> None:
        """
        Delete a Pinecone index.
        
        Args:
            index_name: Name of the index to delete. Defaults to self.index_name.
            
        Raises:
            ValueError: If index deletion fails
        """
        index_name = index_name or self.index_name
        
        try:
            if index_name in pinecone.list_indexes():
                pinecone.delete_index(index_name)
                print(f"Deleted Pinecone index: {index_name}")
            else:
                print(f"Index {index_name} does not exist")
        except Exception as e:
            raise ValueError(f"Failed to delete Pinecone index: {str(e)}")