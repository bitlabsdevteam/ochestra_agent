from data_retriever import DataRetriever

def test_data_retriever():
    # Test with 'openAI' provider
    openai_retriever = DataRetriever("openAI")
    print(f"Provider: openAI, Embedding Model: {openai_retriever.get_embedding_model()}")
    print(f"  - Expected: text-embedding-ada-002")
    
    # Test with different case 'OpenAI' provider
    openai_case_retriever = DataRetriever("OpenAI")
    print(f"Provider: OpenAI, Embedding Model: {openai_case_retriever.get_embedding_model()}")
    print(f"  - Expected: text-embedding-ada-002")
    
    # Test with a different provider
    other_retriever = DataRetriever("other_provider")
    print(f"Provider: other_provider, Embedding Model: {other_retriever.get_embedding_model()}")
    print(f"  - Expected: No embedding model available message")
    
    # Test string representation
    print(f"String representation: {openai_retriever}")

if __name__ == "__main__":
    test_data_retriever()