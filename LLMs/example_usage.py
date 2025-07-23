"""Example usage of the LLMFactory class.

This script demonstrates how to use the LLMFactory class to create OpenAI LLM instances.
"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from llm_factory import LLMFactory  # Use direct import instead of from LLMs

# Load environment variables
load_dotenv()

def main():
    """Run the example."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Set OPENAI_API_KEY in your .env file.")
        return
    
    # Example with direct LLM invocation
    try:
        print("\n=== Testing Direct LLM Invocation ===\n")
        
        # Get OpenAI LLM
        llm = LLMFactory.get_llm("openai", temperature=0.7)
        
        # Direct invocation with a simple prompt
        prompt = "hello, how are you today?"
        print(f"Prompt: {prompt}")
        
        # Invoke the LLM directly
        response = llm.invoke(prompt)
        print(f"Response: {response.content}")
        print(f"Response type: {type(response)}")
        
    except Exception as e:
        print(f"Error with LLM invocation: {str(e)}")
        
    # Example with a chain (for comparison)
    try:
        print("\n=== Using LLM with Chain ===\n")
        
        # Create a simple prompt template
        prompt_template = PromptTemplate.from_template(
            "You are a helpful assistant. Answer the following question: {question}"
        )
        
        # Get default LLM
        default_llm = LLMFactory.get_llm(temperature=0)
        
        # Create a simple chain
        chain = prompt_template | default_llm | StrOutputParser()
        
        # Run the chain
        question = "What are the benefits of using a factory pattern for LLMs?"
        print(f"Question: {question}")
        response = chain.invoke({"question": question})
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error with LLM chain: {str(e)}")

if __name__ == "__main__":
    main()