#!/usr/bin/env python3
"""
Example usage of the LLMFactory class with OpenAI LLM.

This script demonstrates how to use the LLMFactory to create and use OpenAI LLM instances
with the gpt-4o model. It includes examples of direct LLM invocation, using LangChain
chains, and generating creative content.

Note: Before running this script, make sure to set the OPENAI_API_KEY environment variable:
    export OPENAI_API_KEY=your_openai_api_key_here

Or create a .env file with the following content:
    OPENAI_API_KEY=your_openai_api_key_here
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory to path to allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLMs.llm_factory import LLMFactory

def check_api_key():
    """Check if the OpenAI API key is set and provide guidance if not."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("\nPlease set your OpenAI API key using one of the following methods:")
        print("\n1. Create a .env file in the project root with the following content:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        print("\n2. Set the environment variable directly:")
        print("   export OPENAI_API_KEY=your_openai_api_key_here  # Linux/macOS")
        print("   set OPENAI_API_KEY=your_openai_api_key_here     # Windows Command Prompt")
        print("   $env:OPENAI_API_KEY='your_openai_api_key_here'  # Windows PowerShell")
        return False
    return True

def main():
    """Run the example."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not check_api_key():
        return
    
    # Example 1: Direct LLM invocation
    print("\n=== Example 1: Direct OpenAI LLM invocation ===\n")
    
    try:
        # Get an OpenAI LLM instance with default settings
        llm = LLMFactory.get_llm(prefix="openai")
        
        # Invoke the LLM directly
        response = llm.invoke("What is the capital of France?")
        print("Response:")
        print(response.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: LLM with Chain
    print("\n=== Example 2: OpenAI LLM with Chain ===\n")
    
    try:
        # Get an OpenAI LLM instance with custom temperature
        llm = LLMFactory.get_llm(prefix="openai", temperature=0.7)
        
        # Create a simple prompt template
        template = """You are a helpful assistant that provides information about {topic}.
        
        Please provide a brief overview of {topic}.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create a chain
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain
        response = chain.invoke({"topic": "artificial intelligence"})
        print("Response:")
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 3: Creative content generation with OpenAI
    print("\n=== Example 3: Creative content generation with OpenAI ===\n")
    
    try:
        # Get OpenAI LLM instance
        openai_llm = LLMFactory.get_llm(prefix="openai", temperature=0.8)
        
        # Create a simple prompt template
        template = """Generate a short poem about {topic}."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        openai_chain = prompt | openai_llm | StrOutputParser()
        
        # Run the chain
        topic = "machine learning"
        print(f"Topic: {topic}\n")
        
        print("OpenAI response:")
        try:
            openai_response = openai_chain.invoke({"topic": topic})
            print(openai_response)
        except Exception as e:
            print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()