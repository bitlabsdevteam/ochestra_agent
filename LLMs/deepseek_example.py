#!/usr/bin/env python3
"""
Example usage of the LLMFactory class with Deepseek LLM via Groq API.

This script demonstrates how to use the LLMFactory to create and use Deepseek LLM instances
with the deepseek-r1-distill-llama-70b model via Groq API. It includes examples of direct LLM invocation, 
using LangChain chains, and generating creative content.

Note: Before running this script, make sure to set the GROQ_API_KEY environment variable:
    export GROQ_API_KEY=your_groq_api_key_here

Or create a .env file with the following content:
    GROQ_API_KEY=your_groq_api_key_here
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
    """Check if the Groq API key is set and provide guidance if not."""
    if not os.getenv("GROQ_API_KEY"):
        print("\nError: GROQ_API_KEY environment variable is not set.")
        print("\nPlease set your Groq API key using one of the following methods:")
        print("\n1. Create a .env file in the project root with the following content:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("\n2. Set the environment variable directly:")
        print("   export GROQ_API_KEY=your_groq_api_key_here  # Linux/macOS")
        print("   set GROQ_API_KEY=your_groq_api_key_here     # Windows Command Prompt")
        print("   $env:GROQ_API_KEY='your_groq_api_key_here'  # Windows PowerShell")
        return False
    return True

def main():
    """Run the example."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if Groq API key is set
    if not check_api_key():
        return
    
    # Example 1: Direct LLM invocation
    print("\n=== Example 1: Direct Deepseek LLM invocation ===\n")
    
    try:
        # Get a Deepseek LLM instance with default settings
        llm = LLMFactory.get_llm(prefix="deepseek")
        
        # Invoke the LLM directly
        response = llm.invoke("What is the capital of France?")
        print("Response:")
        print(response.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: LLM with Chain
    print("\n=== Example 2: Deepseek LLM with Chain ===\n")
    
    try:
        # Get a Deepseek LLM instance with custom temperature
        llm = LLMFactory.get_llm(prefix="deepseek", temperature=0.7)
        
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

    # Example 3: Creative content generation with Deepseek
    print("\n=== Example 3: Creative content generation with Deepseek ===\n")
    
    try:
        # Get Deepseek LLM instance
        deepseek_llm = LLMFactory.get_llm(prefix="deepseek", temperature=0.8)
        
        # Create a simple prompt template
        template = """Generate a short poem about {topic}."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        deepseek_chain = prompt | deepseek_llm | StrOutputParser()
        
        # Run the chain
        topic = "machine learning"
        print(f"Topic: {topic}\n")
        
        print("Deepseek response:")
        try:
            deepseek_response = deepseek_chain.invoke({"topic": topic})
            print(deepseek_response)
        except Exception as e:
            print(f"Error: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()