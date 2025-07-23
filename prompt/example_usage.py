"""Example usage of the PromptLibrary class.

This script demonstrates how to use the travel agent prompt template.
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompt_library import PromptLibrary

# Load environment variables
load_dotenv()

def main():
    """Run the example."""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0.7)
    
    # Example 1: Using the string prompt directly
    print("\n=== Example 1: Using the string prompt directly ===\n")
    
    # Get the travel agent prompt string
    prompt_str = PromptLibrary.get_travelling_agent_prompt()
    
    # Replace the placeholder with an actual query
    formatted_prompt = prompt_str.replace("{user_query}", "I want to visit Japan for 7 days in April. I love food, culture, and nature.")
    
    # Print the formatted prompt
    print("Formatted Prompt:")
    print(formatted_prompt)
    
    # Example 2: Using the PromptTemplate
    print("\n=== Example 2: Using the PromptTemplate ===\n")
    
    # Get the travel agent prompt template
    prompt_template = PromptLibrary.get_travelling_agent_prompt_template()
    
    # Format the prompt template with a query
    formatted_prompt = prompt_template.format(user_query="I'm planning a beach vacation in Thailand for 5 days. I want to relax but also do some water activities.")
    
    # Print the formatted prompt
    print("Formatted Prompt:")
    print(formatted_prompt)
    
    # Example 3: Using the ChatPromptTemplate with an LLM
    print("\n=== Example 3: Using the ChatPromptTemplate with an LLM ===\n")
    
    # Get the travel agent chat template
    chat_template = PromptLibrary.get_travelling_agent_chat_template()
    
    # Create a simple chain
    chain = chat_template | llm | StrOutputParser()
    
    # Run the chain with a query
    query = "I want to visit New York City for a weekend. I'm interested in museums, Broadway shows, and trying local food."
    print(f"Query: {query}")
    
    try:
        response = chain.invoke({"user_query": query})
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()