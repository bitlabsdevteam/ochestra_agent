"""Example usage of the OrchestraAgent as a travel agent.

This script demonstrates how to use the OrchestraAgent with travel agent functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestra_agent import OrchestraAgent

# Load environment variables
load_dotenv()

def main():
    """Run the travel agent example."""
    # Check if required API keys are set
    if not os.getenv("GROQ_API_KEY"):
        print("\nPlease set the GROQ_API_KEY in your .env file to use the DeepSeek LLM.")
        print("You can sign up for a free API key at https://console.groq.com/")
        return
    
    if not os.getenv("WEATHERAPI_KEY"):
        print("\nPlease set the WEATHERAPI_KEY in your .env file for the weather tool.")
        print("You can sign up for a free API key at https://www.weatherapi.com/my/")
        return
    
    # Create the OrchestraAgent with DeepSeek LLM
    print("\n=== Initializing Travel Agent with DeepSeek LLM ===")
    agent = OrchestraAgent(llm_prefix="openai", temperature=0.7, use_tools=True)
    
    # Example travel queries
    travel_queries = [
        "I want to know about Paris such as weather, local time, and City Facts?"
    
    ]
    
    # Process each query using the travel agent functionality
    for i, query in enumerate(travel_queries, 1):
        print(f"\n=== Travel Query {i} ===")
        print(f"Query: {query}")
        print("\nProcessing...\n")
        
        # Use the travel agent to process the query
        result = agent.process_query(query, use_travel_agent=True)
        
        # Check if the result is a structured response or a raw string
        if isinstance(result, dict):
            print("\nThinking:")
            print(result["thinking"])
            
            print("\nFunction Calls:")
            if result["function_calls"]:
                for i, func_call in enumerate(result["function_calls"], 1):
                    print(f"  {i}. Tool: {func_call.name}")
                    print(f"     Parameters: {func_call.arguments}")
            else:
                print("  No function calls")
            
            print("\nResponse:")
            print(result["response"])
        else:
            print(f"Response:\n{result}")
        
        print("-" * 80)
    
    # Example of using the agent without travel functionality
    print("\n=== Using Agent Without Travel Functionality ===")
    regular_query = "What's the capital of France?"
    print(f"Query: {regular_query}")
    
    result = agent.process_query(regular_query, use_travel_agent=False)
    
    # Check if the result is a structured response or a raw string
    if isinstance(result, dict):
        print("\nThinking:")
        print(result["thinking"])
        
        print("\nFunction Calls:")
        if result["function_calls"]:
            for i, func_call in enumerate(result["function_calls"], 1):
                print(f"  {i}. Tool: {func_call.name}")
                print(f"     Parameters: {func_call.arguments}")
        else:
            print("  No function calls")
        
        print("\nResponse:")
        print(result["response"])
    else:
        print(f"\nResponse:\n{result}")

if __name__ == "__main__":
    main()