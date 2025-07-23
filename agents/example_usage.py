"""Example usage of the OrchestraAgent class.

This module demonstrates how to use the OrchestraAgent class.
"""

import os
from dotenv import load_dotenv
from orchestra_agent import OrchestraAgent

# Load environment variables
load_dotenv()

def main():
    """Run the example."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Set OPENAI_API_KEY in your .env file.")
        return
    
    # Create an OrchestraAgent instance
    agent = OrchestraAgent(
        system_prompt="You are a helpful assistant that provides concise and accurate information.",
        temperature=0.7
    )
    
    # Process a query
    print("\n=== Example Query 1 ===\n")
    query1 = "What are the benefits of using a factory pattern for creating LLM instances?"
    print(f"Query: {query1}")
    response1 = agent.process_query(query1)
    print(f"Response: {response1}\n")
    
    # Process another query that builds on the conversation
    print("\n=== Example Query 2 (Follow-up) ===\n")
    query2 = "Can you provide a simple code example of this pattern?"
    print(f"Query: {query2}")
    response2 = agent.process_query(query2)
    print(f"Response: {response2}\n")
    
    # Display conversation history
    print("\n=== Conversation History ===\n")
    history = agent.get_conversation_history()
    for message in history:
        if message["role"] != "system":  # Skip system message for clarity
            print(f"{message['role'].capitalize()}: {message['content']}\n")
    
    # Clear history and start a new conversation
    agent.clear_history()
    print("\n=== New Conversation After Clearing History ===\n")
    query3 = "Hello, can you tell me about orchestration in software architecture?"
    print(f"Query: {query3}")
    response3 = agent.process_query(query3)
    print(f"Response: {response3}\n")

if __name__ == "__main__":
    main()