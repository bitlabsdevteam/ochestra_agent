"""Example usage of the LangChain tools.

This script demonstrates how to use the weather, time, and city facts tools.
"""

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Import our custom tools
from weather_tool import get_weather, WeatherTool
from time_tool import get_time, get_current_time, TimeTool
from city_facts_tool import get_city_facts, search_wikipedia, CityFactsTool

# Load environment variables
load_dotenv()

def main():
    """Run the example."""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0)
    
    # Method 1: Using the tool functions directly
    print("\n=== Using Tool Functions Directly ===")
    
    # Get weather
    print("\nWeather in New York:")
    weather_tool = WeatherTool()
    weather_data = weather_tool._run("New York", "US")
    print(weather_data)
    
    # Get time
    print("\nTime in London:")
    time_data = get_time("London")
    print(time_data)
    
    # Get current time
    print("\nCurrent time in UTC:")
    current_time = get_current_time()
    print(current_time)
    
    # Get city facts
    print("\nFacts about Paris:")
    city_facts = get_city_facts("Paris")
    print(city_facts)
    
    # Search Wikipedia
    print("\nWikipedia search for 'Tokyo':")
    wiki_results = search_wikipedia("Tokyo")
    print(wiki_results)
    
    # Method 2: Using the tools with an agent
    print("\n\n=== Using Tools with an Agent ===")
    
    # Create tool instances
    weather_tool = WeatherTool()
    time_tool = TimeTool()
    city_facts_tool = CityFactsTool()
    
    # Create a list of tools
    tools = [weather_tool, time_tool, city_facts_tool]
    
    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Run the agent
    print("\nAsking the agent about the weather in Tokyo:")
    agent.run("What's the weather like in Tokyo right now?")
    
    print("\nAsking the agent about the time in New York:")
    agent.run("What time is it in New York?")
    
    print("\nAsking the agent about facts about Berlin:")
    agent.run("Tell me some interesting facts about Berlin.")

if __name__ == "__main__":
    main()