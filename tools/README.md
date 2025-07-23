# LangChain Tools

This directory contains custom LangChain tools for weather information, time data, and city facts.

## Tools Overview

### Weather Tool

The Weather Tool provides current weather information for cities using the WeatherAPI.com API.

- **File**: `weather_tool.py`
- **Classes**: `WeatherTool`
- **Functions**: `get_weather()`
- **Requirements**: WeatherAPI.com API key (set as `WEATHERAPI_KEY` environment variable)

### Time Tool

The Time Tool provides current time information for different cities around the world.

- **File**: `time_tool.py`
- **Classes**: `TimeTool`
- **Functions**: `get_time()`, `get_current_time()`
- **Features**: Includes a mapping of common cities to their timezones and falls back to WorldTimeAPI

### City Facts Tool

The City Facts Tool provides information and facts about cities using Wikipedia.

- **File**: `city_facts_tool.py`
- **Classes**: `CityFactsTool`
- **Functions**: `get_city_facts()`, `search_wikipedia()`
- **Features**: Uses both a custom Wikipedia implementation and LangChain's built-in WikipediaQueryRun tool

## Usage

### Direct Usage

```python
from tools.weather_tool import get_weather
from tools.time_tool import get_time, get_current_time
from tools.city_facts_tool import get_city_facts, search_wikipedia

# Get weather for a city
weather_data = get_weather("New York", "US")

# Get time for a city
time_data = get_time("London")

# Get current time (UTC by default)
current_time = get_current_time()

# Get facts about a city
city_facts = get_city_facts("Paris")

# Search Wikipedia
wiki_results = search_wikipedia("Tokyo")
```

### Usage with LangChain Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from tools.weather_tool import WeatherTool
from tools.time_tool import TimeTool
from tools.city_facts_tool import CityFactsTool

# Create tool instances
weather_tool = WeatherTool()
time_tool = TimeTool()
city_facts_tool = CityFactsTool()

# Create a list of tools
tools = [weather_tool, time_tool, city_facts_tool]

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
agent.run("What's the weather like in Tokyo right now?")
```

## Example

See `example_usage.py` for a complete example of how to use these tools.