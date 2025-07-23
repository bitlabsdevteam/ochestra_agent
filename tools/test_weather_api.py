"""Test script for the WeatherAPI.com integration.

This script demonstrates how to use the weather tool with a valid WeatherAPI.com API key.
"""

import os
from dotenv import load_dotenv
from weather_tool import get_weather, WeatherTool

# Load environment variables
load_dotenv()

def main():
    """Run the test."""
    # Check if WeatherAPI.com API key is set
    api_key = os.getenv("WEATHERAPI_KEY")
    if not api_key or api_key == "4c6e8f9c9c9e4a9c9c9c9c9c9c9c9c9c":
        print("\nPlease set a valid WeatherAPI.com API key in the .env file.")
        print("You can sign up for a free API key at https://www.weatherapi.com/my/")
        print("Then add it to your .env file as:")
        print("WEATHERAPI_KEY=your_api_key_here")
        return
    
    # Method 1: Using the tool function
    print("\n=== Using the get_weather() function ===\n")
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        print(f"Weather in {city}:")
        weather_data = get_weather(city)
        
        if "error" in weather_data:
            print(f"  Error: {weather_data['error']}")
        else:
            print(f"  City: {weather_data['city']}, {weather_data['country']}")
            print(f"  Temperature: {weather_data['temperature']}")
            print(f"  Feels like: {weather_data['feels_like']}")
            print(f"  Weather: {weather_data['weather']}")
            print(f"  Humidity: {weather_data['humidity']}")
            print(f"  Wind speed: {weather_data['wind_speed']}")
        print()
    
    # Method 2: Using the WeatherTool class directly
    print("\n=== Using the WeatherTool class directly ===\n")
    weather_tool = WeatherTool()
    
    # Test with city and country
    print("Weather in Paris, France:")
    weather_data = weather_tool._run("Paris", "FR")
    
    if "error" in weather_data:
        print(f"  Error: {weather_data['error']}")
    else:
        print(f"  City: {weather_data['city']}, {weather_data['country']}")
        print(f"  Temperature: {weather_data['temperature']}")
        print(f"  Weather: {weather_data['weather']}")

if __name__ == "__main__":
    main()