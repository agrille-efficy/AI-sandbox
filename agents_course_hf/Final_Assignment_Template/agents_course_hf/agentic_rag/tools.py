from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
import random
from huggingface_hub import list_models

search_tool = DuckDuckGoSearchRun()


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""

    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Sunny", "temp_c": 25},
        {"condition": "Cloudy", "temp_c": 20},
        {"condition": "Snowy", "temp_c": -5},
    ]

    data = random.choice(weather_conditions) 
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool

weather_info_tool = Tool(
    name="weather_info_tool",
    func=get_weather_info, 
    description="Fetches dummy weather information for a given location."
)

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models: 
            model = models[0]
            return f"Most downloaded model by {author} is {model.id} with {model.downloads} downloads." 
        else: 
            return f"No models found for author {author}." 
    except Exception as e: 
        return f"Error fetching models for {author}: {str(e)}" 
    
# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub." 
)

tools = [search_tool, weather_info_tool, hub_stats_tool]