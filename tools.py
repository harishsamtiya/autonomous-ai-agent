"""
Tool definitions for the Autonomous AI Agent.
Integrates 8 tools for multi-step reasoning tasks.
"""

import json
import math
import datetime
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information on any topic.
    Args:
        query: The search query string.
    Returns:
        Search results as a formatted string.
    """
    # In production, integrate with SerpAPI / Tavily / Google Search API
    return f"[Web Search] Results for '{query}': This is a placeholder. Connect a real search API (e.g., Tavily, SerpAPI) for live results."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Args:
        expression: A valid Python math expression (e.g., '2**10 + math.sqrt(144)').
    Returns:
        The computed result as a string.
    """
    try:
        allowed_names = {
            "math": math, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum,
            "pow": pow, "int": int, "float": float,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time.
    Returns:
        Current date and time in a readable format.
    """
    now = datetime.datetime.now()
    return now.strftime("Current date: %B %d, %Y | Time: %I:%M %p")


@tool
def wikipedia_search(topic: str) -> str:
    """Search Wikipedia for information about a topic.
    Args:
        topic: The topic to search for on Wikipedia.
    Returns:
        A summary of the Wikipedia article.
    """
    try:
        import wikipedia
        summary = wikipedia.summary(topic, sentences=3)
        return f"[Wikipedia] {summary}"
    except ImportError:
        return "[Wikipedia] wikipedia package not installed. Run: pip install wikipedia"
    except Exception as e:
        return f"[Wikipedia] Error: {str(e)}"


@tool
def text_summarizer(text: str) -> str:
    """Summarize a given text into key points.
    Args:
        text: The text to summarize.
    Returns:
        A concise summary of the text.
    """
    sentences = text.split(". ")
    if len(sentences) <= 3:
        return f"Summary: {text}"

    key_sentences = sentences[:3]
    summary = ". ".join(key_sentences) + "."
    return f"Summary ({len(sentences)} sentences → 3): {summary}"


@tool
def json_analyzer(json_string: str) -> str:
    """Analyze and extract insights from JSON data.
    Args:
        json_string: A valid JSON string to analyze.
    Returns:
        Analysis of the JSON structure and key insights.
    """
    try:
        data = json.loads(json_string)
        analysis = {
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "length": len(data) if isinstance(data, (list, dict)) else None,
        }
        return f"JSON Analysis: {json.dumps(analysis, indent=2)}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


@tool
def text_translator(text: str, target_language: str = "Spanish") -> str:
    """Translate text to a target language.
    Args:
        text: The text to translate.
        target_language: The target language for translation.
    Returns:
        Translated text (placeholder - connect a real translation API).
    """
    return f"[Translation to {target_language}] '{text}' → Connect Google Translate API or DeepL for production use."


@tool
def file_reader(file_path: str) -> str:
    """Read the contents of a text file.
    Args:
        file_path: Path to the file to read.
    Returns:
        The file contents as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"File contents ({len(content)} chars):\n{content[:2000]}"
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def weather_lookup(city: str) -> str:
    """Get current weather information for a city.
    Args:
        city: The city name to look up weather for.
    Returns:
        Weather information (placeholder - connect OpenWeatherMap API).
    """
    return f"[Weather for {city}] Connect OpenWeatherMap API for live data. API key required."


def get_all_tools():
    """Return a list of all available tools."""
    return [
        web_search,
        calculator,
        get_current_datetime,
        wikipedia_search,
        text_summarizer,
        json_analyzer,
        text_translator,
        file_reader,
        weather_lookup,
    ]
