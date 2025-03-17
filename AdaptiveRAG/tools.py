from langchain_community.tools.tavily_search import TavilySearchResults
import os

tavily_api_key = os.getenv('TAVILY_API_KEY')

def get_web_search_tool():
    return TavilySearchResults()