from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ['TAVILY_API_KEY'] = "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"

def get_web_search_tool():
    return TavilySearchResults()