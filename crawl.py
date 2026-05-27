from langchain_tavily import TavilyCrawl, TavilyMap
from langchain_tavily.tavily_search import TavilySearch
from dotenv import load_dotenv
load_dotenv()
import os

tavily_crawl=TavilyCrawl()
def crawl(url):
    response=tavily_crawl.invoke({"url": url, "max_depth": 5, "max_breadth": 10,"limit": 500})
    return [{"url": source["url"], "content": source["raw_content"]} for source in response["results"]]


if __name__ == "__main__":
    url = "https://python.langchain.com"
    response=crawl(url)    
    print(response)