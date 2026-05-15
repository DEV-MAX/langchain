from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.tools  import tool
from tavily import TavilyClient 
from pydantic import BaseModel
from typing import cast

class IPL(BaseModel):
    '''Model to represent the information of a IPL match which contains today's match information and the standing of a team in the IPL season point table'''
    today_match: TodayMatch
    standing: list[IPLStanding]


class TodayMatch(BaseModel):
    '''Model to represent today's match information'''
    team1: str
    team2: str
    date: str
    venue: str
    time: str 
    weather:WeatherInfo      

class IPLStanding(BaseModel):
    '''Model to represent the standing of a team in the IPL season point table'''
    team: str
    matches_played: int
    wins: int
    losses: int
    points: int
    NRR: float

class IPLSeasonPointTable(BaseModel):
    '''Model to represent the IPL season point table'''
    season: str
    standings: list[IPLStanding]    

class WeatherInfo(BaseModel):
    '''Model to represent the weather information'''
    location: str
    temperature: float
    description: str   

load_dotenv()

tavily=TavilyClient()

@tool(description="Search for information on the web")
def search_tool(query:str)-> str:
    websearch=tavily.search(query=query)
    return websearch


@tool(description="Fetch today's match information")
def today_match_info(query:str)-> str:
    match_info=tavily.search(query=query)
    return match_info   

@tool(description="Fetch weather information of a venue location")
def get_weather(query:str)-> str:
    weather_search=tavily.search(query=query)
    return weather_search

model=ChatOpenRouter(model="inclusionai/ring-2.6-1t:free", temperature=0.1)
tools=[search_tool, today_match_info, get_weather]
agent=create_agent(model,tools,response_format=IPL)

def main():
    print("Hello from langchain!")
    result= agent.invoke(input={"messages":[{"role":"user","content":"Can you fetch the IPL 2026 point table for me and also today match and the weather details of the venue?"}]})
    response = cast(IPL, result["structured_response"])    
    print("response from agent: ", response)

if __name__ == "__main__":
    main()
