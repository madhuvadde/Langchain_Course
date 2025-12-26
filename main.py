import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources """

    answer:str = Field(description="The agents answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources to generate the answer")

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the Internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for: {query}")
#     result = tavily.search(query=query)
#     return result
# return "Tokyo weather is sunny"


llm = ChatOpenAI(
    temperature=0,
    model="openai/gpt-4o",
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.github.ai/inference",
)

# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        # {"messages": HumanMessage("What is the weather in Hyderabad, India?")}
        {"messages": HumanMessage(content="Search for 3 job postings for an AI engineer using Langchain in the New York on linkedIn and list their details?")}
    )
    print(result)


if __name__ == "__main__":
    main()
