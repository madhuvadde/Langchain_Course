import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(
    temperature=0,
    model="openai/gpt-4o",
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.github.ai/inference",
)
react_prompt = hub.pull('hwchase17/react')
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
chain = agent_executor

def main():
    # print("Hello from langchain-course!")
    result = chain.invoke(input={
        "input":"Search for 3 job postings for an AI engineer using Langchain in the New York on linkedIn and list their details"
    })
    print(result)


if __name__ == "__main__":
    main()
