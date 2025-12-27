import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
# from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(
    temperature=0,
    model="openai/gpt-4o",
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.github.ai/inference",
)
structured_llm = llm.with_structured_output(AgentResponse)
# react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "tools",
        "input",
        "tool_names",
        "agent_scratchpad",
    ],
).partial(format_instructions="")
# ).partial(format_instructions=output_parser.get_format_instructions())


# agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
# chain = agent_executor | extract_output | parse_output
chain = agent_executor | extract_output | structured_llm


def main():
    # print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an AI engineer using Langchain in the New York on linkedIn and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
