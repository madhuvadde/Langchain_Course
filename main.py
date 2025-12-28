import os
from typing import List, Tuple, Union

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents.output_parsers import \
    ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description
from langchain_openai import ChatOpenAI

from callbacks import AgentCallBackHandler

load_dotenv()


def main():
    print("Hello from langchain-course!")


@tool
def get_text_length(text: str) -> str:
    """Returns the length of text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n'").replace(
        '"', ""
    )  # stripping away non-alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    # print(f"tools: {tools}")
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Could not find the tool with name: {tool_name}")


def format_log_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    # print(f"format_log_str intermediate_steps: {intermediate_steps=}")
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


if __name__ == "__main__":
    main()
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    intermediate_steps = []
    # final_prompt = prompt.format(input='What is the length of text: "Madhu Vadde"')
    # print(final_prompt)
    llm = ChatOpenAI(
        temperature=0,
        model="openai/gpt-4o",
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        stop=["\Observation"],
        callbacks=[AgentCallBackHandler()],
    )
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "what is the text length of 'HELLO' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        # print(agent_step)
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            # print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))
            # print(f"{intermediate_steps=}")

    if isinstance(agent_step, AgentFinish):
        print(f"{agent_step.return_values=}")


# Removed {agent_scratchpad} on line 43 to discuss later in the course
