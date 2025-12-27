REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS = """ 
Answer the following questions as best you can using English. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\n
Observation: the result of the action\n
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question formatted according to the format_instructions: {format_instructions}\n
\n
Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
