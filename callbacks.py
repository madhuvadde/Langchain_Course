from typing import Any

from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult


class AgentCallBackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        """Run when LLM Starting"""
        print(f"****prompt to LLM was:***\n {prompts[0]}")
        print("*************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run When LLM ends running"""
        print(f"****LLM Response:***\n {response.generations[0][0].text}")
        print("*************")
