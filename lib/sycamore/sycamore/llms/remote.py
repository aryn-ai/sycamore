from typing import Optional, Awaitable

from sycamore.llms import LLM


class RemoteLLM(LLM):
    def __init__(self, model_name: str, endpoint: str):
        super().__init__(model_name)
        self._model_name = model_name
        self._endpoint = endpoint

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        pass

    def is_chat_mode(self) -> bool:
        return False

    async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Awaitable[str]:
        raise ValueError("No implementation for llm futures exists")
