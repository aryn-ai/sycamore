from abc import ABC, abstractmethod
from typing import Optional

from sycamore.utils.cache import Cache


class LLM(ABC):
    """
    Initializes a new LLM instance. This class is abstract and should be subclassed to implement specific LLM providers.
    """

    def __init__(self, model_name, cache: Optional[Cache] = None):
        self._model_name = model_name
        self._cache = cache

    """
    Generates a response from the LLM for the given prompt and LLM parameters.
    """

    @abstractmethod
    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        pass

    """
    Returns True if the LLM is in chat mode, False otherwise.
    """

    @abstractmethod
    def is_chat_mode(self) -> bool:
        pass

    async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        raise ValueError("No implementation for llm futures exists")


class FakeLLM(LLM):
    """Useful for tests where the fake LLM needs to run in a ray function because mocks are not serializable"""

    def __init__(self, *, return_value="trivial"):
        super().__init__("trivial")
        self._return_value = return_value

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        return self._return_value

    def is_chat_mode(self) -> bool:
        return False
