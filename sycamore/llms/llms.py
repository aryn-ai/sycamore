from abc import ABC, abstractmethod
from typing import Any, Optional


class LLM(ABC):
    def __init__(self, model_name):
        self._model_name = model_name

    @abstractmethod
    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Any:
        pass

    @abstractmethod
    def is_chat_mode(self):
        pass
