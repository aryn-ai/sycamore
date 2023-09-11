import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any

import openai
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type


class OpenAIModels(Enum):
    TEXT_DAVINCI = "text-davinci-003"
    GPT_3_5_TURBO = "gpt-3.5-turbo-0613"


class LLM(ABC):
    def __init__(self, model_name, **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs

    @abstractmethod
    def generate(self, *, prompt_kwargs: Dict, llm_kwargs: Dict) -> Any:
        pass

    def is_chat_mode(self):
        chat_model_pattern = r"^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$"
        return re.match(chat_model_pattern, self._model_name)


class OpenAI(LLM):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, **kwargs)
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)

        assert api_key is not None, (
            "You must provide an API key to "
            "use the LLM. Either pass it in "
            "the constructor or set the "
            "OPENAI_API_KEY environment "
            "variable."
        )
        self._api_key = api_key

    def generate(self, *, prompt_kwargs: Dict, llm_kwargs: Dict = None) -> Any:
        if llm_kwargs is not None:
            return self._generate_using_openai(prompt_kwargs, llm_kwargs)

        else:
            return self._generate_using_guidance(prompt_kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random(min=1, max=2),
        retry=retry_if_exception_type(openai.error.RateLimitError),
    )
    def _generate_using_openai(self, prompt_kwargs, llm_kwargs) -> Any:
        kwargs = {
            "api_key": self._api_key,
            "temperature": 0,
            **llm_kwargs,
            **self._kwargs,
        }

        prompt = prompt_kwargs.get("prompt")
        messages = [{"role": "user", "content": f"{prompt}"}]
        completion = openai.ChatCompletion.create(model=self._model_name, messages=messages, **kwargs)
        return completion.choices[0].message

    def _generate_using_guidance(self, prompt_kwargs) -> Any:
        import guidance

        guidance.llm = guidance.llms.OpenAI(model=self._model_name, api_key=self._api_key, **self._kwargs)

        guidance_program = guidance(prompt_kwargs.pop("prompt"))
        prediction = guidance_program(**prompt_kwargs)
        return prediction
