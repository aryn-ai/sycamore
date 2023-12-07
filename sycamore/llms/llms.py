import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type


class OpenAIModels(Enum):
    TEXT_DAVINCI = "text-davinci-003"
    GPT_3_5_TURBO = "gpt-3.5-turbo-1106"


class LLM(ABC):
    def __init__(self, model_name, **kwargs):
        self._model_name = model_name
        self._kwargs = kwargs

    @abstractmethod
    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Any:
        pass

    @property
    def is_chat_mode(self):
        chat_model_pattern = r"^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$"
        return re.match(chat_model_pattern, self._model_name)


@dataclass
class OpenAIClientParameters:
    api_type: str = openai.api_type
    api_key: Optional[str] = openai.api_key
    api_base: str = openai.api_base
    api_version: Optional[str] = openai.api_version

    def is_azure(self):
        return self.api_type in {"azure", "azure_ad", "azuread"}

    def merge(self, model_or_deployment_id: str, **kwargs) -> dict[str, Any]:
        kwargs_dict = asdict(self)
        if self.is_azure():
            kwargs_dict["deployment_id"] = model_or_deployment_id
        else:
            kwargs_dict["model"] = model_or_deployment_id
        kwargs_dict.update(kwargs)
        return kwargs_dict


class OpenAI(LLM):
    def __init__(self, model_name, api_key=None, params: OpenAIClientParameters = OpenAIClientParameters(), **kwargs):
        super().__init__(model_name, **kwargs)

        self._params = params

        if api_key is not None:
            self._params.api_key = api_key

        assert self._params.api_key is not None, (
            "You must provide an API key to "
            "use the LLM. Either pass it in "
            "the constructor or set the "
            "OPENAI_API_KEY environment "
            "variable."
        )

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Any:
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
            "temperature": 0,
            **llm_kwargs,
            **self._kwargs,
        }

        prompt = prompt_kwargs.get("prompt")
        messages = [{"role": "user", "content": f"{prompt}"}]
        completion = openai.ChatCompletion.create(**self._params.merge(self._model_name, messages=messages, **kwargs))
        return completion.choices[0].message

    def _generate_using_guidance(self, prompt_kwargs) -> Any:
        import guidance

        guidance.llm = guidance.llms.OpenAI(**self._params.merge(self._model_name, **self._kwargs))

        guidance_program = guidance(prompt_kwargs.pop("prompt"))
        prediction = guidance_program(**prompt_kwargs)
        return prediction
