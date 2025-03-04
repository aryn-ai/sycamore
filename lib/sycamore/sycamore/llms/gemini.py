from dataclasses import dataclass
import datetime
from enum import Enum
from typing import Any, Optional, Union
import os
import io

from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.utils.cache import Cache
from sycamore.utils.import_utils import requires_modules

DEFAULT_MAX_TOKENS = 1024


@dataclass
class GeminiModel:
    name: str
    is_chat: bool = False


class GeminiModels(Enum):
    """Represents available Gemini models. More info: https://googleapis.github.io/python-genai/"""

    # Note that the models available on a given Gemini account may vary.
    GEMINI_2_FLASH = GeminiModel(name="gemini-2.0-flash", is_chat=True)
    GEMINI_2_FLASH_LITE = GeminiModel(name="gemini-2.0-flash-lite", is_chat=True)
    GEMINI_2_FLASH_THINKING = GeminiModel(name="gemini-2.0-flash-thinking-exp", is_chat=True)
    GEMINI_2_PRO = GeminiModel(name="gemini-2.0-pro-exp-02-05", is_chat=True)
    GEMINI_1_5_PRO = GeminiModel(name="gemini-1.5-pro", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


class Gemini(LLM):
    """This is an LLM implementation that uses the Google Gemini API to generate text.

    Args:
        model_name: The name of the Gemini model to use.
        cache: A cache object to use for caching results.
    """

    @requires_modules("google.genai", extra="google-genai")
    def __init__(
        self,
        model_name: Union[GeminiModels, str],
        cache: Optional[Cache] = None,
        api_key: Optional[str] = None,
    ):
        from google.genai import Client

        self.model_name = model_name

        if isinstance(model_name, GeminiModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = GeminiModel(name=model_name)
        api_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        self._client = Client(api_key=api_key)
        super().__init__(self.model.name, cache)

    def __reduce__(self):
        def deserializer(kwargs):
            return Gemini(**kwargs)

        kwargs = {"model_name": self.model_name, "cache": self._cache}
        return deserializer, (kwargs,)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def get_generate_kwargs(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        from google.genai import types

        kwargs: dict[str, Any] = {}
        config = {
            "temperature": 0,
            "candidate_count": 1,
            **(llm_kwargs or {}),
        }
        config["max_output_tokens"] = config.get("max_output_tokens", DEFAULT_MAX_TOKENS)
        if prompt.response_format:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = prompt.response_format
        content_list: list[types.Content] = []
        for message in prompt.messages:
            if message.role == "system":
                config["system_instruction"] = message.content
                continue
            role = "model" if message.role == "assistant" else "user"
            content = types.Content(parts=[types.Part.from_text(text=message.content)], role=role)
            if message.images:
                for image in message.images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_bytes = buffered.getvalue()
                    content.parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            content_list.append(content)
        kwargs["config"] = None
        if config:
            kwargs["config"] = types.GenerateContentConfig(**config)
        kwargs["content"] = content_list
        return kwargs

    def _metadata_from_response(self, kwargs, response, starttime) -> dict:
        wall_latency = datetime.datetime.now() - starttime
        md = response.usage_metadata
        in_tokens = int(md.prompt_token_count) if md and md.prompt_token_count else 0
        out_tokens = int(md.candidates_token_count) if md and md.candidates_token_count else 0
        output = " ".join(part.text if part else "" for part in response.candidates[0].content.parts)
        ret = {
            "output": output,
            "wall_latency": wall_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
        return ret

    def generate_metadata(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            return ret
        assert ret is None

        kwargs = self.get_generate_kwargs(prompt, llm_kwargs)

        start = datetime.datetime.now()
        response = self._client.models.generate_content(
            model=self.model.name, contents=kwargs["content"], config=kwargs["config"]
        )
        ret = self._metadata_from_response(kwargs, response, start)
        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt=prompt, llm_kwargs=llm_kwargs)
        return d["output"]

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            return ret["output"]
        assert ret is None

        kwargs = self.get_generate_kwargs(prompt, llm_kwargs)

        start = datetime.datetime.now()
        response = await self._client.aio.models.generate_content(
            model=self.model.name, contents=kwargs["content"], config=kwargs["config"]
        )
        ret = self._metadata_from_response(kwargs, response, start)
        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret["output"]
