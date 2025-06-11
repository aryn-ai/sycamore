import datetime
import logging
from typing import Any, Optional, Union
import os
import io

from sycamore.llms.config import GeminiModel, GeminiModels
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.utils.cache import Cache
from sycamore.utils.import_utils import requires_modules

DEFAULT_MAX_TOKENS = 1024


logger = logging.getLogger(__name__)


def gemini_deserializer(kwargs):
    return Gemini(**kwargs)


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
        default_mode: LLMMode = LLMMode.ASYNC,
        cache: Optional[Cache] = None,
        api_key: Optional[str] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        from google.genai import Client

        self.model_name = model_name

        if isinstance(model_name, GeminiModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = GeminiModel(name=model_name)
        api_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        self._client = Client(api_key=api_key)
        super().__init__(self.model.name, default_mode, cache, default_llm_kwargs=default_llm_kwargs)

    def __reduce__(self):
        kwargs = {
            "model_name": self.model_name,
            "cache": self._cache,
            "default_mode": self._default_mode,
            "default_llm_kwargs": self._default_llm_kwargs,
        }
        return gemini_deserializer, (kwargs,)

    def default_mode(self) -> LLMMode:
        if self._default_mode is not None:
            return self._default_mode
        return LLMMode.ASYNC

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def get_generate_kwargs(
        self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model_name_override: Optional[str] = None
    ) -> dict:
        # model_name_override is not directly used here but available for future use if needed.
        # The actual model name for API calls is determined by the calling methods (generate_metadata, generate_async).
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
                    assert content.parts is not None  # mypy
                    content.parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            content_list.append(content)
        kwargs["config"] = None
        if thinking_budget := config.pop("thinking_budget", None):
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
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
        from google.genai.types import FinishReason

        reason = response.candidates[0].finish_reason
        if reason != FinishReason.STOP:
            logger.warning(f"Gemini model stopped for unexpected reason {reason}. Full response:\n{response}")
        ret = {
            "output": output,
            "wall_latency": wall_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
        return ret

    def generate_metadata(
        self, *, prompt: RenderedPrompt, model_name: Optional[str] = None, llm_kwargs: Optional[dict] = None
    ) -> dict:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        current_model_name = model_name or self.model.name

        ret = self._llm_cache_get(prompt, current_model_name, llm_kwargs)
        if isinstance(ret, dict):
            return ret
        assert ret is None

        kwargs = self.get_generate_kwargs(prompt, llm_kwargs, model_name_override=current_model_name)

        start = datetime.datetime.now()
        response = self._client.models.generate_content(
            model=current_model_name, contents=kwargs["content"], config=kwargs["config"]
        )
        ret = self._metadata_from_response(kwargs, response, start)
        self._llm_cache_set(prompt, current_model_name, llm_kwargs, ret)
        return ret

    def generate(
        self, *, prompt: RenderedPrompt, model_name: Optional[str] = None, llm_kwargs: Optional[dict] = None
    ) -> str:
        d = self.generate_metadata(prompt=prompt, model_name=model_name, llm_kwargs=llm_kwargs)
        return d["output"]

    async def generate_async(
        self, *, prompt: RenderedPrompt, model_name: Optional[str] = None, llm_kwargs: Optional[dict] = None
    ) -> str:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        current_model_name = model_name or self.model.name

        ret = self._llm_cache_get(prompt, current_model_name, llm_kwargs)
        if isinstance(ret, dict):
            # Assuming the cached 'ret' is the dictionary from generate_metadata
            return ret["output"]
        assert ret is None # If not a dict, it must be None, meaning cache miss for the full metadata object

        kwargs = self.get_generate_kwargs(prompt, llm_kwargs, model_name_override=current_model_name)

        start = datetime.datetime.now()
        response = await self._client.aio.models.generate_content(
            model=current_model_name, contents=kwargs["content"], config=kwargs["config"]
        )
        # The result from _metadata_from_response is a dict, which is what we'd cache.
        # For generate_async, we only return the text part.
        metadata_result = self._metadata_from_response(kwargs, response, start)
        self._llm_cache_set(prompt, current_model_name, llm_kwargs, metadata_result)
        return metadata_result["output"]
