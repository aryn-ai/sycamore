import inspect
from abc import ABC, abstractmethod
import copy
from enum import Enum
import logging
import pickle
import base64
from PIL import Image
from typing import Any, Optional
import pydantic
from sycamore.utils.cache import Cache
from sycamore.utils.thread_local import ThreadLocalAccess, ADD_METADATA_TO_OUTPUT
from sycamore.data.metadata import add_metadata
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage

from sycamore.utils.deprecate import deprecated


class LLMMode(Enum):
    SYNC = 1
    ASYNC = 2
    BATCH = 3


class LLM(ABC):
    """Abstract representation of an LLM instance. and should be subclassed to implement specific LLM providers."""

    def __init__(
        self,
        model_name,
        default_mode: LLMMode,
        cache: Optional[Cache] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._model_name = model_name
        self._cache = cache
        self._default_mode = default_mode
        self._default_llm_kwargs = default_llm_kwargs or {}

    def default_mode(self) -> LLMMode:
        """Returns the default execution mode for the llm"""
        return self._default_mode

    def _merge_llm_kwargs(self, llm_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Merges the default LLM kwargs with any provided LLM kwargs.

        Prefers the passed in values if there is a conflict.
        """
        new_kwargs = copy.copy(self._default_llm_kwargs)
        new_kwargs.update(llm_kwargs or {})
        logging.debug(f"Merging LLM kwargs: {new_kwargs}")
        return new_kwargs

    @abstractmethod
    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters."""
        pass

    @deprecated(version="0.1.31", reason="Use generate, with a RenderedPrompt, instead")
    def generate_old(self, *, prompt_kwargs: dict[str, Any], llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM"""
        from sycamore.llms.prompts.default_prompts import SimplePrompt

        if "prompt" in prompt_kwargs:
            prompt = prompt_kwargs.get("prompt")
            if isinstance(prompt, SimplePrompt):
                prompt = prompt.as_messages()
                for idx, prompt_message in enumerate(prompt):
                    prompt[idx]["content"] = prompt_message["content"].format(**prompt_kwargs)
                rendered = RenderedPrompt(
                    messages=[RenderedMessage(role=m["role"], content=m["content"]) for m in prompt]
                )
            else:
                rendered = RenderedPrompt(messages=[RenderedMessage(role="user", content=f"{prompt}")])
        elif "messages" in prompt_kwargs:
            ms = prompt_kwargs.get("messages", [])
            messages = [RenderedMessage(role=m["role"], content=m["content"]) for m in ms]
            rendered = RenderedPrompt(messages=messages)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be specified in prompt_kwargs")
        return self.generate(prompt=rendered, llm_kwargs=llm_kwargs)

    @abstractmethod
    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        pass

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        """Returns a dictionary containing the specified image suitable for use in an LLM message."""
        raise NotImplementedError("This LLM does not support images.")

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters asynchronously."""
        raise NotImplementedError("This LLM does not support asynchronous generation.")

    @deprecated(version="0.1.31", reason="Use generate_async, with a RenderedPrompt, instead")
    async def generate_async_old(self, *, prompt_kwargs: dict[str, Any], llm_kwargs: Optional[dict] = None) -> str:
        from sycamore.llms.prompts.default_prompts import SimplePrompt

        if "prompt" in prompt_kwargs:
            prompt = prompt_kwargs.get("prompt")
            if isinstance(prompt, SimplePrompt):
                prompt = prompt.as_messages()
                for idx, prompt_message in enumerate(prompt):
                    prompt[idx]["content"] = prompt_message["content"].format(**prompt_kwargs)
                rendered = RenderedPrompt(
                    messages=[RenderedMessage(role=m["role"], content=m["content"]) for m in prompt]
                )
            else:
                rendered = RenderedPrompt(messages=[RenderedMessage(role="user", content=f"{prompt}")])
        elif "messages" in prompt_kwargs:
            ms = prompt_kwargs.get("messages", [])
            messages = [RenderedMessage(role=m["role"], content=m["content"]) for m in ms]
            rendered = RenderedPrompt(messages=messages)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be specified in prompt_kwargs")
        return await self.generate_async(prompt=rendered, llm_kwargs=llm_kwargs)

    def generate_batch(self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None) -> list[str]:
        """Generates a series of responses from the LLM for the given series of prompts. Order is preserved."""
        raise NotImplementedError("This LLM does not support batched generation")

    def __str__(self):
        return f"{self.__class__.__name__}({self._model_name})"

    @staticmethod
    def _pickleable_response_format(prompt: RenderedPrompt) -> Any:
        if inspect.isclass(prompt.response_format) and issubclass(prompt.response_format, pydantic.BaseModel):
            return prompt.response_format.model_json_schema()
        else:
            return prompt.response_format

    def _llm_cache_key(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        """Return a cache key for the given prompt and LLM parameters."""
        assert self._cache
        rf = self._pickleable_response_format(prompt)
        ms = prompt.messages
        combined = {
            "prompt": RenderedPrompt(messages=ms),
            "prompt.response_format": rf,
            "llm_kwargs": llm_kwargs,
            "model_name": self._model_name,
        }
        data = pickle.dumps(combined)
        return self._cache.get_hash_context(data).hexdigest()

    def _use_caching(self, llm_kwargs: Optional[dict]):
        if not self._cache:
            return False
        if llm_kwargs is None:
            return True
        # Only cache when temperature setting is zero.
        return llm_kwargs.get("temperature", 0) == 0

    def _llm_cache_get(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict]) -> Any:
        """Get a cached result for the given prompt and LLM parameters. Returns the cached
        result if found, or otherwise None."""
        if not self._use_caching(llm_kwargs):
            return None
        assert self._cache is not None, "make mypy happy"

        key = self._llm_cache_key(prompt, llm_kwargs)
        hit = self._cache.get(key)
        if hit:
            hit = base64.b64decode(hit)
            hit = pickle.loads(hit)
            assert (
                len(hit) == 5
                and hit.get("prompt") == RenderedPrompt(messages=prompt.messages)
                and hit.get("prompt.response_format") == self._pickleable_response_format(prompt)
                and hit.get("llm_kwargs") == llm_kwargs
                and hit.get("model_name") == self._model_name
                and "result" in hit
            ), f"""
            Found LLM cache content mismatch:
            key={key}
            prompt={prompt}, cached={hit.get("prompt")}
                             cached_response_format={hit.get("prompt.response_format")}
            llm_kwargs={llm_kwargs}, cached={hit.get("llm_kwargs")}
            model_name={self._model_name}, cached={hit.get("model_name")}
            Complete hit: {hit}"""
            return hit.get("result")
        return None

    def _llm_cache_set(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict], result: Any) -> None:
        """Set a cached result for the given key."""
        if not self._use_caching(llm_kwargs):
            return
        assert self._cache is not None, "make mypy happy"

        key = self._llm_cache_key(prompt, llm_kwargs)
        databytes = pickle.dumps(
            {
                "prompt": RenderedPrompt(messages=prompt.messages),
                "prompt.response_format": self._pickleable_response_format(prompt),
                "llm_kwargs": llm_kwargs,
                "model_name": self._model_name,
                "result": result,
            }
        )
        datastr = base64.b64encode(databytes).decode("utf-8")
        self._cache.set(
            key,
            datastr,
        )

    def get_metadata(self, kwargs, response_text, wall_latency, in_tokens, out_tokens) -> dict:
        """Generate metadata for the LLM response."""
        return {
            "model": self._model_name,
            "temperature": kwargs.get("temperature", None),
            "usage": {
                "completion_tokens": in_tokens,
                "prompt_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            },
            "wall_latency": wall_latency,
            "prompt": kwargs.get("prompt") or kwargs.get("messages"),
            "output": response_text,
        }

    def add_llm_metadata(self, kwargs, output, wall_latency, in_tokens, out_tokens):
        tls = ThreadLocalAccess(ADD_METADATA_TO_OUTPUT)
        if tls.present():
            metadata = self.get_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
            add_metadata(**metadata)


class FakeLLM(LLM):
    """Useful for tests where the fake LLM needs to run in a ray function because mocks are not serializable"""

    def __init__(
        self,
        *,
        return_value="trivial",
        cache: Optional[Cache] = None,
        default_mode: LLMMode = LLMMode.SYNC,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__("trivial", cache=cache, default_mode=default_mode, default_llm_kwargs=default_llm_kwargs)
        self._return_value = return_value

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return self._return_value

    def is_chat_mode(self) -> bool:
        return False
