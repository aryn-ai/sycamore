from abc import ABC, abstractmethod
import pickle
from PIL import Image
from typing import Any, Optional
from sycamore.utils.cache import Cache
from sycamore.utils.thread_local import ThreadLocalAccess, ADD_METADATA_TO_OUTPUT
from sycamore.data.metadata import add_metadata


class LLM(ABC):
    """Abstract representation of an LLM instance. and should be subclassed to implement specific LLM providers."""

    def __init__(self, model_name, cache: Optional[Cache] = None):
        self._model_name = model_name
        self._cache = cache

    @abstractmethod
    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters."""
        pass

    @abstractmethod
    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        pass

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        """Returns a dictionary containing the specified image suitable for use in an LLM message."""
        raise NotImplementedError("This LLM does not support images.")

    async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters asynchronously."""
        raise NotImplementedError("This LLM does not support asynchronous generation.")

    def __str__(self):
        return f"{self.__class__.__name__}({self._model_name})"

    def _llm_cache_key(self, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        """Return a cache key for the given prompt and LLM parameters."""
        assert self._cache
        combined = {"prompt_kwargs": prompt_kwargs, "llm_kwargs": llm_kwargs, "model_name": self._model_name}
        data = pickle.dumps(combined)
        return self._cache.get_hash_context(data).hexdigest()

    def _use_caching(self, llm_kwargs: Optional[dict]):
        if not self._cache:
            return False
        if llm_kwargs is None:
            return True
        # Only cache when temperature setting is zero.
        return llm_kwargs.get("temperature", 0) == 0

    def _llm_cache_get(self, prompt_kwargs: dict, llm_kwargs: Optional[dict]) -> Any:
        """Get a cached result for the given prompt and LLM parameters. Returns the cached
        result if found, or otherwise None."""
        if not self._use_caching(llm_kwargs):
            return None
        assert self._cache is not None, "make mypy happy"

        key = self._llm_cache_key(prompt_kwargs, llm_kwargs)
        hit = self._cache.get(key)
        if hit:
            assert (
                len(hit) == 4
                and hit.get("prompt_kwargs") == prompt_kwargs
                and hit.get("llm_kwargs") == llm_kwargs
                and hit.get("model_name") == self._model_name
                and "result" in hit
            ), f"""
            Found LLM cache content mismatch:
            key={key}
            prompt_kwargs={prompt_kwargs}, cached={hit.get("prompt_kwargs")}
            llm_kwargs={llm_kwargs}, cached={hit.get("llm_kwargs")}
            model_name={self._model_name}, cached={hit.get("model_name")}
            Complete hit: {hit}"""
            return hit.get("result")
        return None

    def _llm_cache_set(self, prompt_kwargs: dict, llm_kwargs: Optional[dict], result: Any) -> None:
        """Set a cached result for the given key."""
        if not self._use_caching(llm_kwargs):
            return
        assert self._cache is not None, "make mypy happy"

        key = self._llm_cache_key(prompt_kwargs, llm_kwargs)
        self._cache.set(
            key,
            {
                "prompt_kwargs": prompt_kwargs,
                "llm_kwargs": llm_kwargs,
                "model_name": self._model_name,
                "result": result,
            },
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

    def __init__(self, *, return_value="trivial", cache: Optional[Cache] = None):
        super().__init__("trivial", cache=cache)
        self._return_value = return_value

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        return self._return_value

    def is_chat_mode(self) -> bool:
        return False
