from abc import ABC, abstractmethod
import pickle
from typing import Optional, Tuple

from sycamore.utils.cache import Cache


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

    async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters asynchronously."""
        raise NotImplementedError("This LLM does not support asynchronous generation.")

    def __str__(self):
        return f"{self.__class__.__name__}({self._model_name})"

    def _get_cache_key(self, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        """Return a cache key for the given prompt and LLM parameters."""
        assert self._cache
        combined = {"prompt_kwargs": prompt_kwargs, "llm_kwargs": llm_kwargs, "model_name": self._model_name}
        data = pickle.dumps(combined)
        return self._cache.get_hash_context(data).hexdigest()

    def _cache_get(self, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """Get a cached result for the given prompt and LLM parameters. Returns the cache key
        and the cached result if found, otherwise returns None for both."""
        if (llm_kwargs or {}).get("temperature", 0) != 0 or not self._cache:
            # Never cache when temperature setting is nonzero.
            return (None, None)

        key = self._get_cache_key(prompt_kwargs, llm_kwargs)
        hit = self._cache.get(key)
        if hit:
            assert (
                hit.get("prompt_kwargs") == prompt_kwargs
                and hit.get("llm_kwargs") == llm_kwargs
                and hit.get("model_name") == self._model_name
            ), f"""
            Found LLM cache content mismatch:
            key={key}
            prompt_kwargs={prompt_kwargs}, cached={hit.get("prompt_kwargs")}
            llm_kwargs={llm_kwargs}, cached={hit.get("llm_kwargs")}
            model_name={self._model_name}, cached={hit.get("model_name")}"""
            return (key, hit.get("result"))
        return (key, None)

    def _cache_set(self, key, result):
        """Set a cached result for the given key."""
        if key is None or not self._cache:
            return
        self._cache.set(key, result)


class FakeLLM(LLM):
    """Useful for tests where the fake LLM needs to run in a ray function because mocks are not serializable"""

    def __init__(self, *, return_value="trivial"):
        super().__init__("trivial")
        self._return_value = return_value

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        return self._return_value

    def is_chat_mode(self) -> bool:
        return False
