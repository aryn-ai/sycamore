from datetime import datetime
import logging
from typing import Any, Optional, Union

from PIL import Image

from sycamore.llms.config import OllamaModel, OllamaModels, LLMModel
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts import RenderedPrompt
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import base64_data
from sycamore.utils.import_utils import requires_modules

logger = logging.getLogger(__name__)


def get_generate_kwargs(prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
    kwargs = {
        "options": {"temperature": 0},
        **(llm_kwargs or {}),
    }

    messages = []
    for m in prompt.messages:
        message: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.images:
            message["images"] = [base64_data(im) for im in m.images]
        messages.append(message)

    kwargs["messages"] = messages
    return kwargs


def ollama_deserializer(kwargs):
    return Ollama(**kwargs)


class Ollama(LLM):
    """An LLM interface to a local Ollama server.

    Unlike the other providers, Ollama's model catalog is whatever the user has locally
    `ollama pull`ed, rather than a fixed list. Passing an unrecognized model name string will
    not raise - it will construct an ad hoc model. See OllamaModels for a handful of common
    presets.

    Args:
        model_name: The name of the Ollama model to use. Can be an OllamaModels, an
            OllamaModel, or a string naming any locally-pulled Ollama model.
        host: The base URL of the Ollama server, e.g. "http://localhost:11434". If not
            provided, falls back to the OLLAMA_HOST environment variable, and then to
            Ollama's default of http://127.0.0.1:11434.
        cache: An instance of Cache to use for caching responses. If not provided, no
            caching will be used.
        default_mode: Default execution mode for the llm.
        default_llm_kwargs: Default kwargs merged into every call. Use this to pass
            Ollama-native `options` (e.g. num_ctx, mirostat) or a `format` for structured output.
        client_args: Additional keyword arguments to pass to the Ollama client constructor.
    """

    @requires_modules("ollama", extra="ollama")
    def __init__(
        self,
        model_name: Union[OllamaModels, OllamaModel, str],
        host: Optional[str] = None,
        default_mode: LLMMode = LLMMode.ASYNC,
        cache: Optional[Cache] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
        client_args: Optional[dict[str, Any]] = None,
    ):
        from ollama import Client as OllamaClient
        from ollama import AsyncClient as AsyncOllamaClient

        self.model_name = model_name

        if isinstance(model_name, OllamaModels):
            self.model = model_name.value
        elif isinstance(model_name, OllamaModel):
            self.model = model_name
        elif isinstance(model_name, str):
            self.model = OllamaModels.from_name(model_name)
        else:
            raise TypeError("model_name must be an instance of str, OllamaModel, or OllamaModels")

        self.host = host
        self._client_args = client_args or {}
        self._client = OllamaClient(host=host, **self._client_args)
        self._async_client = AsyncOllamaClient(host=host, **self._client_args)
        super().__init__(self.model.name, default_mode, cache, default_llm_kwargs=default_llm_kwargs)

    def __reduce__(self):
        kwargs = {
            "model_name": self.model_name,
            "host": self.host,
            "cache": self._cache,
            "default_mode": self._default_mode,
            "default_llm_kwargs": self._default_llm_kwargs,
            "client_args": self._client_args,
        }
        return ollama_deserializer, (kwargs,)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        return {"images": [base64_data(image)]}

    def _metadata_from_response(self, model: str, kwargs, response, starttime) -> dict:
        wall_latency = datetime.now() - starttime
        in_tokens = response.get("prompt_eval_count") or 0
        out_tokens = response.get("eval_count") or 0
        output = response["message"]["content"]

        ret = {
            "model": model,
            "output": output,
            "wall_latency": wall_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens, model=model)
        return ret

    def generate_metadata(
        self, *, prompt: RenderedPrompt, model: Optional[LLMModel] = None, llm_kwargs: Optional[dict] = None
    ) -> dict:
        assert model is None or isinstance(
            model, OllamaModel
        ), f"model must be an OllamaModel, got {type(model)} from {model=}"

        if model is not None and model != self.model:
            logger.info(f"Generating response using {model=} instead of {self.model=}")
        model_name = model.name if model else self.model.name
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)

        ret = self._llm_cache_get(prompt, llm_kwargs, model=model_name)
        if isinstance(ret, dict):
            return ret

        kwargs = get_generate_kwargs(prompt, llm_kwargs)
        start = datetime.now()

        response = self._client.chat(model=model_name, **kwargs)
        ret = self._metadata_from_response(model_name, kwargs, response, start)
        logging.debug(f"Generated response from Ollama model: {ret}")

        self._llm_cache_set(prompt, llm_kwargs, ret, model=model_name)
        return ret

    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        d = self.generate_metadata(prompt=prompt, model=model, llm_kwargs=llm_kwargs)
        return d["output"]

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        model_name: str = model.name if model else self.model.name
        if self.model.name != model_name:
            logging.info(f"Overriding Ollama model from {self.model.name} to {model_name}")
        ret = self._llm_cache_get(prompt, llm_kwargs, model=model_name)
        if isinstance(ret, dict):
            return ret["output"]

        kwargs = get_generate_kwargs(prompt, llm_kwargs)
        start = datetime.now()
        response = await self._async_client.chat(model=model_name, **kwargs)
        ret = self._metadata_from_response(model_name, kwargs, response, start)
        logging.debug(f"Generated response from Ollama model: {ret}")

        self._llm_cache_set(prompt, llm_kwargs, ret, model=model_name)
        return ret["output"]
