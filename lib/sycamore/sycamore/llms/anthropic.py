from datetime import datetime
from enum import Enum
import logging
from typing import Any, Optional, Union

from PIL import Image

from sycamore.llms.llms import LLM
from sycamore.llms.prompts.default_prompts import SimplePrompt
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import base64_data
from sycamore.utils.import_utils import requires_modules

DEFAULT_MAX_TOKENS = 1000


class AnthropicModels(Enum):
    """Represents available Claude models."""

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    @classmethod
    def from_name(cls, name: str) -> Optional["AnthropicModels"]:
        for m in iter(cls):
            if m.value == name:
                return m
        return None


def rewrite_system_messages(messages: Optional[list[dict]]) -> Optional[list[dict]]:
    # Anthropic models don't accept messages with "role" set to "system", and
    # requires alternation between "user" and "assistant" roles. So, we rewrite
    # the messages to fold all "system" messages into the "user" role.
    if not messages:
        return messages
    orig_messages = messages.copy()
    cur_system_message = ""
    for i, message in enumerate(orig_messages):
        if message.get("role") == "system":
            cur_system_message += message.get("content", "")
        else:
            if cur_system_message:
                messages[i]["content"] = cur_system_message + "\n" + message.get("content", "")
                cur_system_message = ""
    return [m for m in messages if m.get("role") != "system"]


def get_generate_kwargs(prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> dict:
    kwargs = {
        "temperature": 0,
        **(llm_kwargs or {}),
    }

    kwargs["max_tokens"] = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

    if "prompt" in prompt_kwargs:
        prompt = prompt_kwargs.get("prompt")

        if isinstance(prompt, SimplePrompt):
            kwargs.update({"messages": prompt.as_messages(prompt_kwargs)})
        else:
            kwargs.update({"messages": [{"role": "user", "content": f"{prompt}"}]})

    elif "messages" in prompt_kwargs:
        kwargs.update({"messages": prompt_kwargs["messages"]})
    else:
        raise ValueError("Either prompt or messages must be present in prompt_kwargs.")

    kwargs["messages"] = rewrite_system_messages(kwargs["messages"])

    return kwargs


def format_image(image: Image.Image) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_data(image)},
    }


class Anthropic(LLM):
    """This is an LLM implementation that uses the AWS Claude API to generate text.

    Args:
        model_name: The name of the Claude model to use.
        cache: A cache object to use for caching results.
    """

    @requires_modules("qdrant_client", extra="anthropic")
    def __init__(
        self,
        model_name: Union[AnthropicModels, str],
        cache: Optional[Cache] = None,
    ):

        # We import this here so we can share utility code with the Bedrock
        # LLM implementation without requiring an Anthropic dependency.
        from anthropic import Anthropic as AnthropicClient

        self.model_name = model_name

        if isinstance(model_name, AnthropicModels):
            self.model: AnthropicModels = model_name
        elif isinstance(model_name, str):
            model = AnthropicModels.from_name(name=model_name)
            if model is None:
                raise ValueError(f"Invalid model name: {model_name}")
            self.model = model

        self._client = AnthropicClient()
        super().__init__(self.model.value, cache)

    def __reduce__(self):
        def deserializer(kwargs):
            return Anthropic(**kwargs)

        kwargs = {"model_name": self.model_name, "cache": self._cache}
        return deserializer, (kwargs,)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        return format_image(image)

    def generate_metadata(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> dict:
        key, ret = self._cache_get(prompt_kwargs, llm_kwargs)
        if isinstance(ret, dict):
            return ret

        kwargs = get_generate_kwargs(prompt_kwargs, llm_kwargs)

        start = datetime.now()

        response = self._client.messages.create(model=self.model.value, **kwargs)

        wall_latency = datetime.now() - start

        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        output = response.content[0].text

        ret = {
            "output": output,
            "wall_latency": wall_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }

        logging.debug(f"Generated response from Anthropic model: {ret}")

        value = {
            "result": ret,
            "prompt_kwargs": prompt_kwargs,
            "llm_kwargs": llm_kwargs,
            "model_name": self.model.value,
        }
        self._cache_set(key, value)
        return ret

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt_kwargs=prompt_kwargs, llm_kwargs=llm_kwargs)
        return d["output"]
