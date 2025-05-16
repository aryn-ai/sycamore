from datetime import datetime
import logging
from typing import Any, Optional, Union
import asyncio
import random
import time

from PIL import Image

from sycamore.llms.config import AnthropicModels
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts import RenderedPrompt
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import base64_data
from sycamore.utils.import_utils import requires_modules

DEFAULT_MAX_TOKENS = 1000
INITIAL_BACKOFF = 1
BATCH_POLL_INTERVAL = 10


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


def get_generate_kwargs(prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
    kwargs = {
        "temperature": 0,
        **(llm_kwargs or {}),
    }
    kwargs["max_tokens"] = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

    # Anthropic models require _exactly_ alternation between "user" and "assistant"
    # roles, so we break the messages into groups of consecutive user/assistant
    # messages, treating "system" as "user". Then crunch each group down to a single
    # message to ensure alternation.
    message_groups = []  # type: ignore
    last_role = None

    for m in prompt.messages:
        r = m.role
        if r == "system":
            r = "user"
        if r != last_role:
            message_groups.append([])
        message_groups[-1].append(m)
        last_role = r

    messages = []
    for group in message_groups:
        role = group[0].role
        if role == "system":
            role = "user"
        content = "\n".join(m.content for m in group)
        if any(m.images is not None for m in group):
            images = [im for m in group for im in m.images]
            contents = [{"type": "text", "text": content}]
            for im in images:
                contents.append(
                    {  # type: ignore
                        "type": "image",
                        "source": {  # type: ignore
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data(im),
                        },
                    }
                )
            messages.append({"role": role, "content": contents})
        else:
            messages.append({"role": role, "content": content})

    kwargs["messages"] = messages
    return kwargs


def format_image(image: Image.Image) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_data(image)},
    }


def anthropic_deserializer(kwargs):
    return Anthropic(**kwargs)


class Anthropic(LLM):
    """This is an LLM implementation that uses the AWS Claude API to generate text.

    Args:
        model_name: The name of the Claude model to use.
        cache: A cache object to use for caching results.
    """

    @requires_modules("anthropic", extra="anthropic")
    def __init__(
        self,
        model_name: Union[AnthropicModels, str],
        default_mode: LLMMode = LLMMode.ASYNC,
        cache: Optional[Cache] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):

        # We import this here so we can share utility code with the Bedrock
        # LLM implementation without requiring an Anthropic dependency.
        from anthropic import Anthropic as AnthropicClient
        from anthropic import AsyncAnthropic as AsyncAnthropicClient

        self.model_name = model_name

        if isinstance(model_name, AnthropicModels):
            self.model: AnthropicModels = model_name
        elif isinstance(model_name, str):
            model = AnthropicModels.from_name(name=model_name)
            if model is None:
                raise ValueError(f"Invalid model name: {model_name}")
            self.model = model

        self._client = AnthropicClient()
        self._async_client = AsyncAnthropicClient()
        super().__init__(self.model.value, default_mode, cache, default_llm_kwargs=default_llm_kwargs)

    def __reduce__(self):
        kwargs = {
            "model_name": self.model_name,
            "cache": self._cache,
            "default_mode": self._default_mode,
            "default_llm_kwargs": self._default_llm_kwargs,
        }
        return anthropic_deserializer, (kwargs,)

    def default_mode(self) -> LLMMode:
        if self._default_mode is not None:
            return self._default_mode
        return LLMMode.ASYNC

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        return format_image(image)

    def _metadata_from_response(self, kwargs, response, starttime) -> dict:
        wall_latency = datetime.now() - starttime
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        output = response.content[0].text

        ret = {
            "output": output,
            "wall_latency": wall_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
        return ret

    def generate_metadata(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)

        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            return ret

        kwargs = get_generate_kwargs(prompt, llm_kwargs)
        start = datetime.now()

        response = self._client.messages.create(model=self.model.value, **kwargs)
        ret = self._metadata_from_response(kwargs, response, start)
        logging.debug(f"Generated response from Anthropic model: {ret}")

        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt=prompt, llm_kwargs=llm_kwargs)
        return d["output"]

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        from anthropic import RateLimitError, APIConnectionError

        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)

        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            return ret["output"]

        kwargs = get_generate_kwargs(prompt, llm_kwargs)
        start = datetime.now()
        done = False
        retries = 0
        response = None
        while not done:
            try:
                response = await self._async_client.messages.create(model=self.model.value, **kwargs)
                done = True
            except (RateLimitError, APIConnectionError):
                backoff = INITIAL_BACKOFF * (2**retries)
                jitter = random.uniform(0, 0.1 * backoff)
                await asyncio.sleep(backoff + jitter)
                retries += 1

        ret = self._metadata_from_response(kwargs, response, start)
        logging.debug(f"Generated response from Anthropic model: {ret}")

        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret["output"]

    def generate_batch(self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None) -> list[str]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)

        cache_hits = [self._llm_cache_get(p, llm_kwargs) for p in prompts]

        calls = []
        for p, ch, i in zip(prompts, cache_hits, range(len(prompts))):
            if ch is not None:
                continue
            kwargs = get_generate_kwargs(p, llm_kwargs)
            kwargs["model"] = self.model.value
            kwargs["max_tokens"] = kwargs.get("max_tokens", 1024)
            mparams = MessageCreateParamsNonStreaming(**kwargs)  # type: ignore
            rq = Request(custom_id=str(i), params=mparams)
            calls.append(rq)

        starttime = datetime.now()
        batch = self._client.messages.batches.create(requests=calls)

        while batch.processing_status == "in_progress":
            time.sleep(BATCH_POLL_INTERVAL)
            batch = self._client.messages.batches.retrieve(batch.id)

        results = self._client.messages.batches.results(batch.id)
        for rs, call in zip(results, calls):
            if rs.result.type != "succeeded":
                raise ValueError(f"Call failed: {rs}")
            id = int(rs.custom_id)
            in_kwargs = get_generate_kwargs(prompts[id], llm_kwargs)
            ret = self._metadata_from_response(in_kwargs, rs.result.message, starttime)
            cache_hits[id] = ret
            self._llm_cache_set(prompts[id], llm_kwargs, ret)

        return [ch["output"] for ch in cache_hits]
