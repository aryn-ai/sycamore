from dataclasses import dataclass
import datetime
from enum import Enum
import json
from typing import Any, Optional, Union

from PIL import Image
from google.genai import Client

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
    """Represents available Gemini models."""

    # Note that the models available on a given Gemini account may vary.
    GEMINI_2_FLASH = GeminiModel(name="gemini-2.0-flash-exp", is_chat=True)
    GEMINI_2_FLASH_LITE = GeminiModel(name="gemini-2.0-flash-lite-preview-02-05", is_chat=True)
    GEMINI_2_FLASH_THINKING = GeminiModel(name="gemini-2.0-flash-thinking-exp", is_chat=True)
    GEMINI_2_PRO = GeminiModel(name="gemini-2.0-pro-exp", is_chat=True)

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

    @requires_modules("google-genai")
    def __init__(
        self,
        model_name: Union[GeminiModels, str],
        cache: Optional[Cache] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name

        if isinstance(model_name, GeminiModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = GeminiModel(name=model_name)
        if api_key is not None:
            self._client = Client(api_key=api_key)
        self._client = Client()
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

    def generate_metadata(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            print(f"cache return {ret}")
            return ret
        assert ret is None

        kwargs = self.get_generate_kwargs(prompt, llm_kwargs)

        body = json.dumps(kwargs)
        start = datetime.datetime.now()
        response = self._client.models.generate_content(
            model=self.model.name,
        )

        self._client.invoke_model(
            body=body, modelId=self.model.name, accept="application/json", contentType="application/json"
        )
        wall_latency = datetime.datetime.now() - start
        md = response["ResponseMetadata"]
        assert md["HTTPStatusCode"] == 200, f"Request failed {md['HTTPStatusCode']}"
        hdrs = md["HTTPHeaders"]
        server_latency = datetime.timedelta(milliseconds=int(hdrs["x-amzn-bedrock-invocation-latency"]))
        in_tokens = int(hdrs["x-amzn-bedrock-input-token-count"])
        out_tokens = int(hdrs["x-amzn-bedrock-output-token-count"])
        response_body = json.loads(response.get("body").read())
        output = response_body.get("content", {})[0].get("text", "")
        ret = {
            "output": output,
            "wall_latency": wall_latency,
            "server_latency": server_latency,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
        }
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt=prompt, llm_kwargs=llm_kwargs)
        return d["output"]
