from dataclasses import dataclass
import datetime
from enum import Enum
import boto3
import json
from typing import Any, Optional, Union

from PIL import Image

from sycamore.llms.llms import LLM
from sycamore.llms.anthropic import format_image, get_generate_kwargs
from sycamore.utils.cache import Cache

DEFAULT_MAX_TOKENS = 1000
DEFAULT_ANTHROPIC_VERSION = "bedrock-2023-05-31"


@dataclass
class BedrockModel:
    name: str
    is_chat: bool = False


class BedrockModels(Enum):
    """Represents available Bedrock models."""

    # Note that the models available on a given Bedrock account may vary.
    CLAUDE_3_HAIKU = BedrockModel(name="anthropic.claude-3-haiku-20240307-v1:0", is_chat=True)
    CLAUDE_3_SONNET = BedrockModel(name="anthropic.claude-3-sonnet-20240229-v1:0", is_chat=True)
    CLAUDE_3_OPUS = BedrockModel(name="anthropic.claude-3-opus-20240229-v1:0", is_chat=True)
    CLAUDE_3_5_SONNET = BedrockModel(name="anthropic.claude-3-5-sonnet-20240620-v1:0", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


class Bedrock(LLM):
    """This is an LLM implementation that uses the AWS Bedrock API to generate text.

    Args:
        model_name: The name of the Bedrock model to use.
        cache: A cache object to use for caching results.
    """

    def __init__(
        self,
        model_name: Union[BedrockModels, str],
        cache: Optional[Cache] = None,
    ):
        self.model_name = model_name

        if isinstance(model_name, BedrockModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = BedrockModel(name=model_name)

        self._client = boto3.client(service_name="bedrock-runtime")
        super().__init__(self.model.name, cache)

    def __reduce__(self):
        def deserializer(kwargs):
            return Bedrock(**kwargs)

        kwargs = {"model_name": self.model_name, "cache": self._cache}
        return deserializer, (kwargs,)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        if self.model.name.startswith("anthropic."):
            return format_image(image)
        raise NotImplementedError("Images not supported for non-Anthropic Bedrock models.")

    def generate_metadata(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> dict:
        key, ret = self._cache_get(prompt_kwargs, llm_kwargs)
        if isinstance(ret, dict):
            print(f"cache return {ret}")
            return ret

        kwargs = get_generate_kwargs(prompt_kwargs, llm_kwargs)
        if self._model_name.startswith("anthropic."):
            anthropic_version = (
                DEFAULT_ANTHROPIC_VERSION
                if llm_kwargs is None
                else llm_kwargs.get("anthropic_version", DEFAULT_ANTHROPIC_VERSION)
            )
            kwargs["anthropic_version"] = anthropic_version

        body = json.dumps(kwargs)
        start = datetime.datetime.now()
        response = self._client.invoke_model(
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
        value = {
            "result": ret,
            "prompt_kwargs": prompt_kwargs,
            "llm_kwargs": llm_kwargs,
            "model_name": self.model.name,
        }
        self._cache_set(key, value)
        return ret

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt_kwargs=prompt_kwargs, llm_kwargs=llm_kwargs)
        return d["output"]
