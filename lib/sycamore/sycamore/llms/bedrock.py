import datetime
import json
from typing import Any, Optional, Union

from PIL import Image

from sycamore.llms.config import BedrockModel, BedrockModels
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.anthropic import format_image, get_generate_kwargs
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.utils.cache import Cache

DEFAULT_MAX_TOKENS = 1000
DEFAULT_ANTHROPIC_VERSION = "bedrock-2023-05-31"


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
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        import boto3

        self.model_name = model_name

        if isinstance(model_name, BedrockModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = BedrockModel(name=model_name)

        self._client = boto3.client(service_name="bedrock-runtime")
        super().__init__(self.model.name, default_mode=LLMMode.SYNC, cache=cache, default_llm_kwargs=default_llm_kwargs)

    def __reduce__(self):
        def deserializer(kwargs):
            return Bedrock(**kwargs)

        kwargs = {"model_name": self.model_name, "cache": self._cache, "default_llm_kwargs": self._default_llm_kwargs}
        return deserializer, (kwargs,)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        if self.model.name.startswith("anthropic."):
            return format_image(image)
        raise NotImplementedError("Images not supported for non-Anthropic Bedrock models.")

    def generate_metadata(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)

        ret = self._llm_cache_get(prompt, llm_kwargs)
        if isinstance(ret, dict):
            print(f"cache return {ret}")
            return ret
        assert ret is None

        kwargs = get_generate_kwargs(prompt, llm_kwargs)
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
        self.add_llm_metadata(kwargs, output, wall_latency, in_tokens, out_tokens)
        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        d = self.generate_metadata(prompt=prompt, llm_kwargs=llm_kwargs)
        return d["output"]
