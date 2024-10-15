import boto3
import json
from sycamore.utils.cache import Cache
from typing import Dict, Optional


from sycamore.llms.llms import LLM

DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
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
        model_name: str = DEFAULT_BEDROCK_MODEL,
        cache: Optional[Cache] = None,
    ):
        self._client = boto3.client(service_name="bedrock-runtime")
        super().__init__(model_name, cache)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def _get_generate_kwargs(self, prompt_kwargs: Dict, llm_kwargs: Optional[Dict] = None) -> Dict:
        kwargs = {
            **(llm_kwargs or {}),
        }
        if self._model_name.startswith("anthropic."):
            kwargs["anthropic_version"] = kwargs.get("anthropic_version", DEFAULT_ANTHROPIC_VERSION)
            kwargs["max_tokens"] = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        if "prompt" in prompt_kwargs:
            prompt = prompt_kwargs.get("prompt")
            kwargs.update({"messages": [{"role": "user", "content": f"{prompt}"}]})
        elif "messages" in prompt_kwargs:
            kwargs.update({"messages": prompt_kwargs["messages"]})
        else:
            raise ValueError("Either prompt or messages must be present in prompt_kwargs.")
        return kwargs

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        key, ret = self._cache_get(prompt_kwargs, llm_kwargs)
        if ret is not None:
            return ret

        kwargs = self._get_generate_kwargs(prompt_kwargs, llm_kwargs)
        body = json.dumps(kwargs)
        response = self._client.invoke_model(
            body=body, modelId=self._model_name, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        ret = response_body.get("content", {})[0].get("text", "")
        value = {
            "result": ret,
            "prompt_kwargs": prompt_kwargs,
            "llm_kwargs": llm_kwargs,
            "model_name": self._model_name,
        }
        self._cache_set(key, value)
        return ret
