from dataclasses import dataclass
from enum import Enum
import boto3
import json
from typing import Dict, List, Optional, Union


from sycamore.llms.llms import LLM
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
        if isinstance(model_name, BedrockModels):
            self.model = model_name.value
        elif isinstance(model_name, str):
            self.model = BedrockModel(name=model_name)

        self._client = boto3.client(service_name="bedrock-runtime")
        super().__init__(self.model.name, cache)

    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        return True

    def _rewrite_system_messages(self, messages: Optional[List[Dict]]) -> Optional[List[Dict]]:
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

    def _get_generate_kwargs(self, prompt_kwargs: Dict, llm_kwargs: Optional[Dict] = None) -> Dict:
        kwargs = {
            "temperature": 0,
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
            if self._model_name.startswith("anthropic."):
                kwargs["messages"] = self._rewrite_system_messages(kwargs["messages"])
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
            body=body, modelId=self.model.name, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        ret = response_body.get("content", {})[0].get("text", "")
        value = {
            "result": ret,
            "prompt_kwargs": prompt_kwargs,
            "llm_kwargs": llm_kwargs,
            "model_name": self.model.name,
        }
        self._cache_set(key, value)
        return ret
