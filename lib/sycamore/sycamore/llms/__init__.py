from typing import Callable, Dict

from sycamore.llms.llms import LLM
from sycamore.llms.openai import OpenAI, OpenAIClientType, OpenAIModels, OpenAIClientParameters, OpenAIClientWrapper
from sycamore.llms.bedrock import Bedrock, BedrockModels
from sycamore.llms.anthropic import Anthropic, AnthropicModels

# Register the model constructors.
MODELS: Dict[str, Callable[..., LLM]] = {}
MODELS.update(
    {f"openai.{model.value.name}": lambda **kwargs: OpenAI(model.value.name, **kwargs) for model in OpenAIModels}
)
MODELS.update(
    {f"bedrock.{model.value.name}": lambda **kwargs: Bedrock(model.value.name, **kwargs) for model in BedrockModels}
)
MODELS.update(
    {f"anthropic.{model.value}": lambda **kwargs: Anthropic(model.value, **kwargs) for model in AnthropicModels}
)


def get_llm(model_name: str) -> Callable[..., LLM]:
    """Returns a function that instantiates the given model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}")
    return MODELS[model_name]


__all__ = [
    "MODELS",
    "get_llm",
    "LLM",
    "OpenAI",
    "OpenAIClientType",
    "OpenAIModels",
    "OpenAIClientParameters",
    "OpenAIClientWrapper",
    "Bedrock",
    "BedrockModels",
    "Anthropic",
    "AnthropicModels",
]
