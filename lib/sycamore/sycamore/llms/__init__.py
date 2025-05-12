from typing import Callable, Dict

from sycamore.llms.llms import LLM
from sycamore.llms.config import AnthropicModels, BedrockModels, GeminiModels, OpenAIModels


def AnthropicTrampoline(name, **kwargs):
    from sycamore.llms.anthropic import Anthropic

    return Anthropic(name, **kwargs)


def BedrockTrampoline(name, **kwargs):
    from sycamore.llms.bedrock import Bedrock

    return Bedrock(name, **kwargs)


def GeminiTrampoline(name, **kwargs):
    from sycamore.llms.gemini import Gemini

    return Gemini(name, **kwargs)


def OpenAITrampoline(name, **kwargs):
    from sycamore.llms.openai import OpenAI

    return OpenAI(name, **kwargs)


# Register the model constructors.
MODELS: Dict[str, Callable[..., LLM]] = {}
MODELS.update(
    {
        f"openai.{model.value.name}": lambda **kwargs: OpenAITrampoline(model.value.name, **kwargs)
        for model in OpenAIModels
    }
)
MODELS.update(
    {
        f"bedrock.{model.value.name}": lambda **kwargs: BedrockTrampoline(model.value.name, **kwargs)
        for model in BedrockModels
    }
)
MODELS.update(
    {
        f"anthropic.{model.value}": lambda **kwargs: AnthropicTrampoline(model.value, **kwargs)
        for model in AnthropicModels
    }
)
MODELS.update(
    {f"gemini.{model.value}": lambda **kwargs: GeminiTrampoline(model.value.name, **kwargs) for model in GeminiModels}
)


def get_llm(model_name: str) -> Callable[..., LLM]:
    """Returns a function that instantiates the given model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}")
    return MODELS[model_name]


# commented out bits can be removed after 2025-08-01; they are here to help people
# find where things should be imported from
__all__ = [
    "MODELS",
    "get_llm",
    "LLM",
    "OpenAIModels",
    #   "OpenAI", # sycamore.llms.openai
    #   "OpenAIClientType", # sycamore.llms.openai
    #   "OpenAIClientParameters", # sycamore.llms.openai
    #   "OpenAIClientWrapper", # sycamore.llms.openai
    #   "Bedrock", # sycamore.llms.bedrock
    "BedrockModels",
    # "Anthropic", # sycamore.llms.anthropic
    "AnthropicModels",
    # "Gemini", # sycamore.llms.gemini
    "GeminiModels",
]
