from typing import Callable, Dict

from sycamore.llms.llms import LLM
from sycamore.llms.config import AnthropicModels, BedrockModels, GeminiModels, OpenAIModels


def Anthropic(name, **kwargs):
    from sycamore.llms.anthropic import Anthropic as AnthropicReal

    return AnthropicReal(name, **kwargs)


def Bedrock(name, **kwargs):
    from sycamore.llms.bedrock import Bedrock as BedrockReal

    return BedrockReal(name, **kwargs)


def Gemini(name, **kwargs):
    from sycamore.llms.gemini import Gemini as GeminiReal

    return GeminiReal(name, **kwargs)


def OpenAI(name, **kwargs):
    from sycamore.llms.openai import OpenAI as OpenAIReal

    return OpenAIReal(name, **kwargs)


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
MODELS.update({f"gemini.{model.value}": lambda **kwargs: Gemini(model.value.name, **kwargs) for model in GeminiModels})


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
    "OpenAI",
    "OpenAIModels",
    #   "OpenAIClientType", # sycamore.llms.openai
    #   "OpenAIClientParameters", # sycamore.llms.openai
    #   "OpenAIClientWrapper", # sycamore.llms.openai
    "Bedrock",
    "BedrockModels",
    "Anthropic",
    "AnthropicModels",
    "Gemini",
    "GeminiModels",
]
