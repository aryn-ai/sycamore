from typing import Callable, Dict

from sycamore.llms.llms import LLM
from sycamore.llms.openai import OpenAI, OpenAIClientType, OpenAIModels, OpenAIClientParameters, OpenAIClientWrapper
from sycamore.llms.bedrock import Bedrock, BedrockModels

# Register the model constructors.
MODELS: Dict[str, Callable[..., LLM]] = {}
MODELS.update(
    {f"openai.{model.value.name}": lambda **kwargs: OpenAI(model.value.name, **kwargs) for model in OpenAIModels}
)
MODELS.update(
    {f"bedrock.{model.value.name}": lambda **kwargs: Bedrock(model.value.name, **kwargs) for model in BedrockModels}
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
]
