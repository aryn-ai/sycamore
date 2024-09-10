# ruff: noqa: F401

from sycamore.llms.prompts import default_prompts

from sycamore.llms.prompts.default_prompts import (
    SimplePrompt,
    EntityExtractorZeroShotGuidancePrompt,
    EntityExtractorFewShotGuidancePrompt,
    TextSummarizerGuidancePrompt,
    SchemaZeroShotGuidancePrompt,
    PropertiesZeroShotGuidancePrompt,
    TaskIdentifierZeroShotGuidancePrompt,
    ExtractTablePropertiesTablePrompt,
    ExtractTablePropertiesPrompt,
)
from sycamore.llms.prompts.default_prompts import _deprecated_prompts

prompts = [
    "SimplePrompt",
    "EntityExtractorZeroShotGuidancePrompt",
    "EntityExtractorFewShotGuidancePrompt",
    "TextSummarizerGuidancePrompt",
    "SchemaZeroShotGuidancePrompt",
    "PropertiesZeroShotGuidancePrompt",
    "ExtractTablePropertiesTablePrompt",
    "ExtractTablePropertiesPrompt",
] + list(_deprecated_prompts.keys())

__all__ = prompts


def __getattr__(name):
    if name in _deprecated_prompts:
        return getattr(default_prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
