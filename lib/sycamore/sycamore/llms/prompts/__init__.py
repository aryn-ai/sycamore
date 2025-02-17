# ruff: noqa: F401

from sycamore.llms.prompts import default_prompts

from sycamore.llms.prompts.default_prompts import (
    SimplePrompt,
    EntityExtractorZeroShotJinjaPrompt,
    EntityExtractorFewShotJinjaPrompt,
    TextSummarizerGuidancePrompt,
    SchemaZeroShotJinjaPrompt,
    PropertiesZeroShotGuidancePrompt,
    TaskIdentifierZeroShotGuidancePrompt,
    GraphEntityExtractorPrompt,
    GraphRelationshipExtractorPrompt,
    ExtractTablePropertiesPrompt,
)
from sycamore.llms.prompts.default_prompts import _deprecated_prompts
from sycamore.llms.prompts.prompts import (
    RenderedPrompt,
    RenderedMessage,
    SycamorePrompt,
    ElementListPrompt,
    ElementPrompt,
    StaticPrompt,
)

prompts = [
    "SimplePrompt",
    "EntityExtractorZeroShotJinjaPrompt",
    "EntityExtractorFewShotJinjaPrompt",
    "TextSummarizerGuidancePrompt",
    "SchemaZeroShotJinjaPrompt",
    "PropertiesZeroShotGuidancePrompt",
    "GraphEntityExtractorPrompt",
    "GraphRelationshipExtractorPrompt",
    "ExtractTablePropertiesPrompt",
] + list(_deprecated_prompts.keys())

_all = prompts + [
    "RenderedPrompt",
    "RenderedMessage",
    "SycamorePrompt",
    "ElementListPrompt",
    "ElementPrompt",
    "StaticPrompt",
]

__all__ = _all


def __getattr__(name):
    if name in _deprecated_prompts:
        return getattr(default_prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
