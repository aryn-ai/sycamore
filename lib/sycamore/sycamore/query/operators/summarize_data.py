from pydantic import Field

from sycamore import DocSet
from sycamore.query.logical_plan import Node


class SummarizeData(Node):
    """
    This operation generates an English response to a user query based on the input data provided.

    The response should be in Markdown format. It can contain links, tables, and other
    Markdown elements.

    Whenever possible, provide links to relevant data sources and documents.
    """

    question: str = Field(..., json_schema_extra={"exclude_from_comparison": True})
    """The question to ask the LLM."""

    @property
    def input_types(self) -> set[type]:
        return {DocSet, float, int, str}

    @property
    def output_type(self) -> type:
        return str
