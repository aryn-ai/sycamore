from pydantic import Field

from sycamore import DocSet
from sycamore.query.logical_plan import Node


class LlmExtractEntity(Node):
    """Adds a new field to the input database based on extracting information from an
    existing field.

    Returns a database.
    """

    question: str = Field(..., json_schema_extra={"exclude_from_comparison": True})
    """The prompt to the LLM for creating the new field. Be descriptive with the question and
    include examples if possible."""

    field: str
    """The name of the existing field for the LLM to use."""

    new_field: str = Field(..., json_schema_extra={"exclude_from_comparison": True})
    """The name of the new field to add."""

    new_field_type: str
    """The type of the new field, e.g. int or string."""

    discrete: bool = False
    """True if the new field has a known finite number of possible values (e.g. number, letter,
    continent, color). False otherwise (e.g., for any free text outputs)."""

    @property
    def input_types(self) -> set[type]:
        return {DocSet}

    @property
    def output_type(self) -> type:
        return DocSet
