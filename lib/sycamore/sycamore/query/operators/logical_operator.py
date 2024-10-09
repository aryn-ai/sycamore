from dataclasses import dataclass
from typing import Dict, List, Optional

from sycamore.query.logical_plan import Node


@dataclass
class LogicalOperatorSchemaField:
    field_name: str
    description: Optional[str]
    type_hint: str


class LogicalOperator(Node):
    """
    Logical operator class for LLM prompting.
    """

    input: Optional[List[int]] = None
    """A list of node IDs that this operation depends on."""

    @classmethod
    def usage(cls) -> str:
        """Return a detailed description of the usage of this operator. Used by the planner."""
        return f"""**{cls.__name__}**: {cls.__doc__}"""

    @classmethod
    def input_schema(cls) -> Dict[str, LogicalOperatorSchemaField]:
        """Return a dict mapping field name to type hint for each input field."""
        return {k: LogicalOperatorSchemaField(k, v.description, str(v.annotation)) for k, v in cls.model_fields.items()}
