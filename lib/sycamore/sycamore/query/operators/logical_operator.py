from dataclasses import dataclass
from typing import Any, Dict, Optional, get_type_hints

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

    description: Optional[str] = None
    """A detailed description of why this operator was chosen for this query plan."""

    @classmethod
    def usage(cls) -> str:
        """Return a detailed description of the usage of this operator. Used by the planner."""
        return f"""**{cls.__name__}**: {cls.__doc__}"""

    @classmethod
    def input_schema(cls) -> Dict[str, LogicalOperatorSchemaField]:
        """Return a dict mapping field name to type hint for each input field."""
        return {k: LogicalOperatorSchemaField(k, v.description, str(v.annotation)) for k, v in cls.model_fields.items()}
