from typing import Any, Dict, Optional, get_type_hints

from sycamore.query.logical_plan import Node


class LogicalOperator(Node):
    """
    Logical operator class for LLM prompting.

    Args:
        description: The description of why this operator was chosen for this query plan.
    """

    description: Optional[str] = None

    @classmethod
    def usage(cls) -> str:
        """Return a detailed description of the usage of this operator. Used by the planner."""
        return f"""**{cls.__name__}**: {cls.__doc__}"""

    @classmethod
    def input_schema(cls) -> Dict[str, Any]:
        """Return a dict mapping field name to type hint for each input field."""
        return {k: str(v) for k, v in get_type_hints(cls).items()}
