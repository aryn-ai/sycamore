from typing import Any

from sycamore.query.operators.logical_operator import LogicalOperator


class Sort(LogicalOperator):
    """Sorts a database based on the value of a field.

    Returns a database.
    """

    descending: bool = False
    """Determines whether to sort in descending order (greatest value first)."""

    field: str
    """The name of the database field to sort based on."""

    default_value: Any
    """The default value to use for the field in case it is not present in a particular record."""
