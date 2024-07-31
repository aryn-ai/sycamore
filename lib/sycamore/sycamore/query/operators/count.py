from typing import Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Count(LogicalOperator):
    """Determines the length of a particular database (number of records).

    Returns a number.
    """

    field: Optional[str] = None
    """Non-primary database field to return a count based on."""

    primary_field: Optional[str] = None
    """Primary field that represents what a unique entry is considered for the data provided."""
