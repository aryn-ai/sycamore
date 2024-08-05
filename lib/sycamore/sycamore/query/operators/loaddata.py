from typing import Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class LoadData(LogicalOperator):
    """Loads data from a specified index."""

    index: str
    """The index to load data from."""

    query: Optional[str] = None
    """The initial query to search for when loading data."""
