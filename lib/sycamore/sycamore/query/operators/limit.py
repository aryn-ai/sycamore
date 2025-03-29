from sycamore.query.logical_plan import Node
from typing import Optional


class Limit(Node):
    """Limits a database to the first num_records records.

    Returns a database.
    """

    num_records: int
    field: Optional[str] = None
    """The number of records of the database to return."""
