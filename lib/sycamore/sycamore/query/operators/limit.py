from sycamore.query.logical_plan import Node


class Limit(Node):
    """Limits a database to the first num_records records.

    Returns a database.
    """

    num_records: int
    """The number of records of the database to return."""
