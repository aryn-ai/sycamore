from sycamore.query.operators.logical_operator import LogicalOperator


class Limit(LogicalOperator):
    """Limits a database to the first num_records records.

    Returns a database.
    """

    num_records: int
    """The number of records of the database to return."""
