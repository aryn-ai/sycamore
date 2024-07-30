from sycamore.query.operators.logical_operator import LogicalOperator


class Limit(LogicalOperator):
    """Limits a database to the first K records.

    Parameters are *description*, *num_records*, *input*, and *id*.

    Returns a database.

    - *description* is a written description of the purpose of this operation in this context
        and justification of why you chose to use it.
    - *num_records* is the number of records of the database to return.
    - *input* is a list of operation ids that this operation depends on. For this operation,
      *input* should only contain one id of an operation that returns a database
      (len(input) == 1).
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    num_records: int
