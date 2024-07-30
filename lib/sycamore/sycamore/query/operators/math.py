from sycamore.query.operators.logical_operator import LogicalOperator


class Math(LogicalOperator):
    """
    Performs arithmetic operations on numbers.

    Parameters are *description*, *operation*, *input*, and *id*.

    Returns a number.

    - *description* is a written description of the purpose of this operation in this context
        and justification of why you chose to use it.
    - *operation* is the arithmetic operation to perform on the inputs, options are
        “add”, “subtract”, “multiply”, or “divide”.
    - *input* is a list of node ids that this operation depends on. For this operation,
        *input* should contain two node ids that each return a number.
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    operation: str
