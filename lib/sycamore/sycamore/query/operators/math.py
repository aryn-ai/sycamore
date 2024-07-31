from pydantic import Field

from sycamore.query.operators.logical_operator import LogicalOperator


class Math(LogicalOperator):
    """
    Performs an arithmetic operation on two input numbers.

    Returns a number.
    """

    operation: str = Field(pattern="^add$|^subtract$|^multiply$|^divide$")
    """The arithmetic operation to perform on the inputs. Options are "add", "subtract",
    "multiply", or "divide"."""
