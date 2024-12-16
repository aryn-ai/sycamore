from pydantic import Field

from sycamore.query.logical_plan import Node


class Math(Node):
    """
    Performs an arithmetic operation on two input numbers.

    Returns a number.
    """

    operation: str = Field(pattern="^add$|^subtract$|^multiply$|^divide$")
    """The arithmetic operation to perform on the inputs. Options are "add", "subtract",
    "multiply", or "divide"."""

    @property
    def input_types(self) -> set[type]:
        return {int, float}

    @property
    def output_type(self) -> type:
        return float  # note: this can be an integer too, we're just using a compatible type here
