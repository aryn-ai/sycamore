from sycamore.query.logical_plan import Node


class Unroll(Node):
    """Unroll based on a particular field."""

    field: str
    """The field to be unrolled"""
