from sycamore.query.operators.logical_operator import LogicalOperator


class LlmFilter(LogicalOperator):
    """Uses an LLM to filter a database based on the value of a field. Used in
    cases where a basic match cannot be performed with an existing field (when a
    range or match filter — exact or substring will not suffice).

    Returns a database.
    """

    question: str
    """The prompt to the LLM for filtering the existing field."""

    field: str
    """The name of the existing field to filter based on."""
