from sycamore.query.operators.logical_operator import LogicalOperator


class SummarizeData(LogicalOperator):
    """
    This operation generates an English response to a user query based on the input data provided.

    The response should be in Markdown format. It can contain links, tables, and other
    Markdown elements.

    Whenever possible, provide links to relevant data sources and documents.
    """

    question: str
    """The question to ask the LLM."""
