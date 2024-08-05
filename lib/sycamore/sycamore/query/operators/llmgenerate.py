from sycamore.query.operators.logical_operator import LogicalOperator


class LlmGenerate(LogicalOperator):
    """
    LLM generates a conversational English response given a question and its answer
    (or data from which the answer can be determined).

    Returns a string that contains the conversational English response.
    """

    question: str
    """The question to ask the LLM."""
