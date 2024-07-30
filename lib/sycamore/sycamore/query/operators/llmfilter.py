from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class LlmFilter(LogicalOperator):
    """
    Logical operator for filtering a database field using LLMs.
    """

    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None):
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **LlmFilter**: Uses an LLM to filter a database based on the value of a field. Used in
        cases where there may not be an exact match with an existing field.
        Parameters are *description*, *question*, *field*, *input*, and *id*. Returns a database.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *question* is the prompt to the LLM for filtering the existing field.
        - *field* is the name of the existing field to filter based on.
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a database
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"LlmFilter"',
            "description": "string",
            "question": "string",
            "field": "string",
            "input": "number array",
            "id": "number",
        }
        return schema
