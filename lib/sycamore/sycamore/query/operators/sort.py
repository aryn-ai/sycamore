from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Sort(LogicalOperator):
    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **Sort**: Sorts a data table based on the value of a field.
        Parameters are *description*, *descending*, *field*, *input*, and *id*.
        Returns an ordered data table.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *descending* is a Boolean that determines whether to sort in descending order
            (greatest value first).
        - *field* is the name of the database field to sort based on.
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a data table
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"Sort"',
            "description": "string",
            "descending": "Boolean",
            "field": "string",
            "input": "number array",
            "id": "number",
        }
        return schema
