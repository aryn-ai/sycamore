from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Limit(LogicalOperator):
    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **Limit**: Limits a database to the first K records.
            Parameters are *description*, *K*, *input*, and *id*. Returns a database.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *K* is the number of records of the database to return.
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a database
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"Limit"',
            "description": "string",
            "K": "number",
            "input": "number array",
            "id": "number",
        }
        return schema
