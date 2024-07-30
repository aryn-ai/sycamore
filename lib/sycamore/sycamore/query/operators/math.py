from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Math(LogicalOperator):
    """
    Logical operator for performing basic arithmetic.
    """

    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None):
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **Math**: Performs arithmetic operations on numbers.
        Parameters are *description*, *type*, *input*, and *id*. Returns a number.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *type* is the arithmetic operation to perform on the inputs, options are
            “add”, “subtract”, “multiply”, or “divide”.
        - *input* is a list of node ids that this operation depends on. For this operation,
            *input* should contain two node ids that each return a number.
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = schema = {
            "operatorName": '"Math"',
            "description": "string",
            "type": "string",
            "input": "number array",
            "id": "number",
        }
        return schema
