from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Sort(LogicalOperator):
    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **Sort**: Sorts a database based on the value of a field. Parameters are *description*, 
            *descending*, *field*, *defaultValue*, *input*, and *id*. Returns a database.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *descending* is a Boolean that determines whether to sort in descending order
            (greatest value first).
        - *field* is the name of the database field to sort based on.
        - *defaultValue* is a required field. This is the default value to use for the field in 
            case it is not present in a particular record. *defaultValue* should be the same type 
            as a database value corresponding to *field*. It will determine where the database 
            records with missing values corresponding to *feild* will belong.
        - *input* is a list of operation ids that this operation depends on. For this operation, 
            *input* should only contain one id of an operation that returns a database 
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
            "defaultValue": "Any",
            "input": "number array",
            "id": "number",
        }
        return schema
