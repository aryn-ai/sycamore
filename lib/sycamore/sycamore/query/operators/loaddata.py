from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class LoadData(LogicalOperator):
    """
    Logical operator for loading a data index.
    """

    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None):
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **LoadData**: Loads data from a specified index.
        Parameters are *description*, *index*, *query*, and *id*. Returns a database with fields 
        from the schema.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *index* is the index to load data from.
        - *query* is the initial query to search for when loading data (so that only the relevant
            data is used).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"LoadData"',
            "description": "string",
            "index": "string",
            "query": "string",
            "id": "number",
        }
        return schema
