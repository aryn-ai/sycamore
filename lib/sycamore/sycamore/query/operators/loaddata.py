from typing import Dict, Any

from sycamore.query.operators.logical_operator import LogicalOperator


class LoadData(LogicalOperator):
    def __init__(self, node_id: str, data: Dict[Any, Any] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **LoadData**: Loads data from a specified index.
        Parameters are *description*, *index*, *query*, and *id*. Returns a data table.
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
