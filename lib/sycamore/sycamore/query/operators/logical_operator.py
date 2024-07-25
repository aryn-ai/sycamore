import json
from overrides import override
from typing import Any, Dict, Optional

from sycamore.query.logical_plan import Node


class LogicalOperator(Node):
    """
    Logical operator class for LLM prompting.
    """

    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None):
        super().__init__(node_id)
        self.data = data

    @override
    def show(self, indent=0, verbose=False):
        print(
            " " * indent + f"Id: {self.node_id} Operator type: {type(self).__name__} "
            f"Data description: {self.data.get('description', 'none')}"
        )
        if verbose:
            print(" " * indent + f"  {json.dumps(self.data, indent=indent)}")

    @staticmethod
    def description() -> str:
        raise NotImplementedError

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def output_schema() -> Dict[str, Any]:
        raise NotImplementedError
