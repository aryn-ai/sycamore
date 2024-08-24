from abc import abstractmethod
from typing import List, Any, Optional, Tuple, Dict

from sycamore.query.logical_plan import Node
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.query.operators.math import Math

from sycamore.query.execution.operations import math_operation


def get_var_name(node: Node):
    return f"output_{node.node_id}"


def get_str_for_dict(value: Dict):
    items = []
    for k, v in value.items():
        if isinstance(v, type):  # Check if the value is a class
            items.append(f"{repr(k)}: {v.__name__}")
        else:
            items.append(f"{repr(k)}: {repr(v)}")
    return "{" + ", ".join(items) + "}"


class PhysicalOperator:
    """
    This interface represents a physical operator that executes a logical plan operator.

    Args:
        logical_node (LogicalOperator): The logical query plan node to execute. Contains runtime params based on type.
        query_id (str): Query id
        inputs (List[Any]): List of inputs required to execute the node. Varies based on node type.
    """

    def __init__(self, logical_node: LogicalOperator, query_id: str, inputs: Optional[List[Any]] = None) -> None:
        super().__init__()
        self.logical_node = logical_node
        self.query_id = query_id
        self.inputs = inputs

    @abstractmethod
    def execute(self) -> Any:
        """
        execute the node
        :return: execution result, type varies based on node type
        """
        pass

    @abstractmethod
    def script(self, input_var: Optional[str] = None, output_var: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        :return: formatted string representing python implementation on this logic, and a list of imports required
        """
        pass


class MathOperator(PhysicalOperator):
    """
    Perform basic math on 2 input variables.
    """

    def __init__(self, logical_node: Math, query_id: str, inputs: List[Any]) -> None:
        super().__init__(logical_node, query_id, inputs)

    def execute(self) -> Any:
        assert isinstance(self.logical_node, Math)
        operator = self.logical_node.operation
        assert (
            self.inputs is not None
            and len(self.inputs) == 2
            and (
                isinstance(self.inputs[0], int)
                or isinstance(self.inputs[0], float)
                and isinstance(self.inputs[1], int)
                or isinstance(self.inputs[1], float)
            )
            and operator is not None
            and isinstance(operator, str)
        )
        result = math_operation(val1=self.inputs[0], val2=self.inputs[1], operator=operator)
        return result

    def script(self, input_var: Optional[str] = None, output_var: Optional[str] = None) -> Tuple[str, List[str]]:
        assert isinstance(self.logical_node, Math)
        assert self.logical_node.dependencies is not None and len(self.logical_node.dependencies) == 2
        operator = self.logical_node.operation
        result = f"""
{output_var or get_var_name(self.logical_node)} = math_operation(
    val1={input_var or get_var_name(self.logical_node.dependencies[0])},
    val2={input_var or get_var_name(self.logical_node.dependencies[1])},
    operator='{operator}'
)
"""
        return result, ["from sycamore.query.execution.operations import math_operation"]
