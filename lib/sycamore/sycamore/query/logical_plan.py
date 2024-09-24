from enum import Enum
from functools import wraps
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, SerializeAsAny


def exclude_from_comparison(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._exclude_from_comparison = True
    return wrapper


class Node(BaseModel):
    """Represents a node in a logical query plan.

    Args:
        node_id: The ID of the node.
        _dependencies: The nodes that this node depends on.
        _downstream_nodes: The nodes that depend on this node.
    """

    # This allows pydantic to pick up field descriptions from
    # docstrings.
    model_config = ConfigDict(use_attribute_docstrings=True)

    node_id: int
    """A unique integer ID representing this node."""

    # These are underscored here to prevent them from leaking out to the
    # input_schema used by the planner.

    _dependencies: List["Node"] = []
    _downstream_nodes: List["Node"] = []

    @property
    def dependencies(self) -> Optional[List["Node"]]:
        """The nodes that this node depends on."""
        return self._dependencies

    @property
    def downstream_nodes(self) -> Optional[List["Node"]]:
        """The nodes that depend on this node."""
        return self._downstream_nodes

    def __str__(self) -> str:
        return f"Id: {self.node_id} Op: {type(self).__name__}"

    def logical_compare(self, other):
        if not isinstance(other, Node):
            return False

        # explicitly use dict to compare and exclude keys if needed
        self_dict = {
            k: v
            for k, v in self.__dict__.items()
            if not (self.__fields__[k].json_schema_extra or {}).get("exclude_from_comparison", False)
        }
        other_dict = {
            k: v
            for k, v in other.__dict__.items()
            if not (other.__fields__[k].json_schema_extra or {}).get("exclude_from_comparison", False)
        }

        return self_dict == other_dict


class LogicalNodeDiffType(Enum):
    OPERATOR_TYPE = "operator_type"
    OPERATOR_DATA = "operator_data"
    PLAN_STRUCTURE = "plan_structure"


class LogicalPlanDiffEntry(BaseModel):
    node_a: SerializeAsAny[Node]
    node_b: SerializeAsAny[Node]
    diff_type: LogicalNodeDiffType
    message: Optional[str] = None


class LogicalPlan(BaseModel):
    """Represents a logical query plan.

    Args:
        result_node: The node that is the result of the query.
        query: The query that the plan is for.
        nodes: A mapping of node IDs to nodes.
        llm_prompt: The LLM prompt that was used to generate this query plan.
        llm_plan: The LLM plan that was used to generate this query plan.
    """

    result_node: SerializeAsAny[Node]
    query: str
    nodes: Mapping[int, SerializeAsAny[Node]]
    llm_prompt: Optional[Any] = None
    llm_plan: Optional[str] = None

    def compare(self, other: "LogicalPlan") -> list[LogicalPlanDiffEntry]:
        """
        A simple method to compare 2 logical plans. This comparator traverses a plan 'forward', i.e. it attempts to
        start from node_id == 0 which is typically a data source query. This helps us detect differences in the plan
        in the natural flow of data. If the plans diverge structurally, i.e. 2 nodes have different number of downstream
        nodes we stop traversing.

        @param other: plan to compare against
        @return: List of comparison metrics.
        """
        assert 0 in self.nodes, "Plan a requires at least 1 node indexed [0]"
        assert 0 in other.nodes, "Plan b requires at least 1 node indexed [0]"
        return compare_graphs(self.nodes[0], other.nodes[0], set(), set())


def compare_graphs(node_a: Node, node_b: Node, visited_a: set[int], visited_b: set[int]) -> list[LogicalPlanDiffEntry]:
    """
    Traverse and compare 2 graphs given a node pointer in each. Computes different comparison metrics per node.
    The function will continue to traverse as long as the graph structure is identical, i.e. same number of outgoing
    nodes per node. It also assumes that the "downstream_nodes"/edges are ordered - this is the current logical
    plan implementation to support operations like math.


    @param node_a: graph node a
    @param node_b: graph node b
    @param visited_a: helper to track traversal in graph a
    @param visited_b: helper to track traversal in graph b
    @return: list of LogicalPlanDiffEntry
    """
    diff_results: list[LogicalPlanDiffEntry] = []

    if node_a.node_id in visited_a and node_b.node_id in visited_b:
        return diff_results

    visited_a.add(node_a.node_id)
    visited_b.add(node_b.node_id)

    # Compare node types
    if type(node_a) != type(node_b):
        diff_results.append(
            LogicalPlanDiffEntry(node_a=node_a, node_b=node_b, diff_type=LogicalNodeDiffType.OPERATOR_TYPE)
        )

    # Compare node data
    if not node_a.logical_compare(node_b):
        diff_results.append(
            LogicalPlanDiffEntry(node_a=node_a, node_b=node_b, diff_type=LogicalNodeDiffType.OPERATOR_DATA)
        )

    # Compare the structure (inputs)
    if len(node_a._downstream_nodes) != len(node_b._downstream_nodes):
        diff_results.append(
            LogicalPlanDiffEntry(node_a=node_a, node_b=node_b, diff_type=LogicalNodeDiffType.PLAN_STRUCTURE)
        )
    else:
        for input1, input2 in zip(node_a._downstream_nodes, node_b._downstream_nodes):
            diff_results.extend(compare_graphs(input1, input2, visited_a, visited_b))
    return diff_results
