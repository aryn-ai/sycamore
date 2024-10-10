from dataclasses import dataclass
from enum import Enum
from functools import wraps
import json
from typing import Any, Dict, List, Mapping, Optional
from hashlib import sha256


from pydantic import BaseModel, ConfigDict, SerializeAsAny, computed_field, model_validator


def exclude_from_comparison(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._exclude_from_comparison = True
    return wrapper


# This is a mapping from class name to subclasses of Node, which is used for deserialization.
_NODE_SUBCLASSES: Dict[str, Any] = {}


@dataclass
class NodeSchemaField:
    field_name: str
    description: Optional[str]
    type_hint: str


class Node(BaseModel):
    """Represents a node in a logical query plan.

    Args:
        node_id: The ID of the node.
        _inputs: The nodes that this node depends on.
    """

    # This allows pydantic to pick up field descriptions from
    # docstrings.
    model_config = ConfigDict(use_attribute_docstrings=True)

    def __init_subclass__(cls, **kwargs: Any):
        """Called when subclasses of Node are initialized. Used to register all subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.__name__ in _NODE_SUBCLASSES:
            raise ValueError(f"Duplicate node type: {cls.__name__}")
        _NODE_SUBCLASSES[cls.__name__] = cls

    node_id: int
    """A unique integer ID representing this node."""

    description: Optional[str] = None
    """A detailed description of why this operator was chosen for this query plan."""

    # These are underscored here to prevent them from leaking out to the
    # input_schema used by the planner.

    # The nodes that this node depends on.
    _inputs: List["Node"] = []
    # The cache key for this node.
    _cache_key: Optional[str] = None

    def get_inputs(self) -> List["Node"]:
        """Return the nodes that this node depends on."""
        return self._inputs

    @computed_field
    @property
    def node_type(self) -> str:
        """Returns the type of this node."""
        return type(self).__name__

    @computed_field
    @property
    def inputs(self) -> List[int]:
        """The serialized form of the inputs to this node."""
        return [dep.node_id for dep in self._inputs]

    def __str__(self) -> str:
        return f"Id: {self.node_id} Op: {type(self).__name__}"

    def logical_compare(self, other: "Node") -> bool:
        """Logically compare two instances of a Node."""
        if not isinstance(other, Node):
            return False

        # explicitly use dict to compare and exclude keys if needed
        self_dict = {
            k: v
            for k, v in self.__dict__.items()
            if not (self.model_fields[k].json_schema_extra or {}).get("exclude_from_comparison", False)
        }
        other_dict = {
            k: v
            for k, v in other.__dict__.items()
            if not (other.model_fields[k].json_schema_extra or {}).get("exclude_from_comparison", False)
        }

        return self_dict == other_dict

    def __hash__(self) -> int:
        # Note that this hash value will change from run to run as Python's built-in hash()
        # is not deterministic.
        return hash(self.model_dump_json())

    def cache_dict(self) -> dict:
        """Returns a dict representation of this node that can be used for comparison."""

        # We want to exclude fields that may change from plan to plan, but which do not
        # affect the semantic equivalence of the plan.
        retval = self.model_dump(exclude={"node_id", "input", "description"})
        # Recursively include inputs.
        retval["inputs"] = [dep.cache_dict() for dep in self.get_inputs()]
        return retval

    def cache_key(self) -> str:
        """Returns the cache key of this node, used for caching intermediate query results during
        execution."""
        if self._cache_key:
            return self._cache_key
        cache_key = self.cache_dict()
        self._cache_key = sha256(json.dumps(cache_key).encode()).hexdigest()
        return self._cache_key

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Node":
        """Used to deserialize a Node from a dictionary, by returning the appropriate Node subclass."""
        if "node_type" not in data:
            raise ValueError("Serialized Node missing node_type field")
        if data["node_type"] in _NODE_SUBCLASSES:
            return _NODE_SUBCLASSES[data["node_type"]](**data)
        else:
            raise ValueError(f"Unknown node type: {data['node_type']}")

    @classmethod
    def usage(cls) -> str:
        """Return a detailed description of the this query operator. Used by the planner."""
        return f"""**{cls.__name__}**: {cls.__doc__}"""

    @classmethod
    def input_schema(cls) -> Dict[str, NodeSchemaField]:
        """Return a dict mapping field name to type hint for each input field."""
        return {k: NodeSchemaField(k, v.description, str(v.annotation)) for k, v in cls.model_fields.items()}


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
        query: The query that the plan is for.
        nodes: A mapping of node IDs to nodes.
        result_node: The node that is the result of the query.
        llm_prompt: The LLM prompt that was used to generate this query plan.
        llm_plan: The LLM plan that was used to generate this query plan.
    """

    query: str
    nodes: Mapping[int, SerializeAsAny[Node]]
    result_node: int
    llm_prompt: Optional[Any] = None
    llm_plan: Optional[str] = None

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "LogicalPlan":
        """Deserialize a LogicalPlan from a dictionary. This is a little complex, due to our use
        of duck typing for Nodes, and the fact that node inputs are serialized as node IDs."""
        if "nodes" not in data:
            raise ValueError("No nodes field found in LogicalPlan")

        # Create Nodes from the serialized data.
        nodes: Dict[int, Node] = {}
        data_nodes = data["nodes"]

        if not isinstance(data_nodes, dict):
            raise ValueError("nodes field must be a dictionary")
        for node_id, data_node in data_nodes.items():
            if not isinstance(data_node, dict):
                raise ValueError("Each node in the nodes field must be a dictionary")
            if "node_type" not in data_node:
                raise ValueError("Each node in the nodes field must have a node_type field")
            nodes[node_id] = Node.deserialize(data_node)

        # Set node inputs.
        for node_id, data_node in data_nodes.items():
            if "inputs" in data_node:
                if not isinstance(data_node["inputs"], list):
                    raise ValueError("inputs field must be a list")
                if not all(isinstance(dep_id, int) for dep_id in data_node["inputs"]):
                    raise ValueError("All elements of inputs must be integers")
                if not all(dep_id in nodes for dep_id in data_node["inputs"]):
                    raise ValueError("All elements of inputs must be valid node IDs")
                # pylint: disable=protected-access
                nodes[node_id]._inputs = [nodes[dep_id] for dep_id in data_node["inputs"]]

        if "query" not in data:
            raise ValueError("No query field found in LogicalPlan")
        if "result_node" not in data:
            raise ValueError("No result_node field found in LogicalPlan")
        if data["result_node"] not in nodes:
            raise ValueError(f"result_node {data['result_node']} not found in nodes")

        return LogicalPlan(
            query=data["query"],
            nodes=nodes,
            result_node=data["result_node"],
            llm_prompt=data.get("llm_prompt"),
            llm_plan=data.get("llm_plan"),
        )

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
    nodes per node. It also assumes that the "downstream nodes"/edges are ordered - this is the current logical
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
    # pylint: disable=unidiomatic-typecheck
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
    if len(node_a.get_inputs()) != len(node_b.get_inputs()):
        diff_results.append(
            LogicalPlanDiffEntry(node_a=node_a, node_b=node_b, diff_type=LogicalNodeDiffType.PLAN_STRUCTURE)
        )
    else:
        for input1, input2 in zip(node_a.get_inputs(), node_b.get_inputs()):
            diff_results.extend(compare_graphs(input1, input2, visited_a, visited_b))
    return diff_results
