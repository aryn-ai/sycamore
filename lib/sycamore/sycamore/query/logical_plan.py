from dataclasses import dataclass
from enum import Enum
from functools import wraps
import json
from typing import Any, Dict, List, Mapping, Optional
from hashlib import sha256

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    model_validator,
    field_serializer,
)


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

    node_type: str = Field(default=None)
    """The type of this node."""

    @field_serializer("node_type")
    def serialize_node_type(self, value: str) -> str:
        """Field serializer for node_type that returns the class name as a default."""
        # We can't do this using the "default" argument to Field, because we don't have
        # the class instance yet at the time the field is created.
        return value or type(self).__name__

    node_id: int
    """A unique integer ID representing this node."""

    description: Optional[str] = None
    """A detailed description of why this operator was chosen for this query plan."""

    inputs: List[int] = []
    """A list of node IDs that this operation depends on."""

    # The nodes that this node depends on. This should be populated externally
    # when a LogicalPlan is created.
    _input_nodes: Optional[List["Node"]] = None

    def input_nodes(self) -> List["Node"]:
        """Returns the nodes that this node depends on."""
        if self._input_nodes is None:
            raise ValueError("input_nodes has not been initialized.")
        return self._input_nodes

    # The cache key for this node. Hidden so it is not included in serialization.
    _cache_key: Optional[str] = None

    def __str__(self) -> str:
        return f"Id: {self.node_id} Op: {type(self).__name__}"

    def logical_compare(self, other: "Node") -> bool:
        """Logically compare two instances of a Node."""
        if not isinstance(other, Node):
            return False

        def exclude_field(field: str):
            """Determine whether the given field should be excluded from comparison."""
            if field not in self.model_fields:
                return False
            json_schema_extra = self.model_fields[field].json_schema_extra
            if not json_schema_extra or not hasattr(json_schema_extra, "get"):
                return False
            return json_schema_extra.get("exclude_from_comparison", False)

        # explicitly use dict to compare and exclude keys if needed
        self_dict = {k: v for k, v in self.model_dump().items() if not exclude_field(k)}
        other_dict = {k: v for k, v in other.model_dump().items() if not exclude_field(k)}

        return self_dict == other_dict

    def __hash__(self) -> int:
        # Note that this hash value will change from run to run as Python's built-in hash()
        # is not deterministic.
        return hash(self.model_dump_json())

    def cache_dict(self) -> dict:
        """Returns a dict representation of this node that can be used for comparison."""

        # We want to exclude fields that may change from plan to plan, but which do not
        # affect the semantic equivalence of the plan.
        retval = self.model_dump(exclude={"node_id", "inputs", "description"})
        # Recursively include inputs.
        retval["inputs"] = [dep.cache_dict() for dep in self.input_nodes()]
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
            raise ValueError("No node_type field found in serialized Node")
        if data["node_type"] in _NODE_SUBCLASSES:
            return _NODE_SUBCLASSES[data["node_type"]].model_validate(data)
        else:
            raise ValueError(f"Unknown node type: {data['node_type']}")

    @classmethod
    def usage(cls) -> str:
        """Return a detailed description of the this query operator. Used by the planner."""
        return f"""**{cls.__name__}**: {cls.__doc__}"""

    @classmethod
    def input_schema(cls) -> Dict[str, NodeSchemaField]:
        """Return a dict mapping field name to type hint for each input field."""
        fields = {k: NodeSchemaField(k, v.description, str(v.annotation)) for k, v in cls.model_fields.items()}
        fields.update(
            {k: NodeSchemaField(k, v.description, str(v.return_type)) for k, v in cls.model_computed_fields.items()}
        )
        return fields


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
    """The query that the plan is for."""

    nodes: Mapping[int, SerializeAsAny[Node]]
    """A mapping of node IDs to nodes in the query plan."""

    result_node: int
    """The ID of the node that is the result of the query."""

    llm_prompt: Optional[Any] = None
    """The LLM prompt that was used to generate this query plan."""

    llm_plan: Optional[str] = None
    """The result generated by the LLM."""

    def __init__(self, **kwargs):
        # Ensure that the correct subclass of Node is used.
        if "nodes" not in kwargs:
            raise ValueError("No 'nodes' field provided for LogicalPlan")
        if isinstance(kwargs["nodes"], dict):
            for node_id, node in kwargs["nodes"].items():
                if isinstance(node, dict):
                    kwargs["nodes"][node_id] = Node.deserialize(node)

        super().__init__(**kwargs)

    @model_validator(mode="after")
    def patch_node_inputs(self) -> "LogicalPlan":
        """Model validator for LogicalPlan that sets the _input_nodes values for each node."""
        for node in self.nodes.values():
            # pylint: disable=protected-access
            node._input_nodes = [self.nodes[dep_id] for dep_id in node.inputs]
        return self

    def downstream_nodes(self, node_id: int) -> List[int]:
        """Return the IDs of all nodes that are downstream of the given node."""
        return [n for n in self.nodes.keys() if node_id in self.nodes[n].inputs]

    @model_validator(mode='before')
    @classmethod
    def convert_node_dependencies(cls, data: Any) -> Any:
        if isinstance(data, dict):
            assert (
                'card_number' not in data
            ), 'card_number should not be included'
        return data

    def compare(self, other: "LogicalPlan") -> list[LogicalPlanDiffEntry]:
        """
        A simple method to compare 2 logical plans. This comparator traverses a plan 'forward', i.e. it attempts to
        start from node_id == 0 which is typically a data source query. This helps us detect differences in the plan
        in the natural flow of data. If the plans diverge structurally, i.e. 2 nodes have different number of downstream
        nodes we stop traversing.

        @param other: plan to compare against
        @return: List of comparison metrics.
        """
        assert 0 in self.nodes, "Plan a requires at least 1 node with ID [0]"
        assert 0 in other.nodes, "Plan b requires at least 1 node with ID [0]"
        return compare_graphs(self, other, self.nodes[0].node_id, other.nodes[0].node_id, set(), set())


def compare_graphs(
    plan_a: LogicalPlan, plan_b: LogicalPlan, node_id_a: int, node_id_b: int, visited_a: set[int], visited_b: set[int]
) -> list[LogicalPlanDiffEntry]:
    """
    Traverse and compare 2 graphs given a node pointer in each. Computes different comparison metrics per node.
    The function will continue to traverse as long as the graph structure is identical, i.e. same number of outgoing
    nodes per node. It also assumes that the "downstream nodes"/edges are ordered - this is the current logical
    plan implementation to support operations like math.

    @param plan_a: LogicalPlan a
    @param plan_b: LogicalPlan b
    @param node_id_a: graph node a
    @param node_id_b: graph node b
    @param visited_a: helper to track traversal in graph a
    @param visited_b: helper to track traversal in graph b
    @return: list of LogicalPlanDiffEntry
    """
    diff_results: list[LogicalPlanDiffEntry] = []

    if node_id_a in visited_a and node_id_b in visited_b:
        return diff_results

    visited_a.add(node_id_a)
    visited_b.add(node_id_b)

    node_a = plan_a.nodes[node_id_a]
    node_b = plan_b.nodes[node_id_b]

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
    a_downstream = plan_a.downstream_nodes(node_id_a)
    b_downstream = plan_b.downstream_nodes(node_id_b)

    if len(a_downstream) != len(b_downstream):
        diff_results.append(
            LogicalPlanDiffEntry(node_a=node_a, node_b=node_b, diff_type=LogicalNodeDiffType.PLAN_STRUCTURE)
        )
    else:
        for ds1, ds2 in zip(a_downstream, b_downstream):
            diff_results.extend(compare_graphs(plan_a, plan_b, ds1, ds2, visited_a, visited_b))

    return diff_results
