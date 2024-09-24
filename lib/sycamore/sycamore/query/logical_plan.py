from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, SerializeAsAny


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

    # Allows you to exclude certain keys when comparing nodes. This is useful for llm generated strings.
    _keys_to_exclude_for_comparison: set[str] = set()

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
        self_dict = self.dict(exclude=self._keys_to_exclude_for_comparison)
        other_dict = other.dict(exclude=self._keys_to_exclude_for_comparison)

        return self_dict == other_dict


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
