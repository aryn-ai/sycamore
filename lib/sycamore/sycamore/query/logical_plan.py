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

    _dependencies: Optional[List["Node"]] = None
    _downstream_nodes: Optional[List["Node"]] = None

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
