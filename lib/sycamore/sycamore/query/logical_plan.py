from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Node(BaseModel):
    """Represents a node in a logical query plan.

    Args:
        node_id: The ID of the node.
        dependencies: The nodes that this node depends on.
        downstream_nodes: The nodes that depend on this node.
    """

    node_id: str
    dependencies: Optional[List["Node"]] = None
    downstream_nodes: Optional[List["Node"]] = None

    def __str__(self) -> str:
        return f"Id: {self.node_id} Op: {type(self).__name__}"


class LogicalPlan(BaseModel):
    """Represents a logical query plan.

    Args:
        result_node: The node that is the result of the query.
        query: The query that the plan is for.
        nodes: A mapping of node IDs to nodes.
        openai_plan: The OpenAI plan that was used to generate this plan.
    """

    result_node: Node
    query: str
    nodes: Dict[str, Node]
    openai_plan: Optional[str] = None
