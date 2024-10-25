import io
from typing import Any, Dict, Optional

from pydantic import BaseModel

from sycamore.query.logical_plan import LogicalPlan
from sycamore import DocSet


class NodeExecution(BaseModel):
    """Represents the execution of a node in a query plan, including metrics and debug info."""

    node_id: int
    """The ID of the node in the query plan."""

    trace_dir: Optional[str] = None
    """The directory where the trace for this node is stored."""


class SycamoreQueryResult(BaseModel):
    """Represents a result from a Sycamore Query operation."""

    query_id: str
    """The unique ID of the query operation."""

    plan: LogicalPlan
    """The logical query plan that was executed."""

    result: Any
    """The result of the query operation. Depending on the query, this could be a list of documents,
    a single document, a string, an integer, etc.
    """

    code: Optional[str] = None
    """The Python code corresponding to the query plan."""

    execution: Optional[Dict[int, NodeExecution]] = None
    """A mapping from node ID to the NodeExecution object for that node."""

    def to_str(self, limit: int = 100) -> str:
        """Convert a query result to a string.

        Args:
            result: The result to convert.
            limit: The maximum number of documents to include in the result.
        """
        if isinstance(self.result, str):
            return self.result
        elif isinstance(self.result, DocSet):
            out = io.StringIO()
            self.result.show(limit=limit, stream=out)
            return out.getvalue()
        else:
            return str(self.result)
