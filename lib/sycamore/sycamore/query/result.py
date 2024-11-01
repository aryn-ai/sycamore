import io
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel

import sycamore
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

    def retrieved_docs(self) -> Set[str]:
        """Return a set of Document paths for the documents retrieved by the query."""

        context = sycamore.init()

        if self.execution is None:
            raise ValueError("No execution data available.")

        # We want to return the set of documents from the deepest node in the query plan
        # that yields "true" documents from the data source. To do this, we recurse up the query
        # plan tree, collecting the set of documents from each node that has a trace directory
        # and which contain documents with "path" properties.

        def get_source_docs(context: sycamore.Context, node_id: int) -> Set[str]:
            """Helper function to recursively retrieve the source document paths for a given node."""
            if self.execution is not None and node_id in self.execution:
                node_trace_dir = self.execution[node_id].trace_dir
                if node_trace_dir:
                    try:
                        mds = context.read.materialize(node_trace_dir)
                        keep = mds.filter(lambda doc: doc.properties.get("path") is not None)
                        if keep.count() > 0:
                            return {doc.properties.get("path") for doc in keep.take_all()}
                    except ValueError:
                        # This can happen if the materialize directory is empty.
                        # Ignore and move onto the next node.
                        pass

            # Walk up the tree.
            node = self.plan.nodes[node_id]
            retval: Set[str] = set()
            for input_node_id in node.inputs:
                retval = retval.union(get_source_docs(context, input_node_id))
            return retval

        return get_source_docs(context, self.plan.result_node)
