import io
from collections import OrderedDict
from typing import Any, Dict, Optional

from pydantic import BaseModel

import sycamore
from sycamore.query.logical_plan import LogicalPlan
from sycamore import DocSet
from sycamore.data import MetadataDocument


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

    def retrieved_docs(self, identifier_field: str = "properties.path") -> list[str]:
        """Return a set of Document identifiers for the documents retrieved by the query.

        Args:
            identifier_field: The field to use to identify each document (and what to return).
                Default is 'properties.path'
        """

        context = sycamore.init()

        if self.execution is None:
            raise ValueError("No execution data available.")

        # We want to return the set of documents from the deepest node in the query plan
        # that yields "true" documents from the data source. To do this, we recurse up the query
        # plan tree, collecting the set of documents from each node that has a trace directory
        # and which contain documents with "path" properties.

        def get_source_docs(context: sycamore.Context, node_id: int, sort_by_properties=None) -> list[str]:
            """Helper function to recursively retrieve the source document paths for a given node.

            Args:
                context: The Sycamore context used to read and materialize documents.
                node_id: The ID of the node in the query plan, typically the result node if you want docs for the query.
                sort_by_properties: A list of properties to sort the documents by. defaults: ["score", "_rerank_score"].

            Returns:
                A list of unique document paths sorted by the specified properties.
            """
            if sort_by_properties is None:
                sort_by_properties = ["score", "_rerank_score"]
            if self.execution is not None and node_id in self.execution:
                node_trace_dir = self.execution[node_id].trace_dir
                if node_trace_dir:
                    try:
                        mds = context.read.materialize(node_trace_dir)
                        keep = mds.filter(lambda doc: doc.field_to_value(identifier_field) is not None)
                        results = keep.take_all()
                        if len(results) > 0:
                            for prop in sort_by_properties:
                                results = sorted(
                                    results, key=lambda doc: doc.properties.get(prop, float("-inf")), reverse=True
                                )
                            unique_ids = OrderedDict((doc.field_to_value(identifier_field), None) for doc in results)
                            return list(unique_ids.keys())
                    except ValueError:
                        # This can happen if the materialize directory is empty.
                        # Ignore and move onto the next node.
                        pass

            # Walk up the tree.
            node = self.plan.nodes[node_id]
            retval: OrderedDict = OrderedDict()
            for input_node_id in node.inputs:
                for path in get_source_docs(context, input_node_id):
                    retval[path] = None  # Using None as a placeholder value
            return list(retval.keys())

        return get_source_docs(context, self.plan.result_node)

    def get_metadata(self) -> dict[int, list[MetadataDocument]]:
        from sycamore.materialize import Materialize

        if self.execution is None:
            raise ValueError("No execution data available.")

        result_node = self.plan.result_node
        to_check = {result_node}
        final_nodes_with_docsets = set()
        while len(to_check) > 0:
            n_id = to_check.pop()
            n = self.plan.nodes[n_id]
            if n.output_type is DocSet:
                final_nodes_with_docsets.add(n_id)
            else:
                to_check |= set(n.inputs) - final_nodes_with_docsets

        # This context should never actually execute anything
        ctx = sycamore.init(exec_mode=sycamore.ExecMode.UNKNOWN)
        ret = {}
        for n_id in final_nodes_with_docsets:
            trace = self.execution[n_id].trace_dir
            assert trace is not None, f"Node {n_id} with output type docset has no trace dir!"
            mds = ctx.read.materialize(path=trace).plan.get_plan_nodes(Materialize)[0].load_metadata()
            ret[n_id] = mds

        return ret
