import io
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel

from sycamore.query.logical_plan import LogicalPlan
from sycamore import DocSet
from sycamore.data import MetadataDocument


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

    trace_dirs: Optional[Dict[int, Optional[str]]] = None
    """A mapping from node ID to the directory where execution traces
    for that node in the query plan can be found."""

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
