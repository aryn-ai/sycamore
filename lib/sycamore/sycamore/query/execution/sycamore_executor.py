import uuid
from typing import Any, Dict, List, Optional

import structlog
from sycamore.query.operators.count import Count
from sycamore.query.operators.filter import Filter
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.llmextract import LlmExtract
from sycamore.query.operators.llmfilter import LlmFilter
from sycamore.query.operators.llmgenerate import LlmGenerate
from sycamore.query.operators.loaddata import LoadData
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.query.execution.physical_operator import PhysicalOperator
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.topk import TopK
from structlog.contextvars import clear_contextvars, bind_contextvars
from sycamore import Context, DocSet

from sycamore.query.execution.physical_operator import MathOperator
from sycamore.query.execution.sycamore_operator import (
    SycamoreLoadData,
    SycamoreLlmGenerate,
    SycamoreLlmFilter,
    SycamoreFilter,
    SycamoreCount,
    SycamoreLlmExtract,
    SycamoreTopK,
    SycamoreSort,
    SycamoreLimit,
)
from sycamore.query.logical_plan import LogicalPlan

log = structlog.get_logger(__name__)


class SycamoreExecutor:
    """The LUnA query executor that processes a logical plan and executes it using Sycamore.

    Args:
        context (Context): The Sycamore context to use.
        s3_cache_path (str): The S3 path to use for caching queries and results.
        os_client_args (dict): The OpenSearch client arguments. Defaults to None.
        trace_dir (str, optional): If set, query execution traces will be written to this directory.
    """

    def __init__(
        self,
        context: Context,
        os_client_args: Any,
        s3_cache_path: Optional[str] = None,
        trace_dir: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()

        self.context = context
        self.s3_cache_path = s3_cache_path
        self.os_client_args = os_client_args
        self.trace_dir = trace_dir
        self.processed: Dict[str, DocSet] = dict()
        self.dry_run = dry_run

        if self.s3_cache_path:
            log.info("Using S3 cache path: %s", s3_cache_path)
        if self.trace_dir:
            log.info("Using tracer: %s", trace_dir)
        if self.dry_run:
            log.info("Executing in dry-mode")
            self.node_id_to_node: Dict[str, LogicalOperator] = {}
            self.node_id_to_code: Dict[str, str] = {}
            self.imports: List[str] = []

    @staticmethod
    def get_node_args(query_id: str, logical_node: LogicalOperator) -> Dict:
        return {"name": str(logical_node.node_id)}

    def process_node(self, logical_node: LogicalOperator, query_id: str) -> Any:
        bind_contextvars(logical_node_data=logical_node.data)
        # This is lifted up here to avoid serialization issues with Ray.
        s3_cache_path = self.s3_cache_path

        if logical_node.node_id in self.processed:
            log.info("Already processed")
            return self.processed[logical_node.node_id]
        log.info("Executing dependencies")
        inputs = []

        # Process dependencies
        if logical_node.dependencies:
            for dependency in logical_node.dependencies:
                inputs += [self.process_node(dependency, query_id)]

        # refresh context as nested execution overrides it
        bind_contextvars(logical_node_data=logical_node.data)
        log.info("Executing node")
        # Process node
        result = None
        operation: Optional[PhysicalOperator] = None
        if isinstance(logical_node, LoadData):
            operation = SycamoreLoadData(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
                os_client_args=self.os_client_args,
            )
        elif isinstance(logical_node, LlmFilter):
            operation = SycamoreLlmFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, Filter):
            operation = SycamoreFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, LlmExtract):
            operation = SycamoreLlmExtract(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, Count):
            operation = SycamoreCount(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, Sort):
            operation = SycamoreSort(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, Limit):
            operation = SycamoreLimit(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
            )
        elif isinstance(logical_node, TopK):
            operation = SycamoreTopK(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
                s3_cache_path=s3_cache_path,
            )
        # Non-DocSet operations
        elif isinstance(logical_node, LlmGenerate):
            operation = SycamoreLlmGenerate(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, Math):
            operation = MathOperator(logical_node=logical_node, query_id=query_id, inputs=inputs)
        else:
            raise Exception(f"Unsupported node type: {str(logical_node)}")
        if result is None:
            if self.dry_run:
                code, imports = operation.script()
                self.imports += imports
                self.node_id_to_code[logical_node.node_id] = code
                self.node_id_to_node[logical_node.node_id] = logical_node
            else:
                result = operation.execute()

        if result is not None:
            assert isinstance(result, DocSet)
            self.processed[logical_node.node_id] = result
        log.info("Executed node", result=str(result))
        return result

    def get_code_string(self):
        result = ""
        unique_import_str = set()
        for import_list in self.imports:
            for import_str in import_list:
                unique_import_str.add(import_str)
        for import_str in unique_import_str:
            result += import_str + "\n"
        for node_id in sorted(self.node_id_to_node):
            result += f"# {self.node_id_to_node[node_id].data['description']}" + "\n"
            result += self.node_id_to_code[node_id] + "\n"
        return result

    def execute(self, plan: LogicalPlan, query_id: Optional[str] = None) -> Any:
        try:
            """Execute a logical plan using Sycamore."""
            if not query_id:
                query_id = str(uuid.uuid4())
            bind_contextvars(query_id=query_id)
            log.info("Executing query")
            start = plan.result_node()
            result = self.process_node(start, query_id)
        finally:
            clear_contextvars()
        if self.dry_run:
            return self.get_code_string()
        return result
