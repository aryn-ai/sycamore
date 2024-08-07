import os
import uuid
from typing import Any, Dict, List, Optional

import structlog
from sycamore.query.operators.count import Count
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.query.execution.physical_operator import PhysicalOperator
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.field_in import FieldIn
from structlog.contextvars import clear_contextvars, bind_contextvars
from sycamore import Context

from sycamore.query.execution.physical_operator import MathOperator
from sycamore.query.execution.sycamore_operator import (
    SycamoreQueryDatabase,
    SycamoreSummarizeData,
    SycamoreLlmFilter,
    SycamoreBasicFilter,
    SycamoreCount,
    SycamoreLlmExtractEntity,
    SycamoreTopK,
    SycamoreSort,
    SycamoreLimit,
    SycamoreFieldIn,
)
from sycamore.query.logical_plan import LogicalPlan

log = structlog.get_logger(__name__)


class SycamoreExecutor:

    OUTPUT_VAR_NAME = "result"

    """The Sycamore Query executor that processes a logical plan and executes it using Sycamore.

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
        codegen_mode: bool = False,
        dry_run: bool = False,
    ) -> None:
        super().__init__()

        self.context = context
        self.s3_cache_path = s3_cache_path
        self.os_client_args = os_client_args
        self.trace_dir = trace_dir
        self.processed: Dict[int, Any] = dict()
        self.dry_run = dry_run
        self.codegen_mode = codegen_mode

        if self.s3_cache_path:
            log.info("Using S3 cache path: %s", s3_cache_path)
        if self.trace_dir:
            log.info("Using trace directory: %s", trace_dir)
        self.node_id_to_node: Dict[int, LogicalOperator] = {}
        self.node_id_to_code: Dict[int, str] = {}
        self.imports: List[str] = []

    @staticmethod
    def get_node_args(query_id: str, logical_node: LogicalOperator) -> Dict:
        return {"name": str(logical_node.node_id)}

    def process_node(self, logical_node: LogicalOperator, query_id: str) -> Any:
        bind_contextvars(logical_node=logical_node)
        # This is lifted up here to avoid serialization issues with Ray.
        s3_cache_path = self.s3_cache_path

        if logical_node.node_id in self.processed:
            log.info("Already processed")
            return self.processed[logical_node.node_id]
        log.info("Executing dependencies")
        inputs = []

        if self.trace_dir:
            trace_dir = os.path.join(self.trace_dir, query_id, str(logical_node.node_id))
            os.makedirs(trace_dir, exist_ok=True)
        else:
            trace_dir = None

        # Process dependencies
        if logical_node.dependencies:
            for dependency in logical_node.dependencies:
                assert isinstance(dependency, LogicalOperator)
                inputs += [self.process_node(dependency, query_id)]

        # refresh context as nested execution overrides it
        bind_contextvars(logical_node=logical_node)
        log.info("Executing node")
        operation: Optional[PhysicalOperator] = None
        if isinstance(logical_node, QueryDatabase):
            operation = SycamoreQueryDatabase(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                os_client_args=self.os_client_args,
            )
        elif isinstance(logical_node, LlmFilter):
            operation = SycamoreLlmFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, BasicFilter):
            operation = SycamoreBasicFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
            )
        elif isinstance(logical_node, LlmExtractEntity):
            operation = SycamoreLlmExtractEntity(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, Count):
            operation = SycamoreCount(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
            )
        elif isinstance(logical_node, Sort):
            operation = SycamoreSort(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
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
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, FieldIn):
            operation = SycamoreFieldIn(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
            )
        # Non-DocSet operations
        elif isinstance(logical_node, SummarizeData):
            operation = SycamoreSummarizeData(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                s3_cache_path=s3_cache_path,
            )
        elif isinstance(logical_node, Math):
            operation = MathOperator(logical_node=logical_node, query_id=query_id, inputs=inputs)
        else:
            raise ValueError(f"Unsupported node type: {str(logical_node)}")

        code, imports = operation.script(
            output_var=(self.OUTPUT_VAR_NAME if not logical_node.downstream_nodes else None)
        )
        self.imports += imports
        self.node_id_to_code[logical_node.node_id] = code
        self.node_id_to_node[logical_node.node_id] = logical_node

        if not self.codegen_mode:
            result = operation.execute()
            if trace_dir and hasattr(result, "materialize"):
                log.info("Materializing result", trace_dir=trace_dir)
                result = result.materialize(trace_dir)

        self.processed[logical_node.node_id] = result
        log.info("Executed node", result=str(result))
        return result

    def get_code_string(self):
        result = ""
        unique_import_str = set()
        for import_str in self.imports:
            unique_import_str.add(import_str)
        for import_str in unique_import_str:
            result += import_str + "\n"
        # Default imports
        result += "from sycamore.query.execution.metrics import SycamoreQueryLogger\n"
        result += "from sycamore.utils.cache import S3Cache\n"
        result += "import sycamore\n\n"
        result += "context = sycamore.init()\n"
        for node_id in sorted(self.node_id_to_node):
            result += f"# {self.node_id_to_node[node_id].description}" + "\n"
            result += self.node_id_to_code[node_id] + "\n"
        return result

    def execute(self, plan: LogicalPlan, query_id: Optional[str] = None) -> Any:
        try:
            """Execute a logical plan using Sycamore."""
            if not query_id:
                query_id = str(uuid.uuid4())
            bind_contextvars(query_id=query_id)
            log.info("Executing query")
            assert isinstance(plan.result_node, LogicalOperator)
            result = self.process_node(plan.result_node, query_id)

            if self.dry_run:
                code = self.get_code_string()
                return code

            if self.codegen_mode:
                code = self.get_code_string()
                global_context: dict[str, Any] = {}
                exec(code, global_context)
                return global_context.get(self.OUTPUT_VAR_NAME)

            return result
        finally:
            clear_contextvars()
