import os
import traceback
import uuid
from typing import Any, Dict, List, Optional

import structlog
from structlog.contextvars import clear_contextvars, bind_contextvars

from sycamore import Context
from sycamore.materialize_config import MaterializeSourceMode
from sycamore.query.logical_plan import LogicalPlan, Node
from sycamore.query.operators.count import Count
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.execution.physical_operator import PhysicalOperator
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.field_in import FieldIn
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
    SycamoreQueryVectorDatabase,
)

log = structlog.get_logger(__name__)


class SycamoreExecutor:
    """The Sycamore Query executor that processes a logical plan and executes it using Sycamore.

    Args:
        context (Context): The Sycamore context to use.
        trace_dir (str, optional): If set, query execution traces will be written to this directory.
        cache_dir (str, optional): If set, intermediate query results will be cached in this
            directory. Query plans will reuse cached results from identical subtrees previously
            executed with the same cache_dir. This can greatly improve performance, but be aware
            that the cache is not automatically invalidated when the source data changes.
            Defaults to None.
        codegen_mode (bool, optional): If set, query execution traces will be done by generating python code.
        dry_run (bool, optional): If set, query will not be executed, only generated python code will be returned.
    """

    OUTPUT_VAR_NAME = "result"

    def __init__(
        self,
        context: Context,
        trace_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        codegen_mode: bool = False,
        dry_run: bool = False,
    ) -> None:
        super().__init__()

        self.context = context
        self.trace_dir = trace_dir
        self.cache_dir = cache_dir
        self.processed: Dict[int, Any] = dict()
        self.dry_run = dry_run
        self.codegen_mode = codegen_mode

        if self.trace_dir and not self.dry_run:
            log.info("Using trace directory: %s", trace_dir)
        if self.cache_dir and not self.dry_run:
            log.info("Using cache directory: %s", cache_dir)
        self.node_id_to_node: Dict[int, Node] = {}
        self.node_id_to_code: Dict[int, str] = {}
        self.imports: List[str] = []

    def process_node(self, logical_node: Node, query_id: str, is_result_node: Optional[bool] = False) -> Any:
        """Process the given node. Recursively processes dependencies first."""

        bind_contextvars(logical_node=logical_node)
        # This is lifted up here to avoid serialization issues with Ray.

        if logical_node.node_id in self.processed:
            log.info("Already processed")
            return self.processed[logical_node.node_id]
        log.info("Executing dependencies")
        inputs: List[Any] = []

        if self.trace_dir and not self.dry_run:
            trace_dir = os.path.join(self.trace_dir, query_id, str(logical_node.node_id))
            os.makedirs(trace_dir, exist_ok=True)
        else:
            trace_dir = None

        if self.cache_dir and not self.dry_run:
            cache_dir = os.path.join(self.cache_dir, logical_node.cache_key())
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = None

        # Process inputs.
        inputs = [self.process_node(n, query_id) for n in logical_node.input_nodes()]

        # refresh context as nested execution overrides it
        bind_contextvars(logical_node=logical_node)
        log.info("Executing node")
        operation: Optional[PhysicalOperator] = None
        if isinstance(logical_node, QueryDatabase):
            operation = SycamoreQueryDatabase(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, QueryVectorDatabase):
            operation = SycamoreQueryVectorDatabase(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, LlmFilter):
            operation = SycamoreLlmFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, BasicFilter):
            operation = SycamoreBasicFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, LlmExtractEntity):
            operation = SycamoreLlmExtractEntity(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
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
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, TopK):
            operation = SycamoreTopK(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, FieldIn):
            operation = SycamoreFieldIn(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        # Non-DocSet operations
        elif isinstance(logical_node, SummarizeData):
            operation = SycamoreSummarizeData(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        elif isinstance(logical_node, Math):
            operation = MathOperator(logical_node=logical_node, query_id=query_id, inputs=inputs)
        else:
            raise ValueError(f"Unsupported node type: {str(logical_node)}")

        code, imports = operation.script(output_var=(self.OUTPUT_VAR_NAME if is_result_node else None))
        self.imports += imports
        self.node_id_to_code[logical_node.node_id] = code
        self.node_id_to_node[logical_node.node_id] = logical_node

        result = "visited"
        if not self.codegen_mode and not self.dry_run:
            result = operation.execute()
            if cache_dir and hasattr(result, "materialize"):
                log.info("Caching node execution", cache_dir=cache_dir)
                result = result.materialize(cache_dir, source_mode=MaterializeSourceMode.USE_STORED)
            if trace_dir and hasattr(result, "materialize"):
                log.info("Materializing result", trace_dir=trace_dir)
                result = result.materialize(trace_dir, source_mode=MaterializeSourceMode.RECOMPUTE)

        self.processed[logical_node.node_id] = result
        log.info("Executed node", result=str(result))
        return result

    def get_code_string(self):
        """Return the generated python code as a string."""

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

        if not self.context.params:
            result += "context = sycamore.init()\n\n"

        for node_id in sorted(self.node_id_to_node):
            description = self.node_id_to_node[node_id].description.strip("n")
            code = self.node_id_to_code[node_id].strip("\n")
            result += f"""
# {description}
{code}
"""
        return result

    def _write_query_plan_to_trace_dir(self, plan: LogicalPlan, query_id: str):
        assert self.trace_dir is not None, "Writing query_plan requires trace_dir to be set"
        path = os.path.join(self.trace_dir, query_id, "metadata")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "query_plan.json"), "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json())

    def execute(self, plan: LogicalPlan, query_id: Optional[str] = None) -> Any:
        """Execute a logical plan using Sycamore."""

        try:
            if not query_id:
                query_id = str(uuid.uuid4())
            bind_contextvars(query_id=query_id)

            log.info("Writing query plan to trace dir")
            if self.trace_dir:
                self._write_query_plan_to_trace_dir(plan, query_id)

            log.info("Executing query")
            result = self.process_node(plan.nodes[plan.result_node], query_id, is_result_node=True)

            if self.dry_run:
                code = self.get_code_string()
                return code

            if self.codegen_mode:
                code = self.get_code_string()
                global_context: dict[str, Any] = {"context": self.context}
                try:
                    # pylint: disable=exec-used
                    exec(code, global_context)
                except Exception as e:
                    # exec(..) doesn't seem to print error messages completely, need to traceback
                    print("Exception occurred:")
                    traceback.print_exc()
                    raise e
                return global_context.get(self.OUTPUT_VAR_NAME)

            return result
        finally:
            clear_contextvars()
