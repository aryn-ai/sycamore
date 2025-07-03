import os
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import structlog
from structlog.contextvars import clear_contextvars, bind_contextvars

from sycamore import Context
from sycamore.data import nanoid36
from sycamore.materialize_config import MaterializeSourceMode
from sycamore.query.logical_plan import LogicalPlan, Node
from sycamore.query.operators.groupby import AggregateCount, AggregateCollect
from sycamore.query.operators.clustering import KMeanClustering, LLMClustering
from sycamore.query.operators.unroll import Unroll
from sycamore.query.result import SycamoreQueryResult, NodeExecution
from sycamore.query.operators.count import Count
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.query_database import (
    QueryDatabase,
    QueryVectorDatabase,
    DataLoader,
)
from sycamore.query.execution.physical_operator import PhysicalOperator
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.groupby import GroupBy
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
    SycamoreGroupBy,
    SycamoreDataLoader,
    SycamoreAggregateCount,
    SycamoreKMeanClustering,
    SycamoreUnroll,
    SycamoreLLMClustering,
    SycamoreAggregateCollect,
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

    def process_node(
        self,
        logical_node: Node,
        result: SycamoreQueryResult,
        is_result_node: Optional[bool] = False,
    ) -> Tuple[Any, bool]:
        """Process the given node. Recursively processes dependencies first.
        Returns the result of the operation and if downstream nodes shouldn't materialize"
        """

        query_id = result.query_id
        bind_contextvars(logical_node=logical_node)
        if logical_node.node_id in self.processed:
            log.info("Already processed")
            return self.processed[logical_node.node_id]
        log.info("Executing dependencies")
        inputs: List[Any] = []

        disable_materialization: bool = False
        # Process inputs first to get their results and sort-affected status
        for n in logical_node.input_nodes():
            op_res_n, flag = self.process_node(n, result, is_result_node=False)
            inputs.append(op_res_n)
            if flag:
                disable_materialization = True

        disable_materialization = isinstance(logical_node, Sort) or disable_materialization

        if self.cache_dir and not self.dry_run:
            cache_dir = os.path.join(self.cache_dir, logical_node.cache_key())
            if result.execution is None:
                result.execution = {}
            if logical_node.node_id not in result.execution:
                result.execution[logical_node.node_id] = NodeExecution(
                    node_id=logical_node.node_id, trace_dir=cache_dir
                )
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = None

        bind_contextvars(logical_node=logical_node)
        log.info("Executing node")
        operation = self.make_sycamore_op(logical_node, query_id, inputs)

        if self.codegen_mode:
            code, imports = operation.script(output_var=(self.OUTPUT_VAR_NAME if is_result_node else None))
            self.imports += imports
            self.node_id_to_code[logical_node.node_id] = code
        self.node_id_to_node[logical_node.node_id] = logical_node

        operation_result = "visited"
        if not self.codegen_mode and not self.dry_run:
            operation_result = operation.execute()
            if cache_dir and hasattr(operation_result, "materialize"):
                log.info("Caching node execution", cache_dir=cache_dir)
                if disable_materialization:
                    operation_result = operation_result.materialize(
                        cache_dir, source_mode=MaterializeSourceMode.RECOMPUTE
                    )
                else:
                    operation_result = operation_result.materialize(
                        cache_dir, source_mode=MaterializeSourceMode.USE_STORED
                    )

        self.processed[logical_node.node_id] = operation_result
        log.info("Executed node", result=str(operation_result))
        return operation_result, disable_materialization

    def make_sycamore_op(self, logical_node: Node, query_id: str, inputs: list[Any]) -> PhysicalOperator:
        if isinstance(logical_node, QueryDatabase):
            return SycamoreQueryDatabase(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, QueryVectorDatabase):
            return SycamoreQueryVectorDatabase(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, DataLoader):
            return SycamoreDataLoader(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, LlmFilter):
            return SycamoreLlmFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, BasicFilter):
            return SycamoreBasicFilter(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, LlmExtractEntity):
            return SycamoreLlmExtractEntity(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, Count):
            return SycamoreCount(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, Sort):
            return SycamoreSort(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, Limit):
            return SycamoreLimit(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, TopK):
            return SycamoreTopK(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, KMeanClustering):
            return SycamoreKMeanClustering(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, LLMClustering):
            return SycamoreLLMClustering(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, GroupBy):
            return SycamoreGroupBy(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, AggregateCount):
            return SycamoreAggregateCount(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, AggregateCollect):
            return SycamoreAggregateCollect(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, Unroll):
            return SycamoreUnroll(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, FieldIn):
            return SycamoreFieldIn(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        # Non-DocSet operations
        if isinstance(logical_node, SummarizeData):
            return SycamoreSummarizeData(
                context=self.context,
                logical_node=logical_node,
                query_id=query_id,
                inputs=inputs,
                trace_dir=self.trace_dir,
            )
        if isinstance(logical_node, Math):
            return MathOperator(logical_node=logical_node, query_id=query_id, inputs=inputs)
        raise ValueError(f"Unsupported node type: {logical_node}")

    def get_code_string(self):
        """Return the generated python code as a string."""

        sio = StringIO()
        unique_import_str = set()
        for import_str in self.imports:
            unique_import_str.add(import_str)
        for import_str in unique_import_str:
            sio.write(f"{import_str}\n")
        # Default imports
        sio.write(
            """from sycamore.query.execution.metrics import SycamoreQueryLogger
from sycamore.utils.cache import S3Cache
import sycamore

"""
        )

        if not self.context.params:
            sio.write("context = sycamore.init()\n\n")

        for node_id in sorted(self.node_id_to_node):
            description = self.node_id_to_node[node_id].description.strip("n")
            code = self.node_id_to_code[node_id].strip("\n")
            sio.write(
                f"""
# {description}
{code}
"""
            )

        sio.write("print(result)")

        return sio.getvalue()

    def execute(self, plan: LogicalPlan, query_id: Optional[str] = None) -> SycamoreQueryResult:
        """Execute a logical plan using Sycamore."""

        try:
            if not query_id:
                query_id = nanoid36()
            bind_contextvars(query_id=query_id)

            result = SycamoreQueryResult(query_id=query_id, plan=plan, result=None)

            log.info("Executing query")
            query_result, _ = self.process_node(plan.nodes[plan.result_node], result, is_result_node=True)

            if self.dry_run:
                if self.codegen_mode:
                    code = self.get_code_string()
                    result.code = code
                return result

            if self.codegen_mode:
                code = self.get_code_string()
                result.code = code
                global_context: dict[str, Any] = {"context": self.context}
                try:
                    # pylint: disable=exec-used
                    exec(code, global_context)
                except Exception as e:
                    # exec(..) doesn't seem to print error messages completely, need to traceback
                    print("Exception occurred:")
                    traceback.print_exc()
                    raise e

                return_value = global_context.get(self.OUTPUT_VAR_NAME)
                result.result = return_value
                return result

            result.result = query_result
            return result
        finally:
            clear_contextvars()
