#!/usr/bin/env python

# This is a simple CLI for Sycamore Query that lets you run queries against an index,
# and see the generated query plan and result.
#
# Example:
#   poetry run sycamore/query/client.py --index const_ntsb \
#       "How many incidents were there in Washington in 2023?"
#
# Use --help for more options.

import argparse
import json
import logging
from typing import List, Optional, Tuple
import os
import uuid

import structlog

import sycamore
from sycamore import Context
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.utils.cache import S3Cache
from sycamore.utils.import_utils import requires_modules

from sycamore.query.execution.sycamore_executor import SycamoreExecutor
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner import LlmPlanner
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaFetcher
from sycamore.query.visualize import visualize_plan

from rich.console import Console


console = Console()


DEFAULT_OS_CONFIG = {"search_pipeline": "hybrid_pipeline"}
DEFAULT_OS_CLIENT_ARGS = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}


def configure_logging(logfile: Optional[str] = None, log_level=logging.WARN):
    """Configure logging for Sycamore query execution."""

    class CustomLoggerFactory(structlog.stdlib.LoggerFactory):
        """Custom logger factory that directs output to a file."""

        def __init__(self, file_handler):
            super().__init__()
            self.file_handler = file_handler

        def __call__(self, *args, **kwargs):
            logger = super().__call__(*args, **kwargs)
            logger.addHandler(self.file_handler)
            return logger

    logging.basicConfig(level=log_level)

    if logfile:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]

        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(log_level)
        structlog.configure_once(
            processors=processors,  # type: ignore
            logger_factory=CustomLoggerFactory(file_handler),
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            cache_logger_on_first_use=True,
        )


class SycamoreQueryClient:
    """A client for the Sycamore Query engine.

    Args:
        s3_cache_path (optional): S3 path to use for LLM result caching.
        os_config (optional): OpenSearch configuration. Defaults to DEFAULT_OS_CONFIG.
        os_client_args (optional): OpenSearch client arguments. Defaults to DEFAULT_OS_CLIENT_ARGS.
        trace_dir (optional): Directory to write query execution trace.
    """

    @requires_modules("opensearchpy", extra="opensearch")
    def __init__(
        self,
        s3_cache_path: Optional[str] = None,
        os_config: dict = DEFAULT_OS_CONFIG,
        os_client_args: dict = DEFAULT_OS_CLIENT_ARGS,
        trace_dir: Optional[str] = None,
    ):
        from opensearchpy import OpenSearch

        self.s3_cache_path = s3_cache_path
        self.os_config = os_config
        self.os_client_args = os_client_args
        self.trace_dir = trace_dir

        self._os_client = OpenSearch(**self.os_client_args)
        self._os_query_executor = OpenSearchQueryExecutor(self.os_client_args)

    def get_opensearch_incides(self) -> List[str]:
        """Get the schema for the provided OpenSearch index."""
        indices = list([str(k) for k in self._os_client.indices.get_alias().keys()])
        return indices

    @requires_modules("opensearchpy.client.indices", extra="opensearch")
    def get_opensearch_schema(self, index: str) -> OpenSearchSchema:
        """Get the schema for the provided OpenSearch index."""
        from opensearchpy.client.indices import IndicesClient

        schema_provider = OpenSearchSchemaFetcher(IndicesClient(self._os_client), index, self._os_query_executor)
        return schema_provider.get_schema()

    def generate_plan(self, query: str, index: str, schema: OpenSearchSchema) -> LogicalPlan:
        """Generate a logical query plan for the given query, index, and schema."""
        llm_client = OpenAI(
            OpenAIModels.GPT_4O.value, cache=S3Cache(self.s3_cache_path) if self.s3_cache_path else None
        )
        planner = LlmPlanner(
            index,
            data_schema=schema,
            os_config=self.os_config,
            os_client=self._os_client,
            llm_client=llm_client,
        )
        plan = planner.plan(query)
        return plan

    def run_plan(
        self, plan: LogicalPlan, context: Optional[Context] = None, dry_run=False, codegen_mode=False
    ) -> Tuple[str, str]:
        """Run the given logical query plan and return a tuple of the query ID and result."""
        if not context:
            context = self._get_default_context()
        executor = SycamoreExecutor(
            context=context,
            trace_dir=self.trace_dir,
            dry_run=dry_run,
            codegen_mode=codegen_mode,
        )
        query_id = str(uuid.uuid4())
        result = executor.execute(plan, query_id)
        return (query_id, result)

    def query(self, query: str, index: str, dry_run: bool = False, codegen_mode: bool = False) -> str:
        """Run a query against the given index."""
        schema = self.get_opensearch_schema(index)
        plan = self.generate_plan(query, index, schema)
        _, result = self.run_plan(plan, dry_run=dry_run, codegen_mode=codegen_mode)
        return result

    def dump_traces(self, logfile: str, query_id: Optional[str] = None):
        """Dump traces from the given logfile."""
        with open(logfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "query_id" in entry and (not query_id or entry["query_id"] == query_id):
                        console.print(entry)
                except json.JSONDecodeError:
                    console.print(line)

    def _get_default_context(self):
        context_params = {
            "default": {
                "llm": OpenAI(
                    OpenAIModels.GPT_4O.value, cache=S3Cache(self.s3_cache_path) if self.s3_cache_path else None
                ),
            },
            "opensearch": {
                "os_client_args": self.os_client_args,
            },
        }
        return sycamore.init(params=context_params)


def main():
    parser = argparse.ArgumentParser(description="Run a Sycamore query against an index.")
    parser.add_argument("query", type=str, help="Query to run against the index.", nargs="?", default=None)
    parser.add_argument("--show-indices", action="store_true", help="Show all indices")
    parser.add_argument("--index", type=str, help="Index name")
    parser.add_argument(
        "--s3-cache-path",
        type=str,
        help="S3 cache path",
        default=None,
    )
    parser.add_argument("--show-schema", action="store_true", help="Show schema extracted from index.")
    parser.add_argument("--show-dag", action="store_true", help="Show DAG of query plan.")
    parser.add_argument("--show-plan", action="store_true", help="Show generated query plan.")
    parser.add_argument("--plan-only", action="store_true", help="Only generate and show query plan.")
    parser.add_argument("--dry-run", action="store_true", help="Generate and show query plan and execution code")
    parser.add_argument("--codegen-mode", action="store_true", help="Execute through codegen")
    parser.add_argument("--trace-dir", help="Directory to write query execution trace.")
    parser.add_argument("--dump-traces", action="store_true", help="Dump traces from the execution.")
    parser.add_argument("--log-level", type=str, help="Log level", default="WARN")
    args = parser.parse_args()

    if args.trace_dir:
        os.makedirs(os.path.abspath(args.trace_dir), exist_ok=True)
        logfile = f"{os.path.abspath(args.trace_dir)}/sycamore.log"
        configure_logging(logfile, log_level=logging.INFO)
    else:
        configure_logging(log_level=args.log_level)

    if args.dump_traces and not args.trace_dir:
        parser.error("--dump-traces requires --trace-dir")

    if args.trace_dir:
        # Make trace_dir absolute.
        args.trace_dir = os.path.abspath(args.trace_dir)

    client = SycamoreQueryClient(s3_cache_path=args.s3_cache_path, trace_dir=args.trace_dir)

    if args.show_indices:
        for index in client.get_opensearch_incides():
            console.print(index)
        return

    if not args.query:
        parser.error("Query is required")

    schema = client.get_opensearch_schema(args.index)
    if args.show_schema:
        console.rule("Extracted schema")
        console.print(schema)
        console.rule()

    plan = client.generate_plan(args.query, args.index, schema)

    if args.show_plan or args.plan_only:
        console.rule("Generated query plan")
        print(plan.llm_plan)
        console.rule()

    if args.plan_only:
        return

    query_id, result = client.run_plan(plan, args.dry_run, args.codegen_mode)

    console.rule(f"Query result [{query_id}]")
    console.print(result)

    if args.dump_traces:
        console.rule(f"Execution traces from {args.trace_dir}/sycamore.log")
        client.dump_traces(os.path.join(os.path.abspath(args.trace_dir), "sycamore.log"), query_id)

    if args.show_dag:
        import matplotlib.pyplot as plt

        visualize_plan(plan)
        plt.show()


if __name__ == "__main__":
    main()
