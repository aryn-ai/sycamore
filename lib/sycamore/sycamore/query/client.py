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
from typing import Any, List, Optional, Tuple, Union
import os
import uuid
import structlog
import yaml

import sycamore
from sycamore import Context, ExecMode
from sycamore.context import OperationTypes
from sycamore.llms import LLM, get_llm, MODELS
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.transforms.similarity import HuggingFaceTransformersSimilarityScorer
from sycamore.utils.cache import cache_from_path
from sycamore.utils.import_utils import requires_modules

from sycamore.query.execution.sycamore_executor import SycamoreExecutor
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner import LlmPlanner, PlannerExample
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaFetcher

from rich.console import Console


console = Console()


DEFAULT_OS_CONFIG = {"search_pipeline": "hybrid_pipeline"}
DEFAULT_OS_CLIENT_ARGS = {
    "hosts": [{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": os.getenv("OPENSEARCH_PORT", 9200)}],
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
        context (optional): a configured Sycamore Context. A fresh one is created if not provided.
        llm_cache_dir (optional): Directory to use for LLM result caching.
        os_config (optional): OpenSearch configuration. Defaults to DEFAULT_OS_CONFIG.
        os_client_args (optional): OpenSearch client arguments. Defaults to DEFAULT_OS_CLIENT_ARGS.
        trace_dir (optional): Directory to write query execution trace.
        cache_dir (optional): Directory to use for caching intermediate query results.

    Notes:
        If you override the context, you cannot override the llm_cache_dir, os_client_args, or llm; you need
        to pass those in via the context paramaters, i.e. sycamore.init(params={...})

        To override os_client_args, set params["opensearch"]["os_client_args"]. You are likely to also need
        params["opensearch"]["text_embedder"] = SycamoreQueryClient.default_text_embedder() or another
        embedder of your choice.

        To override the LLM or cache path, you need to override the llm, for example:
        from sycamore.utils.cache import cache_from_path
        params["default"]["llm"] = OpenAI(OpenAIModels.GPT_40.value, cache=cache_from_path("/example/path"))
    """

    @requires_modules("opensearchpy", extra="opensearch")
    def __init__(
        self,
        context: Optional[Context] = None,
        llm_cache_dir: Optional[str] = None,
        os_config: dict = DEFAULT_OS_CONFIG,
        os_client_args: Optional[dict] = None,
        trace_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        sycamore_exec_mode: ExecMode = ExecMode.RAY,
        llm: Optional[Union[LLM, str]] = None,
    ):
        from opensearchpy import OpenSearch

        self.llm_cache_dir = llm_cache_dir
        self.os_config = os_config
        self.trace_dir = trace_dir
        self.cache_dir = cache_dir
        self.sycamore_exec_mode = sycamore_exec_mode

        # TODO: remove these assertions and simplify the code to get all customization via the
        # context.
        if context and os_client_args:
            raise AssertionError("Setting os_client_args requires context==None. See Notes in class documentation.")

        if context and llm_cache_dir:
            raise AssertionError("Setting llm_cache_dir requires context==None. See Notes in class documentation.")

        if context and llm:
            raise AssertionError("Setting llm requires context==None. See Notes in class documentation.")

        os_client_args = os_client_args or DEFAULT_OS_CLIENT_ARGS
        self.context = context or self._get_default_context(llm_cache_dir, os_client_args, sycamore_exec_mode, llm)

        assert self.context.params, "Could not find required params in Context"
        self.os_client_args = self.context.params.get("opensearch", {}).get("os_client_args", os_client_args)
        self._os_client = OpenSearch(**self.os_client_args)
        self._os_query_executor = OpenSearchQueryExecutor(self.os_client_args)

    def get_opensearch_indices(self) -> List[str]:
        """Get the schema for the provided OpenSearch index."""
        indices = list([str(k) for k in self._os_client.indices.get_alias().keys()])
        return indices

    @requires_modules("opensearchpy.client.indices", extra="opensearch")
    def get_opensearch_schema(self, index: str) -> OpenSearchSchema:
        """Get the schema for the provided OpenSearch index.

        To debug:
        logging.getLogger("sycamore.query.schema").setLevel(logging.DEBUG)
        """
        from opensearchpy.client.indices import IndicesClient

        schema_provider = OpenSearchSchemaFetcher(IndicesClient(self._os_client), index, self._os_query_executor)
        return schema_provider.get_schema()

    def generate_plan(
        self,
        query: str,
        index: str,
        schema: OpenSearchSchema,
        examples: Optional[List[PlannerExample]] = None,
        natural_language_response: bool = False,
    ) -> LogicalPlan:
        """Generate a logical query plan for the given query, index, and schema.

        Args:
            query: The query to generate a plan for.
            index: The index to query against.
            schema: The schema for the index.
            examples: Optional examples to use for planning.
            natural_language_response: Whether to generate a natural language response. If False,
                raw data will be returned.
        """
        llm_client = self.context.params.get("default", {}).get("llm")
        if not llm_client:
            llm_client = OpenAI(OpenAIModels.GPT_4O.value, cache=cache_from_path(self.llm_cache_dir))
        planner = LlmPlanner(
            index,
            data_schema=schema,
            os_config=self.os_config,
            os_client=self._os_client,
            llm_client=llm_client,
            examples=examples,
            natural_language_response=natural_language_response,
        )
        plan = planner.plan(query)
        return plan

    def run_plan(self, plan: LogicalPlan, dry_run=False, codegen_mode=False) -> Tuple[str, Any]:
        assert self.context is not None, "Running a plan requires a configured Context"
        """Run the given logical query plan and return a tuple of the query ID and result."""
        executor = SycamoreExecutor(
            context=self.context,
            cache_dir=self.cache_dir,
            trace_dir=self.trace_dir,
            dry_run=dry_run,
            codegen_mode=codegen_mode,
        )
        query_id = str(uuid.uuid4())
        result = executor.execute(plan, query_id)
        return query_id, result

    def query(
        self,
        query: str,
        index: str,
        dry_run: bool = False,
        codegen_mode: bool = False,
    ) -> Any:
        """Run a query against the given index."""
        schema = self.get_opensearch_schema(index)
        plan = self.generate_plan(query, index, schema)
        _, result = self.run_plan(plan, dry_run=dry_run, codegen_mode=codegen_mode)
        return result

    @staticmethod
    def dump_traces(logfile: str, query_id: Optional[str] = None):
        """Dump traces from the given logfile."""
        with open(logfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "query_id" in entry and (not query_id or entry["query_id"] == query_id):
                        console.print(entry)
                except json.JSONDecodeError:
                    console.print(line)

    @staticmethod
    def default_text_embedder():
        return SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")

    @staticmethod
    def _get_default_context(
        llm_cache_dir: Optional[str],
        os_client_args: Optional[dict],
        sycamore_exec_mode: ExecMode,
        llm: Optional[Union[str, LLM]],
    ) -> Context:

        llm_instance: Optional[LLM] = None
        if llm is not None:
            if isinstance(llm, str):
                llm_instance = get_llm(llm)(cache=cache_from_path(llm_cache_dir))
            elif isinstance(llm, LLM):
                llm_instance = llm
            else:
                raise ValueError(f"Invalid LLM type: {type(llm)}")

        context_params = {
            "default": {"llm": llm_instance or OpenAI(OpenAIModels.GPT_4O.value, cache=cache_from_path(llm_cache_dir))},
            "opensearch": {
                "os_client_args": os_client_args,
                "text_embedder": SycamoreQueryClient.default_text_embedder(),
            },
            OperationTypes.BINARY_CLASSIFIER: {
                "llm": llm_instance or OpenAI(OpenAIModels.GPT_4O_MINI.value, cache=cache_from_path(llm_cache_dir))
            },
            OperationTypes.INFORMATION_EXTRACTOR: {
                "llm": llm_instance or OpenAI(OpenAIModels.GPT_4O_MINI.value, cache=cache_from_path(llm_cache_dir))
            },
            OperationTypes.TEXT_SIMILARITY: {"similarity_scorer": HuggingFaceTransformersSimilarityScorer()},
        }
        return sycamore.init(params=context_params, exec_mode=sycamore_exec_mode)

    @staticmethod
    def result_to_str(result: Any, max_docs: int = 100, max_chars_per_doc: int = 2500) -> str:
        """Convert a query result to a string.

        Args:
            result: The result to convert.
            max_docs: The maximum number of documents to include in the result.
            max_chars_per_doc: The maximum number of characters to include in each document.
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, sycamore.docset.DocSet):
            BASE_PROPS = [
                "filename",
                "filetype",
                "page_number",
                "page_numbers",
                "links",
                "element_id",
                "parent_id",
                "_schema",
                "_schema_class",
                "entity",
            ]
            retval = ""
            for doc in result.take(max_docs):
                if isinstance(doc, sycamore.data.MetadataDocument):
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                props_dict["text_representation"] = (
                    doc.text_representation[:max_chars_per_doc] if doc.text_representation is not None else None
                )
                retval += json.dumps(props_dict, indent=2) + "\n"
            return retval
        else:
            return str(result)


def main():
    parser = argparse.ArgumentParser(description="Run a Sycamore query against an index.")
    parser.add_argument("query", type=str, help="Query to run against the index.", nargs="?", default=None)
    parser.add_argument("--show-indices", action="store_true", help="Show all indices")
    parser.add_argument("--index", type=str, help="Index name")
    parser.add_argument("--schema-file", type=str, help="Schema file")
    parser.add_argument("--llm-cache-dir", type=str, help="Directory to write LLM cache.", default=None)
    parser.add_argument(
        "--raw-data-response", action="store_true", help="Return raw data instead of natural language response."
    )
    parser.add_argument("--show-schema", action="store_true", help="Show schema extracted from index.")
    parser.add_argument("--show-prompt", action="store_true", help="Show planner LLM prompt.")
    parser.add_argument("--show-plan", action="store_true", help="Show generated query plan.")
    parser.add_argument("--plan-only", action="store_true", help="Only generate and show query plan.")
    parser.add_argument("--dry-run", action="store_true", help="Generate and show query plan and execution code")
    parser.add_argument("--codegen-mode", action="store_true", help="Execute through codegen")
    parser.add_argument("--trace-dir", help="Directory to write query execution trace.")
    parser.add_argument("--cache-dir", help="Directory to use for query execution cache.")
    parser.add_argument("--dump-traces", action="store_true", help="Dump traces from the execution.")
    parser.add_argument("--log-level", type=str, help="Log level", default="WARN")
    parser.add_argument("--llm", type=str, help="LLM model name", choices=MODELS.keys())
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

    if args.cache_dir:
        # Make cache_dir absolute.
        args.cache_dir = os.path.abspath(args.cache_dir)

    client = SycamoreQueryClient(
        llm_cache_dir=args.llm_cache_dir, trace_dir=args.trace_dir, cache_dir=args.cache_dir, llm=args.llm
    )

    # Show indices and exit.
    if args.show_indices:
        for index in client.get_opensearch_indices():
            console.print(index)
        return

    # either index or index-file is required
    if not args.index and not args.schema_file:
        parser.error("Either index or schema-file is required")

    # query is required
    if not args.query:
        parser.error("Query is required")

    # get schema (schema_file overrides index)
    # index is read from file
    if args.schema_file:
        try:
            with open(args.schema_file, "r") as file:
                schema = yaml.safe_load(file)

        except FileNotFoundError as e:
            print(f"Schema file {args.schema_file} not found: {e}")
            return
        except PermissionError as e:
            print(f"Permission error when reading schema file {args.schema_file}: {e}")
            return
        except (SyntaxError, ValueError, KeyError, TypeError) as e:
            print(f"Error while parsing schema file: {args.schema_file} {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while reading schema file {args.schema_file}: {e}")
            return

    # index is read from OpenSearch
    else:
        schema = client.get_opensearch_schema(args.index)

    if args.show_schema:
        console.rule("Using schema")
        console.print(schema)
        console.rule()

    plan = client.generate_plan(args.query, args.index, schema, natural_language_response=not args.raw_data_response)

    if args.show_plan or args.plan_only:
        console.rule("Generated query plan")
        console.print(plan.model_dump(exclude=["llm_plan", "llm_prompt"]))
        console.rule()

    if args.show_prompt:
        console.rule("Prompt")
        console.print(plan.llm_prompt)
        console.rule()

    if args.plan_only:
        return

    query_id, result = client.run_plan(plan, args.dry_run, args.codegen_mode)

    console.rule(f"Query result [{query_id}]")
    console.print(client.result_to_str(result))

    if args.dump_traces:
        console.rule(f"Execution traces from {args.trace_dir}/sycamore.log")
        client.dump_traces(os.path.join(os.path.abspath(args.trace_dir), "sycamore.log"), query_id)


if __name__ == "__main__":
    main()
