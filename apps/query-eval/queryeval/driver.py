import datetime
import logging
import os
import time
import traceback
from typing import List, Optional

from pydantic_yaml import to_yaml_str
from rich.console import Console
from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.query.client import SycamoreQueryClient, configure_logging
from yaml import safe_load

from queryeval.types import (
    QueryEvalConfig,
    QueryEvalInputFile,
    QueryEvalMetrics,
    QueryEvalQuery,
    QueryEvalResult,
    QueryEvalResultsFile,
    DocumentSummary,
    DocSetSummary,
)

console = Console()


class QueryEvalDriver:
    """Class to run Sycamore Query evaluations.

    Args:
        input_file_path: Path to the input configuration file. Any options not specified in the
            configuration file can be passed as parameters.
        index: OpenSearch index name.
        results_file: Path to the file where the results will be written.
        log_file: Path to the file where detailed logging output will be written.
        query_cache_path: Path to the query cache directory. If not specified, no query result caching
            will be performed.
        llm_cache_path: Path to the LLM cache directory. If not specified, no LLM result caching will be
            performed.
        dry_run: If True, do not perform any actions. This is useful for testing your configuration.
        natural_language_response: If True, return the response in natural language format. Otherwise,
            return the raw DocSet results.
        doc_limit: Limit the number of documents in each result set to this number.
        llm: LLM model name to use.
        overwrite: If True, overwrite the results file if it already exists.
        tags: List of tags to filter queries by. If empty, all queries will be run.
    """

    def __init__(
        self,
        input_file_path: str,
        index: Optional[str] = None,
        results_file: Optional[str] = None,
        log_file: Optional[str] = None,
        query_cache_path: Optional[str] = None,
        llm_cache_path: Optional[str] = None,
        dry_run: bool = False,
        natural_language_response: bool = True,
        doc_limit: Optional[int] = None,
        llm: Optional[str] = None,
        overwrite: bool = False,
        tags: Optional[List[str]] = None,
    ):
        console.print(":moon: Sycamore Query Eval Driver starting")
        console.print(f"Reading input file: [green]{input_file_path}")
        self.input_file_path = os.path.abspath(input_file_path)
        self.config = QueryEvalDriver.read_input_file(self.input_file_path)

        # Override any config options passed as parameters.
        self.config.config = self.config.config or QueryEvalConfig()
        self.config.config.config_file = self.input_file_path
        self.config.config.results_file = self.config.config.results_file or (
            os.path.abspath(results_file) if results_file else None
        )
        self.config.config.log_file = self.config.config.log_file or (os.path.abspath(log_file) if log_file else None)
        self.config.config.index = self.config.config.index or index
        self.config.config.query_cache_path = self.config.config.query_cache_path or query_cache_path
        self.config.config.llm_cache_path = self.config.config.llm_cache_path or llm_cache_path
        self.config.config.dry_run = self.config.config.dry_run or dry_run
        self.config.config.natural_language_response = (
            self.config.config.natural_language_response or natural_language_response
        )
        self.config.config.doc_limit = self.config.config.doc_limit or doc_limit
        self.config.config.llm = self.config.config.llm or llm
        self.config.config.overwrite = self.config.config.overwrite or overwrite
        self.config.config.tags = self.config.config.tags or tags
        if self.config.config.tags:
            console.print(f":label:  Filtering queries by tags: {self.config.config.tags}")

        # Configure logging.
        if self.config.config.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.config.config.log_file)), exist_ok=True)
        configure_logging(logfile=self.config.config.log_file, log_level=logging.INFO)

        if not self.config.config.index:
            raise ValueError("Index must be specified")
        if not self.config.config.results_file:
            raise ValueError("Results file must be specified")

        if self.config.config.results_file:
            console.print(f"Writing results to: {self.config.config.results_file}")
            os.makedirs(os.path.dirname(os.path.abspath(self.config.config.results_file)), exist_ok=True)

        # Read results file if it exists.
        if (
            not self.config.config.overwrite
            and self.config.config.results_file
            and os.path.exists(self.config.config.results_file)
        ):
            results = self.read_results_file(self.config.config.results_file)
            console.print(
                f":white_check_mark: Read {len(results.results or [])} "
                + f"existing results from {self.config.config.results_file}"
            )
        else:
            results = QueryEvalResultsFile(config=self.config.config, results=[])

        # Build lookup from query string to result object.
        results.results = results.results or []
        self.results_map = {r.query.query: r for r in results.results}

        # Set up Sycamore Query Client.
        self.client = SycamoreQueryClient(
            llm_cache_dir=self.config.config.llm_cache_path,
            cache_dir=self.config.config.query_cache_path,
            llm=self.config.config.llm,
        )

        # Use schema from the results file, input file, or OpenSearch, in that order.
        if results.data_schema:
            self.data_schema = results.data_schema
        if self.config.data_schema:
            self.data_schema = self.config.data_schema
        else:
            self.data_schema = self.client.get_opensearch_schema(self.config.config.index)

        console.print(f"Data schema:\n{self.data_schema}")

        # Use examples from the results file, or input file. Priority is given to the input file.
        self.examples = None
        if results.examples:
            self.examples = results.examples
        if self.config.examples:
            self.examples = self.config.examples

    @staticmethod
    def read_input_file(input_file_path: str) -> QueryEvalInputFile:
        """Read the given input file."""
        with open(input_file_path, "r", encoding="utf8") as input_file:
            retval = safe_load(input_file)
        return QueryEvalInputFile(**retval)

    @staticmethod
    def read_results_file(results_file_path: str) -> QueryEvalResultsFile:
        """Read the given results file."""
        with open(results_file_path, "r", encoding="utf8") as results_file:
            retval = safe_load(results_file)
        if isinstance(retval, dict):
            return QueryEvalResultsFile(**retval)
        else:
            return QueryEvalResultsFile(config=QueryEvalConfig(), results=[])

    def write_results_file(self):
        """Write the results to the results file."""
        if self.config.config.dry_run:
            console.print("[yellow]:point_right: Dry run: skipping writing results file")
            return

        assert self.config.config and self.config.config.results_file

        results_file_obj = QueryEvalResultsFile(
            config=self.config.config,
            data_schema=self.data_schema,
            results=list(self.results_map.values()),
            examples=self.examples,
        )

        with open(self.config.config.results_file, "w", encoding="utf8") as results_file:
            results_file.write(to_yaml_str(results_file_obj))
        console.print(f":white_check_mark: Wrote {len(self.results_map)} results to {self.config.config.results_file}")

    def format_doclist(self, doclist: List[Document]) -> DocSetSummary:
        """Convert a document list query result to a list of dicts."""
        results = []
        for doc in doclist:
            if hasattr(doc, "data"):
                if hasattr(doc.data, "model_dump"):
                    results.append(doc.data.model_dump())
                else:
                    results.append(
                        DocumentSummary(
                            doc_id=doc.doc_id,
                            path=doc.properties.get("path"),
                            text_representation=doc.text_representation,
                        )
                    )
        return DocSetSummary(docs=results)

    def get_result(self, query: QueryEvalQuery) -> Optional[QueryEvalResult]:
        """Get the existing result for the query, or return a new result object."""
        if query.query in self.results_map:
            return self.results_map.get(query.query)

        result = QueryEvalResult(query=query)
        result.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result.metrics = result.metrics or QueryEvalMetrics()
        self.results_map[query.query] = result
        return result

    def _check_tags_match(self, query):
        if not self.config.config or not self.config.config.tags:
            return True
        return set(self.config.config.tags) == set(query.tags or [])

    def do_plan(self, query: QueryEvalQuery, result: QueryEvalResult) -> QueryEvalResult:
        """Generate or return an existing query plan."""
        if self.config.config:
            if self.config.config.dry_run:
                console.print("[yellow]:point_right: Dry run: skipping plan generation")
                return result
            elif not self._check_tags_match(query):
                console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                return result

        if result.plan:
            # Use existing result plan.
            console.print("[blue]:point_right: Using existing query plan from results file")
        elif query.plan:
            # Use plan from input file.
            result.plan = query.plan
            console.print("[blue]:point_right: Using existing query plan from input file")
        else:
            # Generate a plan.
            assert self.config.config
            assert self.config.config.index
            t1 = time.time()
            plan = self.client.generate_plan(
                query.query,
                self.config.config.index,
                self.data_schema,
                examples=self.examples or None,
                natural_language_response=self.config.config.natural_language_response or True,
            )
            t2 = time.time()
            assert result.metrics
            result.metrics.plan_generation_time = t2 - t1

            # For now, remove these fields from the output plan, since they are verbose
            # and not particularly useful for the results file.
            plan.llm_plan = None
            plan.llm_prompt = None
            result.plan = plan
            result.error = None
            console.print(f"[green]:clock1: Generated query plan in {result.metrics.plan_generation_time:.2f} seconds")
            console.print(result.plan)

        return result

    def do_query(self, query: QueryEvalQuery, result: QueryEvalResult) -> QueryEvalResult:
        """Run query plan."""
        if self.config.config:
            if self.config.config.dry_run:
                console.print("[yellow]:point_right: Dry run: skipping query execution")
                return result
            elif not self._check_tags_match(query):
                console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                return result

        if not result.plan:
            console.print("[red]:heavy_exclamation_mark: No plan available - skipping query execution")
            return result

        t1 = time.time()
        result.error = None
        query_result = self.client.run_plan(result.plan)
        if isinstance(query_result.result, str):
            result.result = query_result.result
        query_result = self.client.run_plan(result.plan)
        if isinstance(query_result.result, str):
            result.result = query_result.result
            t2 = time.time()
        elif isinstance(query_result.result, DocSet):
            assert self.config.config
            if self.config.config.doc_limit:
                query_result.result = query_result.result.take(self.config.config.doc_limit)
            else:
                query_result.result = query_result.result.take_all()
            t2 = time.time()
            result.result = self.format_doclist(query_result.result)
        else:
            result.result = str(query_result.result)
            t2 = time.time()
        assert result.metrics
        result.metrics.query_time = t2 - t1
        try:
            result.retrieved_docs = query_result.retrieved_docs()
        except Exception:
            result.retrieved_docs = None

        console.print(f"[green]:clock9: Executed query in {result.metrics.query_time:.2f} seconds")
        console.print(f":white_check_mark: Result: {result.result}")
        return result

    def do_eval(self, query: QueryEvalQuery, result: QueryEvalResult) -> QueryEvalResult:
        """Run query evaluation."""
        if self.config.config:
            if self.config.config.dry_run:
                console.print("[yellow]:point_right: Dry run: skipping eval")
                return result
            elif not self._check_tags_match(query):
                console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                return result

        metrics = result.metrics or QueryEvalMetrics()
        # Evalute query plans
        if not query.expected_plan:
            console.print("[yellow]:construction: No expected query plan found, skipping.. ")
        elif not result.plan:
            console.print("[yellow]:construction: No computed query plan found, skipping.. ")
        else:
            plan_diff = query.expected_plan.compare(result.plan)
            if len(plan_diff) == 0:
                console.print("[green]✔ Plan match")
                metrics.plan_similarity = 1.0
                metrics.plan_diff_count = 0
            else:
                console.print("[red]:x: Plan mismatch")
                for i, diff in enumerate(plan_diff):
                    console.print(f"[{i}]. Diff type: {diff.diff_type.value}")

                    if diff.message:
                        console.print(f"Info: {diff.message}")
                    console.print(f"Expected node: {diff.node_a!r}")
                    console.print(f"Actual node: [red]{diff.node_b!r}")
                    console.print()
                metrics.plan_similarity = max(
                    0.0, (len(query.expected_plan.nodes) - len(plan_diff)) / len(query.expected_plan.nodes)
                )
                metrics.plan_diff_count = len(plan_diff)

        # Evaluate doc retrieval
        if not query.expected_docs:
            console.print("[yellow]:construction: No expected document list found, skipping.. ")
        elif not result.retrieved_docs:
            console.print("[yellow]:construction: No computed document list found, skipping.. ")
        else:
            expected_doc_set = set(query.expected_docs)
            retrieved_doc_set = set(result.retrieved_docs)
            if expected_doc_set.issubset(retrieved_doc_set):
                console.print("[green]✔ Documents retrieved include expected docs")
            else:
                console.print("[red]:x: Documents retrieved don't include all expected docs")
                console.print(f"Missing docs: {expected_doc_set - retrieved_doc_set})")
            metrics.doc_retrieval_recall = len(retrieved_doc_set & expected_doc_set) / len(expected_doc_set)
            metrics.doc_retrieval_precision = len(retrieved_doc_set & expected_doc_set) / len(result.retrieved_docs)

        # Evaluate result
        if not result.result:
            console.print("[yellow] No query execution result available, skipping..", style="italic")

        result.metrics = metrics

        return result

    def plan_all(self):
        """Run the plan stage. All queries without existing plans will have new plans generated."""
        for index, query in enumerate(self.config.queries):
            try:
                if not self._check_tags_match(query):
                    console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                    continue
                console.rule(f"Planning query [{index+1}/{len(self.config.queries)}]: {query.query}")
                result = self.get_result(query)
                result = self.do_plan(query, result)
            except Exception:
                tb = traceback.format_exc()
                console.print(f"[red]Error generating plan: {tb}")
                result.error = f"Error generating plan: {tb}"
        self.write_results_file()
        console.print(":tada: Done!")

    def query_all(self):
        """Run the query stage."""
        for index, query in enumerate(self.config.queries):
            try:
                if not self._check_tags_match(query):
                    console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                    continue
                console.rule(f"Running query [{index+1}/{len(self.config.queries)}]: {query.query}")
                result = self.get_result(query)
                result = self.do_query(query, result)
            except Exception:
                tb = traceback.format_exc()
                console.print(f"[red]Error running query: {tb}")
                result.error = f"Error running query: {tb}"
        self.write_results_file()
        console.print(":tada: Done!")

    def print_metrics_summary(self):
        """Summarize metrics."""
        console.rule("Evaluation summary")

        # Plan metrics
        plan_correct = sum(1 for result in self.results_map.values() if result.metrics.plan_similarity == 1.0)
        console.print(f"Plans correct: {plan_correct}/{len(self.results_map)}")
        average_plan_correctness = sum(
            result.metrics.plan_similarity for result in self.results_map.values() if result.metrics.plan_similarity
        ) / len(self.results_map)
        console.print(f"Avg. plan correctness: {average_plan_correctness}")
        console.print(
            "Avg. plan diff count: "
            + str(
                sum(
                    result.metrics.plan_diff_count
                    for result in self.results_map.values()
                    if result.metrics.plan_diff_count
                )
                / len(self.results_map)
            )
        )
        # Evaluate doc retrieval
        correct_retrievals = sum(
            1 for result in self.results_map.values() if result.metrics.doc_retrieval_recall == 1.0
        )
        expected_retrievals = sum(1 for result in self.results_map.values() if result.query.expected_docs)
        average_precision = (
            sum(
                result.metrics.doc_retrieval_precision
                for result in self.results_map.values()
                if result.metrics.doc_retrieval_precision
            )
            / expected_retrievals
        )
        console.print(f"Successful doc retrievals: {correct_retrievals}/{expected_retrievals}")
        console.print(f"Average precision: {average_precision}")
        # TODO: Query execution metrics
        console.print("Query result correctness: not implemented")

    def eval_all(self):
        """Run the eval stage."""
        for index, query in enumerate(self.config.queries):
            try:
                if not self._check_tags_match(query):
                    console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                    continue
                console.rule(f"Evaluating query [{index+1}/{len(self.config.queries)}]: {query.query}")
                result = self.get_result(query)
                result = self.do_eval(query, result)
            except Exception:
                tb = traceback.format_exc()
                console.print(f"[red]Error running eval: {tb}")
                result.error = f"Error running eval: {tb}"
        self.write_results_file()
        self.print_metrics_summary()

        console.print(":tada: Done!")

    def run(self):
        """Run all stages."""
        for index, query in enumerate(self.config.queries):
            try:
                if not self._check_tags_match(query):
                    console.print("[yellow]:point_right: Skipping query due to tag mismatch")
                    continue
                console.rule(f"Running [{index+1}/{len(self.config.queries)}]: {query.query}")
                result = self.get_result(query)
                result = self.do_plan(query, result)
                result = self.do_query(query, result)
                result = self.do_eval(query, result)
            except Exception:
                tb = traceback.format_exc()
                console.print(f"[red]Error: {tb}")
                result.error = f"Error: {tb}"
        self.write_results_file()
        console.print(":tada: Done!")
