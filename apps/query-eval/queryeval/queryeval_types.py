# This module defines types used for the config, input, and output
# files for the Sycamore Query evaluator.

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.schema import OpenSearchSchema
from sycamore.query.planner import PlannerExample


class QueryEvalConfig(BaseModel):
    """Represents the configuration for a Query Eval run."""

    config_file: Optional[str] = None
    results_file: Optional[str] = None
    log_file: Optional[str] = None
    llm: Optional[str] = None
    index: Optional[str] = None
    query_cache_path: Optional[str] = None
    llm_cache_path: Optional[str] = None
    dry_run: Optional[bool] = False
    natural_language_response: Optional[bool] = True
    doc_limit: Optional[int] = None
    overwrite: Optional[bool] = False
    tags: Optional[List[str]] = None


class QueryEvalQuery(BaseModel):
    """Represents a single query and expected response."""

    query: str
    expected: Optional[Union[str, List[Dict[str, Any]]]] = None
    expected_plan: Optional[LogicalPlan] = None
    expected_docs: Optional[list[str]] = None
    plan: Optional[LogicalPlan] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class QueryEvalInputFile(BaseModel):
    """Represents the format of the query eval input file."""

    config: Optional[QueryEvalConfig] = None
    data_schema: Optional[OpenSearchSchema] = None
    examples: Optional[List[PlannerExample]] = None
    queries: List[QueryEvalQuery]


class QueryEvalMetrics(BaseModel):
    """Represents metrics associated with a result."""

    # Plan metrics
    plan_generation_time: Optional[float] = None
    plan_similarity: Optional[float] = None
    plan_diff_count: Optional[int] = None

    # Documenet retrieval metrics
    doc_retrieval_recall: Optional[float] = None
    doc_retrieval_precision: Optional[float] = None

    # Performance metrics
    query_time: Optional[float] = None

    # String answer metrics
    bleu_score: Optional[float] = None
    rouge_score: Optional[float] = None
    similarity_score: Optional[float] = None


class DocumentSummary(BaseModel):
    """Represents a serializable Document summary."""

    doc_id: Optional[str] = None
    text_representation: Optional[str] = None
    path: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class DocSetSummary(BaseModel):
    """Represents a serializable DocSet."""

    docs: list[Any] = []


class QueryEvalResult(BaseModel):
    """Represents a single result for running a query."""

    timestamp: Optional[str] = None
    query: QueryEvalQuery
    plan: Optional[LogicalPlan] = None
    result: Optional[Union[str, DocSetSummary]] = None
    error: Optional[str] = None
    metrics: Optional[QueryEvalMetrics] = None
    notes: Optional[str] = None
    retrieved_docs: Optional[list[str]] = None


class QueryEvalResultsFile(BaseModel):
    """Represents the format of the query eval results file."""

    config: QueryEvalConfig
    data_schema: Optional[OpenSearchSchema] = None
    examples: Optional[List[PlannerExample]] = None
    results: Optional[List[QueryEvalResult]] = None
