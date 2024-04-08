from enum import Enum

import pytest
from opensearchpy import OpenSearch

from integration.ingests.index_info import IndexInfo
from integration.queries.opensearch import OpenSearchHelper
from integration.queries.options import Option, BooleanOption, OptionSet
from dataclasses import dataclass


class RagMode(Enum):
    OFF = 0
    ONE_SHOT = 1
    CONVERSATIONAL = 2


DEFAULT_OPTIONS = OptionSet(
    BooleanOption("do_hybrid"),
    BooleanOption("do_rerank"),
    BooleanOption("do_dedup"),
    BooleanOption("do_filter"),
    Option("rag_mode", RagMode),
)
QUESTION_PLACEHOLDER = "{{QUESTION}}"


class QueryConfigGenerator:
    """
    Generator for query configurations
    """

    def __init__(self, options: OptionSet):
        self._options = options

    def __iter__(self):
        for settings in self._options:
            yield QueryConfigGenerator._build_query(**settings)

    @staticmethod
    def _build_query(
        do_hybrid: bool = False,
        do_rerank: bool = False,
        do_dedup: bool = False,
        do_filter: bool = False,
        rag_mode: RagMode = RagMode.OFF,
    ):
        return QueryConfig(
            rag_mode=rag_mode, do_hybrid=do_hybrid, do_rerank=do_rerank, do_dedup=do_dedup, do_filter=do_filter
        )


@dataclass
class QueryConfig:
    do_hybrid: bool
    do_rerank: bool
    do_dedup: bool
    do_filter: bool
    rag_mode: RagMode


class QueryGenerator:
    """
    Generator for queries and pipelines.
    Important method is `generate`
    """

    def __init__(self, opensearch: OpenSearch, index_info: IndexInfo):
        self._opensearch = OpenSearchHelper(opensearch)
        self._index_info = index_info
        self._embedding_id = None
        self._rag_model_id = None
        self._reranking_id = None
        self._index_mappings = None

    def _setup_context_if_needed(self):
        if self._embedding_id is None:
            self._embedding_id = self._opensearch.get_embedding_model()
        if self._rag_model_id is None:
            self._rag_model_id = self._opensearch.get_remote_model()
        if self._reranking_id is None:
            self._reranking_id = self._opensearch.get_reranking_model()
        if self._index_mappings is None:
            self._index_mappings = self._opensearch.get_index_mappings(self._index_info.name)

    def generate(self, query_config: QueryConfig):
        """
        Convert a query configuration into an opensearch pipeline and query,
        using information gleaned from the opensearch cluster
        """
        self._setup_context_if_needed()
        pipeline_def = self._generate_pipeline(query_config)
        query_def = self._generate_query(query_config)
        return pipeline_def, query_def

    def _generate_pipeline(self, query_config):
        pipeline = {}
        if query_config.do_hybrid:
            pipeline["phase_results_processors"] = [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.111, 0.889]}},
                    }
                }
            ]
        if query_config.do_dedup:
            pipeline["response_processors"] = [
                {
                    "remote_processor": {
                        "endpoint": "rps:2796/RemoteProcessorService/ProcessResponse",
                        "processor_name": "dedup02",
                    },
                }
            ]
        if query_config.do_rerank:
            if "response_processors" not in pipeline:
                pipeline["response_processors"] = []
            pipeline["response_processors"].append(
                {
                    "rerank": {
                        "ml_opensearch": {"model_id": self._reranking_id},
                        "context": {"document_fields": ["text_representation"]},
                    }
                }
            )
        if query_config.rag_mode != RagMode.OFF:
            if "response_processors" not in pipeline:
                pipeline["response_processors"] = []
            pipeline["response_processors"].append(
                {
                    "retrieval_augmented_generation": {
                        "tag": "openai_pipeline",
                        "description": "Pipeline Using OpenAI Connector",
                        "model_id": self._rag_model_id,
                        "context_field_list": ["text_representation"],
                    }
                }
            )
        return pipeline

    def _generate_query(self, query_config):
        query = {"size": 20}
        if query_config.do_hybrid:
            query["query"] = {
                "hybrid": {
                    "queries": [
                        {
                            "bool": {
                                "must": [
                                    {"exists": {"field": "text_representation"}},
                                    {"match": {"text_representation": QUESTION_PLACEHOLDER}},
                                ]
                            }
                        },
                        {
                            "neural": {
                                "embedding": {
                                    "query_text": QUESTION_PLACEHOLDER,
                                    "k": 100,
                                    "model_id": self._embedding_id,
                                }
                            }
                        },
                    ]
                }
            }
            if query_config.do_filter:
                filter_clause = {"match": {"text_representation": "anything"}}
                query["query"]["hybrid"]["queries"][0]["bool"]["filter"] = filter_clause
                query["query"]["hybrid"]["queries"][1]["neural"]["embedding"]["filter"] = filter_clause
        else:
            query["query"] = {
                "neural": {"embedding": {"query_text": QUESTION_PLACEHOLDER, "k": 100, "model_id": self._embedding_id}}
            }
            if query_config.do_filter:
                filter_clause = {"match": {"text_representation": "anything"}}
                query["query"]["neural"]["embedding"]["filter"] = filter_clause
        if query_config.do_rerank:
            query["ext"] = {"rerank": {"query_context": {"query_text": QUESTION_PLACEHOLDER}}}
        if query_config.rag_mode != RagMode.OFF:
            if "ext" not in query:
                query["ext"] = {}
            query["ext"]["generative_qa_parameters"] = {
                "llm_question": QUESTION_PLACEHOLDER,
                "context_size": 5,
                "llm_model": "gpt-3.5-turbo",
            }
        if query_config.rag_mode == RagMode.CONVERSATIONAL:
            query["ext"]["generative_qa_parameters"]["memory_id"] = self._opensearch.get_memory_id()
        return query


@pytest.fixture(scope="module")
def query_generator(opensearch_client, ingested_index):
    return QueryGenerator(opensearch_client, ingested_index)
