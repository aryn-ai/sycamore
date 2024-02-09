from typing import Tuple

from sycamore import DocSet
from sycamore.data import Document, OpenSearchQuery
from sycamore.evaluation.data import EvaluationDataPoint
from sycamore.evaluation.metrics import EvaluationMetric
from sycamore.transforms.query import OpenSearchQueryExecutor


class EvaluationPipeline:
    def __init__(
        self,
        metrics: list[EvaluationMetric],
        index: str,
        query_executor: OpenSearchQueryExecutor,
        os_config: dict[str, str],
    ) -> None:
        super().__init__()
        self._metrics = metrics
        self._index = index
        self._query_executor = query_executor
        self._os_config = os_config

    def _build_opensearch_query(self, doc: Document) -> Document:
        assert doc.type == "EvaluationDataPoint"
        query = OpenSearchQuery()
        query["index"] = self._index
        if "llm" in self._os_config:
            query["params"] = {"search_pipeline": self._os_config["search_pipeline"], "_source_excludes": "embedding"}
            query["query"] = {
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "must": [
                            {
                                "hybrid": {
                                    "queries": [
                                        {"match": {"text_representation": doc["question"]}},
                                        {
                                            "neural": {
                                                "embedding": {
                                                    "query_text": doc["question"],
                                                    "model_id": self._os_config["embedding_model_id"],
                                                    "k": self._os_config["neural_search_k"],
                                                }
                                            }
                                        },
                                    ]
                                }
                            }
                        ],
                        "filter": [{"exists": {"field": "parent_id"}}],
                    }
                },
                "ext": {
                    "generative_qa_parameters": {
                        "llm_question": doc["question"],
                        "context_size": self._os_config["context_window"],
                        "llm_model": self._os_config["llm"],
                    }
                },
                "size": self._os_config["size"],
            }
        else:
            query["query"] = {
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "must": [
                            {
                                "hybrid": {
                                    "queries": [
                                        {"match": {"text_representation": doc["question"]}},
                                        {
                                            "neural": {
                                                "embedding": {
                                                    "query_text": doc["question"],
                                                    "model_id": self._os_config["embedding_model_id"],
                                                    "k": self._os_config["neural_search_k"],
                                                }
                                            }
                                        },
                                    ]
                                }
                            }
                        ],
                        "filter": [{"exists": {"field": "parent_id"}}],
                    }
                },
                "size": self._os_config["size"],
            }
        return query

    def _evaluate_queries(self, query_result: Document) -> Document:
        assert query_result.type == "OpenSearchQueryResult"
        result = EvaluationDataPoint(query_result)
        metrics = {}
        for metric in self._metrics:
            metrics[metric.metric_name()] = metric.evaluate(result)
        result["metrics"] = metrics
        return result

    def execute(self, input_dataset: DocSet) -> Tuple[DocSet, DocSet]:
        # 1. convert input dataset to opensearch queries [EvaluationDataPoint -> OpenSearchQuery]
        opensearch_queries = input_dataset.map(self._build_opensearch_query)

        # 2. execute opensearch queries [OpenSearchQuery -> OpenSearchQueryResult]
        opensearch_results = opensearch_queries.query(query_executor=self._query_executor)

        # 3. query level metrics [OpenSearchQueryResult -> EvaluatedEvaluationDataPoint]
        query_level_metrics = opensearch_results.map(self._evaluate_queries)

        # 4. aggregation metrics [[EvaluatedEvaluationDataPoint] -> EvaluatedQASummary]
        aggregated_metrics = query_level_metrics.map(lambda x: x)

        return query_level_metrics, aggregated_metrics
