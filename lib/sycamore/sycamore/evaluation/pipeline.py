import logging
import statistics
from typing import Tuple, Optional, Any

from sycamore import DocSet
from sycamore.data import Document, Element, OpenSearchQuery
from sycamore.evaluation.data import EvaluationDataPoint, EvaluationSummary, EvaluationMetric
from sycamore.evaluation.metrics import document_retrieval_metrics, rouge_metrics
from sycamore.transforms.embed import Embedder, SentenceTransformerEmbedder
from sycamore.transforms.query import OpenSearchQueryExecutor

logger = logging.getLogger("ray")


class EvaluationPipeline:
    """
    Standard search pipeline to execute and evaluate a QA dataset. It relies heavily on OpenSearch currently.
    """

    def __init__(
        self,
        index: str,
        os_config: dict[str, str],
        metrics: Optional[list[EvaluationMetric]] = None,
        query_executor: Optional[OpenSearchQueryExecutor] = None,
        os_client_args: Optional[dict] = None,
        subtask_docs: Optional[list[Document]] = None,
        embedder: Embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=100
        ),
        knn_query: bool = False,
    ) -> None:
        super().__init__()
        if metrics is None:
            metrics = [document_retrieval_metrics, rouge_metrics]
        self._metrics = metrics
        self._index = index
        if query_executor is None:
            if os_client_args is None:
                raise RuntimeError("Need to specify a query executor or os_client_args")
            self._query_executor = OpenSearchQueryExecutor(os_client_args)
        else:
            self._query_executor = query_executor
        self._os_config = os_config
        self._subtask_docs = subtask_docs
        self._embedder = embedder
        self._knn_query = knn_query

    def _add_filter(self, query_body: dict, filters: dict[str, str]):
        if self._knn_query:
            knn_query_match = query_body["query"]["knn"]["embedding"]["filter"]["bool"]["must"]
            for key, val in filters.items():
                knn_query_match.append({"match_phrase": {key: val}})
            query_body["query"]["knn"]["embedding"]["filter"]["bool"]["must"] = knn_query_match
        else:
            hybrid_query_match = query_body["query"]["hybrid"]["queries"][0]
            hybrid_query_match = {
                "bool": {
                    "must": [hybrid_query_match],
                    "filter": [{"match_phrase": {k: filters[k]}} for k in filters],
                }
            }
            query_body["query"]["hybrid"]["queries"][0] = hybrid_query_match

            hybrid_query_knn = query_body["query"]["hybrid"]["queries"][1]
            hybrid_query_knn["knn"]["embedding"]["filter"] = {
                "bool": {"must": [{"match_phrase": {k: filters[k]}} for k in filters]}
            }
            query_body["query"]["hybrid"]["queries"][1] = hybrid_query_knn
        return query_body

    def _build_opensearch_query(self, doc: Document) -> Document:
        assert doc.type == "EvaluationDataPoint"
        query = OpenSearchQuery(doc)
        query["index"] = self._index

        if self._subtask_docs and doc.properties["subtasks_reqd"]:
            filtered_docs = [st_doc for st_doc in self._subtask_docs if st_doc.parent_id == doc.doc_id]
            subtask_str = ""
            instructions = filtered_docs[0].properties["instructions"]
            for document in filtered_docs:
                formula = document.text_representation
                if formula:
                    subtask_str += " Formula: " + formula + "; Values: "
                    for elem in document.elements:
                        subtask_str += elem.properties["generated_answer"] + ", "
            subtask_str += " Instructions: " + instructions + " Use this information to answer the following question. "

            doc["question"] = subtask_str.format(**doc.properties["subtask_filters"]) + doc["question"]

        qn_embedding = self._embedder.generate_text_embedding(doc["question"])

        query = OpenSearchQuery(doc)
        query["index"] = self._index

        if self._knn_query:
            query["query"] = {
                "_source": {"excludes": ["embedding"]},
                "size": self._os_config.get("size", 20),
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": qn_embedding,
                            "k": self._os_config.get("neural_search_k", 100),
                            "filter": {
                                "bool": {
                                    "must": [
                                        {"match": {"text_representation": doc["question"]}},
                                    ],
                                },
                            },
                        }
                    }
                },
            }
        else:
            query["query"] = {
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "hybrid": {
                        "queries": [
                            {"match": {"text_representation": doc["question"]}},
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": qn_embedding,
                                        "k": self._os_config.get("neural_search_k", 100),
                                    }
                                }
                            },
                        ]
                    }
                },
                "size": self._os_config.get("size", 20),
            }

        if "llm" in self._os_config:
            query["params"] = {"search_pipeline": self._os_config["search_pipeline"]}
            query["query"]["ext"] = {
                "generative_qa_parameters": {
                    "llm_question": doc["question"],
                    "context_size": self._os_config.get("context_window", 10),
                    "llm_model": self._os_config.get("llm", "gpt-4"),
                }
            }
            if self._os_config.get("rerank", False):
                query["query"]["ext"]["rerank"] = {"query_context": {"query_text": doc["question"]}}
        if "filters" in doc:
            query["query"] = self._add_filter(query["query"], doc["filters"])
        return query

    def _process_queries(self, query_result: Document) -> Document:
        logger.debug(("Query result: " + str(query_result.keys())))
        assert query_result.type == "OpenSearchQueryResult"
        result = EvaluationDataPoint(query_result)
        result.generated_source_documents = [Element(hit) for hit in query_result["hits"]]
        metrics = {}
        for metric in self._metrics:
            metrics[metric.metric_name()] = metric.evaluate(result)
        result["metrics"] = metrics
        return result

    def _aggregate_metrics(self, query_level_metrics: DocSet) -> Document:
        """
        Currently this method averages all metric types for each document. For metrics that need to be handled
        differently we will need to extend this.
        """
        # todo: move these to DocSet level operations
        summary = EvaluationSummary()
        summary.metrics = {}
        metric_data: dict[str, Any] = {}

        if not self._metrics:
            return summary

        for metric in self._metrics:
            metric_data[metric.metric_name()] = {}
            summary.metrics[metric.metric_name()] = {}

        for doc in query_level_metrics.take_all():
            for metric_key, metric_value in metric_data.items():
                for sub_metric_key, sub_metric_value in doc["metrics"].get(metric_key, {}).items():
                    metric_data[metric_key][sub_metric_key] = metric_data[metric_key].get(sub_metric_key, []) + [
                        sub_metric_value
                    ]
        for metric_key, metric_value in metric_data.items():
            for sub_metric_key, sub_metric_value in metric_value.items():
                summary.metrics[metric_key][sub_metric_key] = statistics.mean(sub_metric_value)
        return summary

    def execute(self, input_dataset: DocSet) -> Tuple[DocSet, Document]:
        # 1. convert input dataset to opensearch queries [EvaluationDataPoint -> OpenSearchQuery]
        opensearch_queries = input_dataset.map(self._build_opensearch_query)

        # 2. execute opensearch queries [OpenSearchQuery -> OpenSearchQueryResult]
        opensearch_results = opensearch_queries.query(query_executor=self._query_executor)

        # 3. query level metrics [OpenSearchQueryResult -> EvaluatedEvaluationDataPoint]
        query_level_metrics = opensearch_results.map(self._process_queries)

        # 4. aggregation metrics [[EvaluatedEvaluationDataPoint] -> EvaluatedQASummary]
        aggregated_metrics = self._aggregate_metrics(query_level_metrics)

        return query_level_metrics, aggregated_metrics
