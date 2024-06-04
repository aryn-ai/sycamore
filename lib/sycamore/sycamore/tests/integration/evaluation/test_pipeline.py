from typing import Any

import datasets
import pytest

import sycamore
from sycamore.data import Element
from sycamore.evaluation import EvaluationDataPoint
from sycamore.evaluation.metrics import document_retrieval_metrics, rouge_metrics
from sycamore.evaluation.datasets import EvaluationDataSetReader
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.transforms.query import OpenSearchQueryExecutor


def _hf_to_qa_datapoint(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "question": "question",
        "ground_truth_answer": "answer",
        "_location": "doc_link",
        "page_number": "page_number",
    }
    document = EvaluationDataPoint()
    for k, v in mapping.items():
        document[k] = data[v]

    document.ground_truth_source_documents = [
        Element({"properties": {"_location": data[mapping["_location"]], "page_number": data[mapping["page_number"]]}})
    ]
    document["raw"] = data
    return {"doc": document.serialize()}


class TestEvaluationPipeline:
    INDEX = "test1"

    OS_CLIENT_ARGS = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    OS_CONFIG = {
        "size": 10,
        "neural_search_k": 100,
        "embedding_model_id": "yrJIzY0BAL0TXCC7hRwd",
        "search_pipeline": "hybrid_rag_pipeline",
        "llm": "gpt-3.5-turbo",
        "context_window": "5",
    }

    @pytest.mark.skip(reason="Requires named models to configure os pipeline unless we setup the cluster on each run")
    def test_hf(self):
        context = sycamore.init()
        reader = EvaluationDataSetReader(context)
        hf_dataset = datasets.load_dataset("PatronusAI/financebench", split=datasets.Split.TRAIN)
        input_docset = reader.huggingface(hf_dataset, doc_extractor=_hf_to_qa_datapoint)

        pipeline = EvaluationPipeline(
            index=self.INDEX,
            os_config=self.OS_CONFIG,
            metrics=[document_retrieval_metrics, rouge_metrics],
            query_executor=OpenSearchQueryExecutor(self.OS_CLIENT_ARGS),
        )
        query_level_metrics, aggregated_metrics = pipeline.execute(input_docset.limit(2))
        taken = query_level_metrics.take()
        print(taken)
