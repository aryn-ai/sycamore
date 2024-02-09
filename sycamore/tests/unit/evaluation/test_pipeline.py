import datasets

import sycamore
from sycamore.evaluation.datasets import EvaluationDataSetReader
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.transforms.query import OpenSearchQueryExecutor


class TestEvaluationPipeline:
    INDEX = "toyindex"

    OS_CLIENT_ARGS = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    OS_CONFIG = {
        "size": 10,
        "neural_search_k": 100,
        "embedding_model_id": "pZabXI0BwnxcM6YnF2QF",
        "search_pipeline": "hybrid_rag_pipeline",
        "llm": "gpt-3.5-turbo",
        "context_window": "5",
    }

    def test_hf(self):
        context = sycamore.init()
        reader = EvaluationDataSetReader(context)
        mapping = {"question": "question", "ground_truth_answer": "answer", "ground_truth_document_url": "doc_link"}
        hf_dataset = datasets.load_dataset("PatronusAI/financebench", split=datasets.Split.TRAIN)
        input_docset = reader.huggingface(hf_dataset, field_mapping=mapping)

        pipeline = EvaluationPipeline(
            [], index=self.INDEX, query_executor=OpenSearchQueryExecutor(self.OS_CLIENT_ARGS), os_config=self.OS_CONFIG
        )
        query_level_metrics, aggregated_metrics = pipeline.execute(input_docset.limit(2))
        query_level_metrics.show()
