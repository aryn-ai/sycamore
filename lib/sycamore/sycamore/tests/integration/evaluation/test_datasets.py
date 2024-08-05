from typing import Any

import datasets

import sycamore
from sycamore.data import Element
from sycamore.evaluation import EvaluationDataPoint
from sycamore.evaluation.datasets import EvaluationDataSetReader


def _hf_to_qa_datapoint(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {"question": "question", "ground_truth_answer": "best_answer", "ground_truth_document_url": "source"}
    document = EvaluationDataPoint()
    for k, v in mapping.items():
        if "ground_truth_document_url" == k:
            document.ground_truth_source_documents = [Element({"properties": {"_location": data[v]}})]
        document[k] = data[v]
    document["raw"] = data
    return {"doc": document.serialize()}


class TestEvaluationDataSetReader:
    def test_hf(self):
        from sycamore.tests.integration.evaluation.test_datasets import _hf_to_qa_datapoint

        context = sycamore.init()
        reader = EvaluationDataSetReader(context)
        hf_dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split=datasets.Split.VALIDATION)
        docset = reader.huggingface(hf_dataset, doc_extractor=_hf_to_qa_datapoint)
        sample = docset.take(1)[0]

        # verify mappings
        assert sample["type"] == "EvaluationDataPoint"
        assert sample["ground_truth_answer"] is not None
        assert sample["question"] is not None
        assert sample["ground_truth_document_url"] is not None

        # verify parsing is correct
        assert "generated_answer" not in sample
        assert sample["ground_truth_source_documents"][0].properties["_location"] == sample["ground_truth_document_url"]
