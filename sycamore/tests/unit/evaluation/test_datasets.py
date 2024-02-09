import datasets

import sycamore
from sycamore.evaluation.datasets import EvaluationDataSetReader


class TestEvaluationDataSetReader:
    def test_hf(self):
        context = sycamore.init()
        reader = EvaluationDataSetReader(context)
        mapping = {"question": "question", "answer": "answer", "document_url": "doc_link"}
        hf_dataset = datasets.load_dataset("PatronusAI/financebench", split=datasets.Split.TRAIN)
        docset = reader.huggingface(hf_dataset, field_mapping=mapping)
        sample = docset.take(1)[0]

        # verify mappings
        assert sample["answer"] is not None
        assert sample["question"] is not None
        assert sample["document_url"] is not None

        # verify parsing is correct
        assert "generated_answer" not in sample
