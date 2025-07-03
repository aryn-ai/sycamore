import pytest

import sycamore
from sycamore import EXEC_RAY
from sycamore.data import Document
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.llms import LLMMode
from sycamore.query.execution.operations import (
    MultiStepDocumentSummarizer,
    summarize_data,
)
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner


@pytest.fixture(scope="class")
def llm():
    llm = OpenAI(OpenAIModels.GPT_4_1_MINI)

    yield llm


class TestOperations:

    def test_document_summarizer(self, llm):
        text_path = str(TEST_DIR / "resources/data/texts/Ray.txt")
        text = open(text_path, "r").read()
        dicts = [
            {
                "doc_id": 1,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": 2,
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": 3,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {  # handle element with not text
                "doc_id": 4,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": 5,
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {  # drop because of limit
                "doc_id": 6,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        question = "What is"
        doc_summarizer = MultiStepDocumentSummarizer(llm, question=question)

        docs[0].text_representation = text[:10000]
        doc = doc_summarizer.summarize(docs[0])
        assert doc.properties["summary"]

    def test_document_summarizer_in_sycamore(self, llm):
        question = "What is"
        doc_summarizer = MultiStepDocumentSummarizer(llm, question=question, llm_mode=LLMMode.ASYNC)
        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=EXEC_RAY)
        result = (
            context.read.binary(path, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .summarize(summarizer=doc_summarizer)
            .take_all()
        )
        for doc in result:
            print(doc.properties["summary"])

    def test_summarize_map_reduce(self, llm):
        question = "What is"
        # doc_summarizer = DocumentSummarizer(llm, question)
        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=EXEC_RAY)
        docset = (
            context.read.binary(path, binary_format="pdf").partition(partitioner=UnstructuredPdfPartitioner()).explode()
        )
        final_summary = summarize_data(llm, question, data_description="Ray paper", input_data=[docset])

        print(final_summary)
        assert final_summary
