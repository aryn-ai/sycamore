import random
import string

from sycamore.data import Document, Element
from sycamore.functions import filter_elements
from sycamore.llms import LLM
from sycamore.transforms.summarize import LLMElementTextSummarizer


class TestSummarize:
    def test_summarize_text_does_not_call_llm(self, mocker):
        llm = mocker.Mock(spec=LLM)
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        doc.elements = [element1]

        text_summarizer = LLMElementTextSummarizer(llm, filter_elements_on_length)
        doc = text_summarizer.summarize(doc)

        assert doc["elements"]["array"][0]["properties"] == {}

    def test_summarize_text_calls_llm(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"summary": "summary"}
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        text_summarizer = LLMElementTextSummarizer(llm, filter_elements_on_length)
        doc = text_summarizer.summarize(doc)

        assert doc["elements"]["array"][0]["properties"] == {}
        assert doc["elements"]["array"][1]["properties"] == {"summary": "summary"}


def filter_elements_on_length(
    document: Document,
    minimum_length: int = 10,
) -> list[Element]:
    def filter_func(element: Element):
        if element.text_representation is not None:
            return len(element.text_representation) > minimum_length

    return filter_elements(document, filter_func)
