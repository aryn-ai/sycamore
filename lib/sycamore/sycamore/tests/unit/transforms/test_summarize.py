import random
import string

from sycamore.data import Document, Element
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

        assert doc.elements[0].properties == {}

    def test_summarize_text_calls_llm(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "this is the summary"
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        text_summarizer = LLMElementTextSummarizer(llm, filter_elements_on_length)
        doc = text_summarizer.summarize(doc)

        assert doc.elements[0].properties == {}
        assert doc.elements[1].properties == {"summary": "this is the summary"}


def filter_elements_on_length(element: Element) -> bool:
    return False if element.text_representation is None else len(element.text_representation) > 10
