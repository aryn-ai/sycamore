import random
import string

from sycamore.data import Document, Element
from sycamore.llms import OpenAI
from sycamore.transforms.extract_key_value_pair import ExtractKeyValuePair


class TestExtractKeyValuePair:
    def test_llm_query_text_does_not_call_llm(self, mocker):
        llm = mocker.Mock(spec=OpenAI)
        doc = Document()
        element1 = Element()
        doc.elements = [element1]
        prompt = "Give me a one word summary response about the text"
        output_property = "output_property"
        query_agent = LLMTextQueryAgent(prompt=prompt, llm=llm, output_property=output_property)
        doc = query_agent.execute_query(doc)