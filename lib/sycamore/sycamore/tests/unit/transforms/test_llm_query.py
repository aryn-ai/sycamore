import random
import string

from sycamore.data import Document, Element
from sycamore.llms import OpenAI
from sycamore.transforms.llm_query import LLMTextQueryAgent


class TestLLMQuery:
    def test_llm_query_text_does_not_call_llm(self, mocker):
        llm = mocker.Mock(spec=OpenAI)
        doc = Document()
        element1 = Element()
        doc.elements = [element1]
        prompt = "Give me a one word summary response about the text"
        output_property = "output_property"
        query_agent = LLMTextQueryAgent(prompt=prompt, openai_model=llm, output_property=output_property)
        doc = query_agent.execute_query(doc)

        assert output_property not in doc.elements[0].properties

    def test_summarize_text_element_calls_llm(self, mocker):
        llm = mocker.Mock(spec=OpenAI)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"summary": "summary"}
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]
        prompt = "Give me a one word summary response about the text"
        output_property = "output_property"
        query_agent = LLMTextQueryAgent(prompt=prompt, openai_model=llm, output_property=output_property)
        doc = query_agent.execute_query(doc)

        assert doc.elements[0].properties[output_property] == {"summary": "summary"}
        assert doc.elements[1].properties[output_property] == {"summary": "summary"}

    def test_summarize_text_document_calls_llm(self, mocker):
        llm = mocker.Mock(spec=OpenAI)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"summary": "summary"}
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        doc.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        prompt = "Give me a one word summary response about the text"
        output_property = "output_property"
        query_agent = LLMTextQueryAgent(
            prompt=prompt, openai_model=llm, per_element=False, output_property=output_property
        )
        doc = query_agent.execute_query(doc)

        assert doc.properties[output_property] == {"summary": "summary"}
