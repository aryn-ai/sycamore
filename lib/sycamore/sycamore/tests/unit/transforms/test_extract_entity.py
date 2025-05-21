from typing import Optional
import logging
from unittest.mock import MagicMock

import sycamore
from sycamore.context import Context, OperationTypes, ExecMode
from sycamore.data import Document, Element
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts import RenderedPrompt
from sycamore.tests.unit.test_docset import TestSimilarityScorer, MockTokenizer
from sycamore.tests.unit.test_docset import MockLLM as docsetMockLLM
from sycamore.tests.unit.transforms.test_llm_filter import tokenizer_doc


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model", default_mode=LLMMode.SYNC)

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        print(prompt)
        if len(prompt.messages) == 1:
            usermessage = prompt.messages[0].content
        else:
            usermessage = prompt.messages[1].content

        if "returnnone" in usermessage:
            return "None"

        if usermessage.startswith("Hi"):
            return usermessage

        if usermessage.startswith("ho!"):
            return "ho there! " + prompt.messages[2].content

        if "s3://path" in usermessage:
            return "alt_title"

        if "Jack Black" in usermessage:
            return "Jack Black"

        if "example" in usermessage:
            return "title2"

        if "title" in usermessage:
            return "title1"

        if "age" in usermessage:
            return "42"

        if "weight" in usermessage:
            return "200 pounds"

        logging.error(f"{prompt} // {llm_kwargs}")
        assert False, "Make all generate branches explicitly check the arguments"

    def is_chat_mode(self):
        return True


class TestEntityExtraction:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "content": {"binary": None, "text": "text"},
            "parent_id": None,
            "properties": {"path": "s3://path"},
            "embedding": None,
            "elements": [
                {
                    "type": "title",
                    "content": {"binary": None, "text": "text1"},
                    "text_representation": "text1",
                    "properties": {
                        "coordinates": [(1, 2)],
                        "page_number": 1,
                        "entity": {
                            "author": "Jack Black",
                            "age": "42",
                            "weight": "200 pounds",
                        },
                    },
                },
                {
                    "type": "table",
                    "content": {"binary": None, "text": "text2"},
                    "text_representation": "text2",
                    "properties": {
                        "page_name": "name",
                        "coordinates": [(1, 2)],
                        "coordinate_system": "pixel",
                    },
                },
            ],
        }
    )

    def test_extract_entity_zero_shot(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm)
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "title1"

    def test_extract_entity_zero_shot_custom_field(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm, field="properties.entity.author")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "Jack Black"

    def test_extract_entity_with_context_llm(self, mocker):
        llm = MockLLM()
        context = Context(
            params={
                "default": {"llm": llm},
            }
        )
        extractor = OpenAIEntityExtractor("title")
        llm_map = extractor.as_llm_map(None, context=context)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "title1"

    def test_extract_entity_few_shot(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm, prompt_template="title")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "title2"

    def test_extract_entity_document_field_messages(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm, use_elements=False, prompt=[], field="properties.path")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "alt_title"

    def test_extract_entity_document_field_string(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm, use_elements=False, prompt="", field="properties.path")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        assert out_docs[0].properties.get("title") == "alt_title"

    def test_extract_entity_with_elements_and_string_prompt(self, mocker):
        llm = MockLLM()
        extractor = OpenAIEntityExtractor("title", llm=llm, use_elements=True, prompt="Hi ")
        llm_map = extractor.as_llm_map(None)
        outdocs = llm_map._local_process([self.doc])
        assert outdocs[0].properties.get("title").startswith("Hi")
        assert "text1" in outdocs[0].properties.get("title")
        assert "text2" in outdocs[0].properties.get("title")

    def test_extract_entity_with_elements_and_messages_prompt(self, mocker):
        llm = MockLLM()
        prompt_messages = [
            {"role": "system", "content": "Yo"},
            {"role": "user", "content": "ho!"},
        ]
        extractor = OpenAIEntityExtractor("title", llm=llm, use_elements=True, prompt=prompt_messages)
        llm_map = extractor.as_llm_map(None)
        outdocs = llm_map._local_process([self.doc])
        assert outdocs[0].properties.get("title").startswith("ho there!")
        assert "text1" in outdocs[0].properties.get("title")
        assert "text2" in outdocs[0].properties.get("title")

    def test_extract_entity_iteration_var_oob(self, mocker):
        llm = MockLLM()
        llm.generate = MagicMock(wraps=llm.generate)
        extractor = OpenAIEntityExtractor(
            "returnnone",
            llm=llm,
            field="properties.entity.author",
            tokenizer=MockTokenizer(),
            max_tokens=10,
            prompt="{{ entity }}",
        )
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])

        assert llm.generate.call_count == 2
        assert out_docs[0].properties["returnnone"] == "None"

    def test_extract_entity_with_similarity_sorting(self, mocker):
        doc_list = [
            Document(
                doc_id="doc_1",
                elements=[
                    Element(properties={"_element_index": 1}, text_representation="test1"),
                    Element(properties={"_element_index": 2}, text_representation="test1"),
                ],
            ),
            Document(
                doc_id="doc_2",
                elements=[
                    Element(
                        properties={"_element_index": 4}, text_representation="test1a"
                    ),  # change to test1a to trick the mock llm
                    Element(properties={"_element_index": 9}, text_representation="test2"),
                ],
            ),
            Document(
                doc_id="doc_3",
                elements=[
                    Element(properties={"_element_index": 1}, text_representation="test2"),
                ],
            ),
            Document(
                doc_id="doc_4",
                text_representation="empty elements, maybe an exploded doc",
                elements=[],
            ),
        ]
        mock_llm = docsetMockLLM()
        similarity_scorer = TestSimilarityScorer()
        mock_llm.generate = MagicMock(wraps=mock_llm.generate)
        context = sycamore.init(
            params={
                OperationTypes.BINARY_CLASSIFIER: {"llm": mock_llm},
                OperationTypes.TEXT_SIMILARITY: {"similarity_scorer": similarity_scorer},
            },
            exec_mode=ExecMode.LOCAL,
        )
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMExtractEntityOutput"
        entity_extractor = OpenAIEntityExtractor(
            new_field,
            llm=mock_llm,
            use_elements=True,
            prompt=[],
            field="text_representation",
            similarity_scorer=similarity_scorer,
            similarity_query="this is an unused query because unit test",
        )

        entity_docset = docset.extract_entity(
            entity_extractor=entity_extractor,
        )
        taken = entity_docset.take()
        assert len(taken) == 4
        assert len(taken[0].elements) == 2
        # Element order should be unchanged regardless of scorer
        assert (taken[1].elements[0]["properties"]["_element_index"]) == 4
        assert (taken[1].elements[1]["properties"]["_element_index"]) == 9
        assert (taken[0].elements[1]["properties"]["_element_index"]) == 2

        # Element order should be changed in the prompt
        assert "ELEMENT 1: test2" in taken[1].properties[new_field]
        assert "ELEMENT 2: test1" in taken[1].properties[new_field]

    def test_extract_entity_with_tokenizer(self, mocker):
        mock_llm = docsetMockLLM()
        mock_tokenizer = MockTokenizer()
        similarity_scorer = TestSimilarityScorer()
        mock_llm.generate = MagicMock(wraps=mock_llm.generate)
        context = sycamore.init(
            params={
                OperationTypes.BINARY_CLASSIFIER: {"llm": mock_llm},
                OperationTypes.TEXT_SIMILARITY: {"similarity_scorer": similarity_scorer},
            },
            exec_mode=ExecMode.LOCAL,
        )
        docset = context.read.document(tokenizer_doc)
        new_field = "_autogen_LLMExtractEntityOutput"
        entity_extractor = OpenAIEntityExtractor(
            new_field,
            llm=mock_llm,
            use_elements=True,
            prompt=[],
            field="text_representation",
            tokenizer=mock_tokenizer,
            max_tokens=42,  # Low token limit to test windowing
        )

        entity_docset = docset.extract_entity(
            entity_extractor=entity_extractor,
        )
        taken = entity_docset.take()

        assert taken[0].properties[f"{new_field}_source_indices"] == [0, 1, 2]
        assert taken[1].properties[f"{new_field}_source_indices"] == [1]  # set to array index, not element_index
        assert taken[0].properties[new_field] == "4"
        assert taken[1].properties[new_field] == "5"
        assert taken[0].elements[0]["properties"]["_autogen_LLMExtractEntityOutput_source_indices"] == [0, 1, 2]
        assert taken[0].elements[1]["properties"]["_autogen_LLMExtractEntityOutput_source_indices"] == [0, 1, 2]
        assert taken[0].elements[2]["properties"]["_autogen_LLMExtractEntityOutput_source_indices"] == [0, 1, 2]
        assert taken[1].elements[0]["properties"]["_autogen_LLMExtractEntityOutput_source_indices"] == [0]
        assert taken[1].elements[1]["properties"]["_autogen_LLMExtractEntityOutput_source_indices"] == [1]

    def test_extract_entity_correct_field_type(self, mocker):
        llm = MockLLM()

        # Test for int type
        extractor = OpenAIEntityExtractor("age", entity_type="int", llm=llm, field="properties.entity.age")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        print(out_docs[0].properties)
        assert out_docs[0].properties.get("age") == 42

        # Test for str type
        extractor = OpenAIEntityExtractor("weight", entity_type="str", llm=llm, field="properties.entity.weight")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        print(out_docs[0].properties)
        assert out_docs[0].properties.get("weight") == "200 pounds"

        # Test for false positive int type
        extractor = OpenAIEntityExtractor("weight", entity_type="int", llm=llm, field="properties.entity.weight")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        print(out_docs[0].properties)
        assert not out_docs[0].properties.get("weight")

        # Test for entity that is "None"
        extractor = OpenAIEntityExtractor("returnnone", llm=llm, field="properties.entity.author")
        llm_map = extractor.as_llm_map(None)
        out_docs = llm_map._local_process([self.doc])
        print(out_docs[0].properties)
        assert not out_docs[0].properties.get("returnnone")
