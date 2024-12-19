from typing import Optional
import logging
from unittest.mock import MagicMock

import sycamore
from sycamore.context import Context, OperationTypes, ExecMode
from sycamore.data import Document, Element
from sycamore.transforms import ExtractEntity
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    EntityExtractorFewShotGuidancePrompt,
    EntityExtractorZeroShotGuidancePrompt,
)
from sycamore.tests.unit.test_docset import TestSimilarityScorer, MockTokenizer
from sycamore.tests.unit.test_docset import MockLLM as docsetMockLLM
from sycamore.tests.unit.transforms.test_llm_filter import tokenizer_doc


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if prompt_kwargs == {"messages": [{"role": "user", "content": "s3://path"}]} and llm_kwargs == {}:
            return "alt_title"
        if prompt_kwargs == {"prompt": "s3://path"} and llm_kwargs == {}:
            return "alt_title"

        if (
            prompt_kwargs["entity"] == "title"
            and prompt_kwargs["query"] == "ELEMENT 1: None\nELEMENT 2: None\n"
            and prompt_kwargs["examples"] is None
        ):
            assert isinstance(prompt_kwargs["prompt"], EntityExtractorZeroShotGuidancePrompt)
            assert llm_kwargs is None
            return "title1"

        if (
            prompt_kwargs["entity"] == "title"
            and prompt_kwargs["query"] == "ELEMENT 1: None\nELEMENT 2: None\n"
            and prompt_kwargs["examples"] == "title"
        ):
            assert isinstance(prompt_kwargs["prompt"], EntityExtractorFewShotGuidancePrompt)
            assert llm_kwargs is None
            return "title2"

        if (
            prompt_kwargs["entity"] == "title"
            and prompt_kwargs["query"] == "ELEMENT 1: Jack Black\nELEMENT 2: None\n"
            and prompt_kwargs["examples"] is None
        ):
            assert isinstance(prompt_kwargs["prompt"], EntityExtractorZeroShotGuidancePrompt)
            assert llm_kwargs is None
            return "Jack Black"

        logging.error(f"{prompt_kwargs} // {llm_kwargs}")
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
            "embedding": {"binary": None, "text": None},
            "elements": [
                {
                    "type": "title",
                    "content": {"binary": None, "text": "text1"},
                    "properties": {"coordinates": [(1, 2)], "page_number": 1, "entity": {"author": "Jack Black"}},
                },
                {
                    "type": "table",
                    "content": {"binary": None, "text": "text2"},
                    "properties": {"page_name": "name", "coordinates": [(1, 2)], "coordinate_system": "pixel"},
                },
            ],
        }
    )

    def test_extract_entity_zero_shot(self, mocker):
        llm = MockLLM()
        extract_entity = ExtractEntity(None, entity_extractor=OpenAIEntityExtractor("title", llm=llm))
        out_doc = extract_entity.run(self.doc)
        assert out_doc.properties.get("title") == "title1"

    def test_extract_entity_zero_shot_custom_field(self, mocker):
        llm = MockLLM()
        extract_entity = ExtractEntity(
            None, entity_extractor=OpenAIEntityExtractor("title", llm=llm, field="properties.entity.author")
        )
        out_doc = extract_entity.run(self.doc)
        assert out_doc.properties.get("title") == "Jack Black"

    def test_extract_entity_with_context_llm(self, mocker):
        llm = MockLLM()
        context = Context(
            params={
                "default": {"llm": llm},
            }
        )
        extract_entity = ExtractEntity(None, context=context, entity_extractor=OpenAIEntityExtractor("title"))
        out_doc = extract_entity.run(self.doc)
        assert out_doc.properties.get("title") == "title1"

    def test_extract_entity_few_shot(self, mocker):
        llm = MockLLM()
        extract_entity = ExtractEntity(
            None, entity_extractor=OpenAIEntityExtractor("title", llm=llm, prompt_template="title")
        )
        out_doc = extract_entity.run(self.doc)
        assert out_doc.properties.get("title") == "title2"

    def test_extract_entity_document_field_messages(self, mocker):
        llm = MockLLM()
        extract_entity = ExtractEntity(
            None,
            entity_extractor=OpenAIEntityExtractor(
                "title", llm=llm, use_elements=False, prompt=[], field="properties.path"
            ),
        )
        out_doc = extract_entity.run(self.doc)

        assert out_doc.properties.get("title") == "alt_title"

    def test_extract_entity_document_field_string(self, mocker):
        llm = MockLLM()
        extract_entity = ExtractEntity(
            None,
            entity_extractor=OpenAIEntityExtractor(
                "title", llm=llm, use_elements=False, prompt="", field="properties.path"
            ),
        )
        out_doc = extract_entity.run(self.doc)
        assert out_doc.properties.get("title") == "alt_title"

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
                    Element(properties={"_element_index": 4}, text_representation="test1"),
                    Element(properties={"_element_index": 9}, text_representation="test2"),
                ],
            ),
            Document(
                doc_id="doc_3",
                elements=[
                    Element(properties={"_element_index": 1}, text_representation="test2"),
                ],
            ),
            Document(doc_id="doc_4", text_representation="empty elements, maybe an exploded doc", elements=[]),
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
        entity_docset.show()
        taken = entity_docset.take()
        assert len(taken) == 4
        assert len(taken[0].elements) == 2
        assert (taken[1].elements[0]["properties"]["_element_index"]) == 9
        assert (taken[1].elements[1]["properties"]["_element_index"]) == 4
        assert (taken[0].elements[1]["properties"]["_element_index"]) == 2

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
            max_tokens=10,  # Low token limit to test windowing
        )

        entity_docset = docset.extract_entity(
            entity_extractor=entity_extractor,
        )
        entity_docset.show()
        taken = entity_docset.take()
        for ele in taken:
            print(ele.properties)
        assert taken[0].properties[f"{new_field}_source_element_index"] == {0, 1, 2}
        assert taken[1].properties[f"{new_field}_source_element_index"] == {2}
        print(taken[0].properties[new_field])
        assert taken[0].properties[new_field] == "4"
        assert taken[1].properties[new_field] == "5"
        assert taken[0].elements[0]["properties"]["_autogen_LLMExtractEntityOutput_source_element_index"] == {0, 1, 2}
        assert taken[0].elements[1]["properties"]["_autogen_LLMExtractEntityOutput_source_element_index"] == {0, 1, 2}
        assert taken[0].elements[2]["properties"]["_autogen_LLMExtractEntityOutput_source_element_index"] == {0, 1, 2}
        assert taken[1].elements[0]["properties"]["_autogen_LLMExtractEntityOutput_source_element_index"] == {1}
        assert taken[1].elements[1]["properties"]["_autogen_LLMExtractEntityOutput_source_element_index"] == {2}
        assert taken[0].elements[0]["properties"]["_autogen_LLMExtractEntityOutput"] == "4"
        assert taken[0].elements[1]["properties"]["_autogen_LLMExtractEntityOutput"] == "4"
        assert taken[0].elements[2]["properties"]["_autogen_LLMExtractEntityOutput"] == "4"
        assert taken[1].elements[0]["properties"]["_autogen_LLMExtractEntityOutput"] == "None"
        assert taken[1].elements[1]["properties"]["_autogen_LLMExtractEntityOutput"] == "5"
