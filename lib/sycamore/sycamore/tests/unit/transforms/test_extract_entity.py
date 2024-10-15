from typing import Optional
import logging


from sycamore import Context
from sycamore.data import Document
from sycamore.transforms import ExtractEntity
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    EntityExtractorFewShotGuidancePrompt,
    EntityExtractorZeroShotGuidancePrompt,
)


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
