from typing import Optional

import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms import ExtractEntity
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import LLM


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="mock_model")

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
        if prompt_kwargs == {"messages": [{"role": "user", "content": "s3://path"}]} and llm_kwargs == {}:
            return "alt_title"
        return "title"

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
                    "properties": {"coordinates": [(1, 2)], "page_number": 1},
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
        node = mocker.Mock(spec=Node)
        llm = MockLLM()
        extract_entity = ExtractEntity(node, entity_extractor=OpenAIEntityExtractor("title", llm=llm))
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = extract_entity.execute()
        assert Document.from_row(output_dataset.take(1)[0]).properties.get("title") == "title"

    def test_extract_entity_few_shot(self, mocker):
        node = mocker.Mock(spec=Node)
        llm = MockLLM()
        extract_entity = ExtractEntity(
            node, entity_extractor=OpenAIEntityExtractor("title", llm=llm, prompt_template="title")
        )
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = extract_entity.execute()
        assert Document.from_row(output_dataset.take(1)[0]).properties.get("title") == "title"

    def test_extract_entity_document_field_messages(self, mocker):
        node = mocker.Mock(spec=Node)
        llm = MockLLM()
        extract_entity = ExtractEntity(
            node,
            entity_extractor=OpenAIEntityExtractor(
                "title", llm=llm, use_elements=False, prompt=[], field="properties.path"
            ),
        )
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = extract_entity.execute()
        assert Document.from_row(output_dataset.take(1)[0]).properties.get("title") == "alt_title"

    def test_extract_entity_document_field_string(self, mocker):
        node = mocker.Mock(spec=Node)
        llm = MockLLM()
        extract_entity = ExtractEntity(
            node,
            entity_extractor=OpenAIEntityExtractor(
                "title", llm=llm, use_elements=False, prompt="", field="properties.path"
            ),
        )
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = extract_entity.execute()
        assert Document.from_row(output_dataset.take(1)[0]).properties.get("title") == "alt_title"
