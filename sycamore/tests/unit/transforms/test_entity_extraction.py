import ray.data

from sycamore.plan_nodes import Node
from sycamore.transforms import ExtractEntity
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import OpenAI


class TestEntityExtraction:
    doc = {
        "doc_id": "doc_id",
        "type": "pdf",
        "content": {"binary": None, "text": "text"},
        "parent_id": None,
        "properties": {"path": "s3://path"},
        "embedding": {"binary": None, "text": None},
        "elements": {
            "array": [
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
            ]
        },
    }

    def test_extract_entity_zero_shot(self, mocker):
        node = mocker.Mock(spec=Node)
        llm = OpenAI("openAI", "mockAPIKey")
        extract_entity = ExtractEntity(node, entity_extractor=OpenAIEntityExtractor("title", llm=llm))
        input_dataset = ray.data.from_items([self.doc])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset

        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": "title"}
        output_dataset = extract_entity.execute()

        assert output_dataset.take(1)[0].get("properties").get("title") == "title"

    def test_extract_entity_few_shot(self, mocker):
        node = mocker.Mock(spec=Node)
        llm = OpenAI("openAI", "mockAPIKey")
        extract_entity = ExtractEntity(
            node, entity_extractor=OpenAIEntityExtractor("title", llm=llm, prompt_template="title")
        )
        input_dataset = ray.data.from_items([self.doc])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset

        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": "title"}
        output_dataset = extract_entity.execute()

        assert output_dataset.take(1)[0].get("properties").get("title") == "title"
