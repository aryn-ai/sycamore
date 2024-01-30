import random
import string

import ray.data

from llms.prompts import SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT
from sycamore.data import Document, Element
from sycamore.llms import LLM
from sycamore.plan_nodes import Node
from sycamore.transforms.extract_schema import ExtractBatchSchema
from sycamore.transforms.extract_schema import OpenAISchemaExtractor, OpenAIPropertyExtractor


class TestSchema:
    def test_extract_schema(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": '```json {"accidentNumber": "string"}```'}

        num_of_elements = 10
        max_num_properties = 2
        class_name = "AircraftIncident"

        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        schema_extractor = OpenAISchemaExtractor(
            class_name, llm, num_of_elements=num_of_elements, max_num_properties=max_num_properties
        )
        doc = schema_extractor.extract_schema(doc)

        ground_truth = {
            "_schema": {
                "accidentNumber": "string",
            },
            "_schema_class": "AircraftIncident",
        }
        print(doc.properties)
        assert doc.properties == ground_truth
        generate.assert_called_once_with(
            prompt_kwargs={
                "prompt": SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT,
                "entity": class_name,
                "max_num_properties": max_num_properties,
                "query": schema_extractor._prompt_formatter(doc.elements),
            }
        )

    def test_extract_batch_schema(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": '```json {"accidentNumber": "string"}```'}
        schema_extractor = OpenAISchemaExtractor("AircraftIncident", llm)

        node = mocker.Mock(spec=Node)
        dicts = [
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        input_dataset = ray.data.from_items([{"doc": Document(dict).serialize()} for dict in dicts])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset

        batch_extractor = ExtractBatchSchema(node, schema_extractor)
        output_dataset = batch_extractor.execute()
        dicts = [Document.from_row(doc).data for doc in output_dataset.take()]

        ground_truth = {
            "_schema": {
                "accidentNumber": "string",
            },
            "_schema_class": "AircraftIncident",
        }
        assert dicts[0]["properties"] == ground_truth and dicts[1]["properties"] == ground_truth

    def test_extract_properties(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": '```json {"accidentNumber": "FTW95FA129"}```'}

        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]
        doc.properties = {
            "_schema": {
                "accidentNumber": "string",
            },
            "_schema_class": "AircraftIncident",
        }

        property_extractor = OpenAIPropertyExtractor(llm)
        doc = property_extractor.extract_properties(doc)

        assert doc.properties["entity"]["accidentNumber"] == "FTW95FA129"

    def test_extract_properties_explicit_json(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = {"answer": '{"accidentNumber": "FTW95FA129"}'}

        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]
        doc.properties = {
            "_schema": {
                "accidentNumber": "string",
            },
            "_schema_class": "AircraftIncident",
        }

        property_extractor = OpenAIPropertyExtractor(llm)
        doc = property_extractor.extract_properties(doc)

        assert doc.properties["entity"]["accidentNumber"] == "FTW95FA129"
