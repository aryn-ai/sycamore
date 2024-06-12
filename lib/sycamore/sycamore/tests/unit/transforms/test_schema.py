import random
import string

from ray.util import inspect_serializability

from sycamore.llms.prompts import SchemaZeroShotGuidancePrompt
from sycamore.data import Document, Element
from sycamore.llms.llms import LLM, FakeLLM
from sycamore.transforms.extract_schema import ExtractBatchSchema, SchemaExtractor
from sycamore.transforms.extract_schema import OpenAISchemaExtractor, OpenAIPropertyExtractor
from sycamore.utils.ray_utils import check_serializable


class TrivialExtractor(SchemaExtractor):
    def __init__(self):
        super().__init__("foo")

    def extract_schema(self, document: Document) -> Document:
        return document


class TestSchema:
    def test_serializable(self, mocker):
        t = TrivialExtractor()
        check_serializable(t)

        llm = FakeLLM()
        o = OpenAISchemaExtractor("Foo", llm)
        check_serializable(o)

        llm = mocker.Mock(spec=LLM)
        mocker.patch.object(llm, "generate")
        (ok, log) = inspect_serializability(llm)
        assert not ok

    def test_extract_schema(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '```json {"accidentNumber": "string"}```'

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
                "prompt": SchemaZeroShotGuidancePrompt(),
                "entity": class_name,
                "max_num_properties": max_num_properties,
                "query": schema_extractor._prompt_formatter(doc.elements),
            }
        )

    def test_extract_batch_schema(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '```json {"accidentNumber": "string"}```'
        schema_extractor = OpenAISchemaExtractor("AircraftIncident", llm)

        dicts = [
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        docs = [Document(d) for d in dicts]

        batch_extractor = ExtractBatchSchema(None, schema_extractor)
        dicts = [batch_extractor.run(d).data for d in docs]

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
        generate.return_value = '```json {"accidentNumber": "FTW95FA129"}```'

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
        generate.return_value = '{"accidentNumber": "FTW95FA129"}'

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
