import datetime
import random
import string
from typing import Optional

from ray.util import inspect_serializability

from sycamore.data import Document, Element
from sycamore.llms.llms import LLM, FakeLLM, LLMMode
from sycamore.llms.prompts import RenderedPrompt
from sycamore.plan_nodes import Node
from sycamore.schema import Schema, SchemaField
from sycamore.transforms.base_llm import LLMMap
from sycamore.transforms.map import Map
from sycamore.transforms.extract_schema import ExtractBatchSchema, SchemaExtractor
from sycamore.transforms.extract_schema import LLMSchemaExtractor, LLMPropertyExtractor
from sycamore.utils.ray_utils import check_serializable


class TrivialExtractor(SchemaExtractor):
    def __init__(self):
        super().__init__("foo")

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        return child  # type: ignore

    def extract_schema(self, document: Document) -> Document:
        return document


class TestSchema:
    @staticmethod
    def llm(mocker):
        llm = mocker.Mock(spec=LLM)
        mode = mocker.patch.object(llm, "default_mode")
        mode.return_value = LLMMode.SYNC
        return llm

    def test_serializable(self, mocker):
        t = TrivialExtractor()
        check_serializable(t)

        llm = FakeLLM()
        o = LLMSchemaExtractor("Foo", llm)
        check_serializable(o)

        llm = TestSchema.llm(mocker)
        mocker.patch.object(llm, "generate")
        (ok, log) = inspect_serializability(llm)
        assert not ok

    def test_extract_schema(self, mocker):
        llm = TestSchema.llm(mocker)
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

        schema_extractor = LLMSchemaExtractor(
            class_name, llm, num_of_elements=num_of_elements, max_num_properties=max_num_properties
        )
        doc = schema_extractor.as_llm_map(None)._local_process([doc])[0]

        ground_truth = {
            "_schema": {
                "accidentNumber": "string",
            },
            "_schema_class": "AircraftIncident",
        }
        assert doc.properties == ground_truth
        generate.assert_called_once()
        ca = generate.call_args
        rp = ca.kwargs["prompt"]
        assert isinstance(rp, RenderedPrompt)
        messages = rp.messages
        assert len(messages) == 2
        assert f"ELEMENT None: {element1.text_representation}" in messages[1].content
        assert f"ELEMENT None: {element2.text_representation}" in messages[1].content

    def test_extract_batch_schema(self, mocker):
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '```json {"accidentNumber": "string"}```'
        schema_extractor = LLMSchemaExtractor("AircraftIncident", llm)

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
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '```json {"accidentNumber": "FTW95FA129", "location": "Fort Worth, TX"}```'

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
            "entity": {"weather": "sunny"},
        }

        property_extractor = LLMPropertyExtractor(llm)
        pe_map = property_extractor.as_llm_map(None)
        assert len(pe_map.children) == 1
        pe_llm_map = pe_map.children[0]
        assert isinstance(pe_llm_map, LLMMap)
        assert isinstance(pe_map, Map)

        docs = pe_llm_map.run([doc])
        doc = pe_map.run(docs[0])

        # doc = property_extractor.extract_properties(doc)

        assert doc.properties["entity"]["weather"] == "sunny"
        assert doc.properties["AircraftIncident"]["accidentNumber"] == "FTW95FA129"
        assert doc.properties["AircraftIncident"]["location"] == "Fort Worth, TX"

    def test_extract_properties_default_to_entity(self, mocker):
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '```json {"accidentNumber": "FTW95FA129", "location": "Fort Worth, TX"}```'

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
            "entity": {"weather": "sunny"},
        }

        property_extractor = LLMPropertyExtractor(llm)
        pe_map = property_extractor.as_llm_map(None)
        assert len(pe_map.children) == 1
        pe_llm_map = pe_map.children[0]
        assert isinstance(pe_llm_map, LLMMap)
        assert isinstance(pe_map, Map)

        docs = pe_llm_map.run([doc])
        doc = pe_map.run(docs[0])

        # doc = property_extractor.extract_properties(doc)

        assert doc.properties["entity"]["weather"] == "sunny"
        assert doc.properties["entity"]["accidentNumber"] == "FTW95FA129"
        assert doc.properties["entity"]["location"] == "Fort Worth, TX"

    def test_extract_properties_explicit_json(self, mocker):
        llm = TestSchema.llm(mocker)
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

        property_extractor = LLMPropertyExtractor(llm)
        pe_map = property_extractor.as_llm_map(None)
        assert len(pe_map.children) == 1
        pe_llm_map = pe_map.children[0]
        assert isinstance(pe_llm_map, LLMMap)
        assert isinstance(pe_map, Map)

        docs = pe_llm_map.run([doc])
        doc = pe_map.run(docs[0])

        assert doc.properties["AircraftIncident"]["accidentNumber"] == "FTW95FA129"

    def test_extract_properties_llm_say_none(self, mocker):
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "None"

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

        property_extractor = LLMPropertyExtractor(llm)
        docs = property_extractor.extract_docs([doc])
        doc = docs[0]

        assert len(doc.properties["AircraftIncident"]) == 0

    def test_extract_properties_fixed_json(self, mocker):
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = '{"accidentNumber": "FTW95FA129"}'

        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        property_extractor = LLMPropertyExtractor(
            llm, schema_name="AircraftIncident", schema={"accidentNumber": "string"}
        )
        pe_map = property_extractor.as_llm_map(None)
        assert len(pe_map.children) == 1
        pe_llm_map = pe_map.children[0]
        assert isinstance(pe_llm_map, LLMMap)
        assert isinstance(pe_map, Map)

        docs = pe_llm_map.run([doc])
        doc = pe_map.run(docs[0])

        assert doc.properties["AircraftIncident"]["accidentNumber"] == "FTW95FA129"

        # assert doc.properties["entity"]["accidentNumber"] == "FTW95FA129"

    def test_extract_properties_with_schema(self, mocker):
        llm = TestSchema.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = (
            '{"startDate": "2022-01-22 00:01:31", '
            '"endDate": "2022-01-24 00:01:59", '
            '"someOtherDate": "2024-01--1 00:01:01", '
            '"accidentNumber": "FTW95FA129", '
            '"latitude": "10.00353", '
            '"injuryCount": "5", '
            '"location": ["Fort Worth, TX", "Dallas, TX"]}'
        )

        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        schema = Schema(
            fields=[
                SchemaField(name="startDate", field_type="datetime"),
                SchemaField(name="endDate", field_type="date"),
                SchemaField(name="accidentNumber", field_type="str"),
                SchemaField(name="injuryCount", field_type="int"),
                SchemaField(name="latitude", field_type="float"),
                SchemaField(name="location", field_type="list"),
            ]
        )
        property_extractor = LLMPropertyExtractor(llm, schema=schema)
        pe_map = property_extractor.as_llm_map(None)
        assert len(pe_map.children) == 1
        pe_llm_map = pe_map.children[0]
        assert isinstance(pe_llm_map, LLMMap)
        assert isinstance(pe_map, Map)

        docs = pe_llm_map.run([doc])
        doc = pe_map.run(docs[0])

        assert doc.properties["entity"]["accidentNumber"] == "FTW95FA129"
        assert doc.properties["entity"]["startDate"] == datetime.datetime(2022, 1, 22, 0, 1, 31)
        assert doc.properties["entity"]["endDate"] == datetime.datetime(2022, 1, 24, 0, 1, 59)
        assert doc.properties["entity"]["someOtherDate"] == "2024-01--1 00:01:01"
        assert doc.properties["entity"]["injuryCount"] == 5
        assert doc.properties["entity"]["latitude"] == 10.00353
        assert doc.properties["entity"]["location"] == ["Fort Worth, TX", "Dallas, TX"]
