import pytest
import sycamore
from sycamore import ExecMode
from sycamore.data import Document, Element
from sycamore.schema import Schema, SchemaField
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.anthropic import Anthropic, AnthropicModels
from sycamore.transforms.extract_schema import LLMPropertyExtractor
from sycamore.transforms.embed import OpenAIEmbedder
from sycamore.llms.llms import LLMMode


def get_docs():
    docs = [
        Document(
            {
                "doc_id": "doc_1",
                "text_representation": "My name is Vinayak & I'm a 74 year old software engineer from Honolulu Hawaii. "
                "This information was written on feb 24, 1923",
            }
        ),
        Document(
            {
                "doc_id": "doc_2",
                "text_representation": "is a strange case of anti-viral research found in New Delhi.\n "
                "info date: jan eleven 2014",
            }
        ),
    ]
    return docs


llms = [
    OpenAI(OpenAIModels.GPT_4O),
    Anthropic(AnthropicModels.CLAUDE_3_5_SONNET),
]


@pytest.mark.parametrize("llm", llms)
def test_extract_properties_from_dict_schema(llm):
    docs = get_docs()[:1]  # only validate first doc because of technique reliability
    schema = {"name": "str", "age": "int", "date": "str", "from_location": "str"}
    property_extractor = LLMPropertyExtractor(llm, schema=schema, schema_name="entity")

    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    docs = ctx.read.document(docs)
    docs = docs.extract_properties(property_extractor, llm_mode=LLMMode.SYNC)

    taken = docs.take_all(include_metadata=True)

    assert taken[0].properties["entity"]["name"] == "Vinayak"
    assert taken[0].properties["entity"]["age"] == 74
    assert "Honolulu" in taken[0].properties["entity"]["from_location"]

    assert len(taken) == 4
    assert taken[3].metadata["usage"]["prompt_tokens"] > 0
    assert taken[3].metadata["usage"]["completion_tokens"] > 0


@pytest.mark.parametrize("llm", llms)
def test_extract_metadata(llm):
    docs = get_docs()[:1]

    field1 = SchemaField(
        name="person_name",
        field_type="string",
        description="name of the person, return a tuple with value and pagenumber",
    )
    field2 = SchemaField(
        name="profession",
        field_type="string",
        description="profession of the person, return a tuple of value and pagenumber",
    )
    llm = OpenAI(OpenAIModels.GPT_4_1)
    embedder = OpenAIEmbedder("text-embedding-3-small")

    # Create schema
    schema = Schema(fields=[field1, field2])
    property_extractor = LLMPropertyExtractor(
        llm,
        schema=schema,
        metadata_extraction=True,
        group_size=1,
        embedder=embedder,
        clustering=False,
        schema_name="entity",
    )
    element = Element(
        {
            "type": "text",
            "text_representation": "My name is Vinayak & I'm a 74 year old software engineer from Honolulu Hawaii. ",
            "properties": {"page_number": 1},
        }
    )
    document = Document({"doc_id": "sample_doc_001", "elements": [element], "properties": {"title": "Sample Document"}})

    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    docs = ctx.read.document([document])
    docs = docs.extract_properties(property_extractor)

    taken = docs.take_all(include_metadata=True)
    assert taken[0].properties["entity"]["person_name"] == "Vinayak"
    assert taken[0].properties["entity"]["profession"] == "software engineer"
    assert taken[0].properties["entity_metadata"]["person_name"] == 1
    assert taken[0].properties["entity_metadata"]["profession"] == 1


@pytest.mark.parametrize("llm", llms)
def test_extract_properties_from_schema(llm):
    docs = get_docs()

    schema = Schema(
        fields=[
            SchemaField(
                name="name",
                field_type="str",
                description="This is the name of an entity",
                examples=["Mark", "Ollie", "Winston"],
                default="null",
            ),
            SchemaField(name="age", field_type="int", default=999),
            SchemaField(
                name="date", field_type="str", description="Any date in the doc, extracted in YYYY-MM-DD format"
            ),
            SchemaField(
                name="from_location",
                field_type="str",
                description="This is the location the entity is from. "
                "If it's a US location and explicitly states a city and state, format it as 'City, State' "
                "The state is abbreviated in it's standard 2 letter form.",
                examples=["Ann Arbor, MI", "Seattle, WA", "New Delhi"],
            ),
        ]
    )
    property_extractor = LLMPropertyExtractor(llm, schema=schema)

    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    docs = ctx.read.document(docs)
    docs = docs.extract_properties(property_extractor, llm_mode=LLMMode.SYNC)

    taken = docs.take_all(include_metadata=True)

    assert taken[0].properties["entity"]["name"] == "Vinayak"
    assert taken[0].properties["entity"]["age"] == 74
    assert taken[0].properties["entity"]["from_location"] == "Honolulu, HI", "Invalid location extracted or formatted"
    assert taken[0].properties["entity"]["date"] == "1923-02-24"

    assert taken[1].properties["entity"]["name"] == "None"  # Anthropic isn't generating valid JSON with null values.
    assert taken[1].properties["entity"]["age"] == 999, "Default value not being used correctly"
    assert taken[1].properties["entity"]["from_location"] == "New Delhi"
    assert taken[1].properties["entity"]["date"] == "2014-01-11"

    assert len(taken) == 6
    assert taken[4].metadata["usage"]["prompt_tokens"] > 0
    assert taken[4].metadata["usage"]["completion_tokens"] > 0
    assert taken[5].metadata["usage"]["prompt_tokens"] > 0
    assert taken[5].metadata["usage"]["completion_tokens"] > 0
