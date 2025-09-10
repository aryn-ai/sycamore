import pytest
import sycamore
from sycamore import ExecMode
from sycamore.data import Document, Element
from sycamore.schema import (
    NamedProperty,
    SchemaV2,
    StringProperty,
    IntProperty,
    DateProperty,
    make_named_property,
)
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.anthropic import Anthropic, AnthropicModels
from sycamore.transforms.extract_schema import LLMPropertyExtractor
from sycamore.transforms.embed import OpenAIEmbedder
from sycamore.llms.llms import LLMMode
from sycamore.data.document import split_data_metadata


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

    schema = SchemaV2(
        properties=[
            make_named_property(
                name="person_name",
                type="string",
                description="name of the person, return a tuple with value and pagenumber",
            ),
            make_named_property(
                name="profession",
                type="string",
                description="profession of the person, return a tuple of value and pagenumber",
            ),
        ]
    )

    llm = OpenAI(OpenAIModels.GPT_4_1)
    embedder = OpenAIEmbedder("text-embedding-3-small")

    # Create schema
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

    schema = SchemaV2(
        properties=[
            make_named_property(
                name="name",
                type="string",
                description="This is the name of an entity",
                examples=["Mark", "Ollie", "Winston"],
                default="None",
            ),
            make_named_property(name="age", type="int", default=999),
            make_named_property(
                name="date", type="string", description="Any date in the doc, extracted in YYYY-MM-DD format"
            ),
            make_named_property(
                name="from_location",
                type="string",
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


@pytest.mark.parametrize("llm", llms)
def test_extract(llm):
    docs = get_docs()
    for d in docs:
        d.elements = [Element(d)]
        d.elements[0].element_index = 0

    schema = SchemaV2(
        properties=[
            NamedProperty(
                name="name",
                type=StringProperty(description="This is the name of an entity", examples=["Mark", "Ollie", "Winston"]),
            ),
            NamedProperty(name="age", type=IntProperty()),
            NamedProperty(
                name="date", type=DateProperty(description="Any date in the doc, extracted in YYYY-MM-DD format")
            ),
            NamedProperty(
                name="from_location",
                type=StringProperty(
                    description="This is the location the entity is from. "
                    "If it's a US location and explicitly states a city and state, format it as 'City, State' "
                    "The state is abbreviated in it's standard 2 letter form.",
                    examples=["Ann Arbor, MI", "Seattle, WA", "New Delhi"],
                ),
            ),
        ]
    )
    ctx = sycamore.init(exec_mode=ExecMode.RAY)
    docs = ctx.read.document(docs)
    docs = docs.extract(schema, llm)

    taken = docs.take_all(include_metadata=True)

    real, meta = split_data_metadata(taken)
    real.sort(key=lambda d: d.doc_id or "")
    assert len(real) == 2
    assert real[0].properties["entity_metadata"]["name"].value == "Vinayak"
    assert real[0].properties["entity"]["age"] == 74
    assert (
        real[0].properties["entity_metadata"]["from_location"].value == "Honolulu, HI"
    ), "Invalid location extracted or formatted"
    assert real[0].properties["entity_metadata"]["date"].value == "1923-02-24"

    assert real[1].properties["entity"]["name"] is None
    assert real[1].properties["entity"]["age"] is None
    assert real[1].properties["entity"]["from_location"] == "New Delhi"
    assert real[1].properties["entity"]["date"] == "2014-01-11"

    llm_meta = [m for m in meta if "lineage_links" not in m.metadata]
    assert len(llm_meta) == 2
    assert llm_meta[0].metadata["usage"]["prompt_tokens"] > 0
    assert llm_meta[0].metadata["usage"]["completion_tokens"] > 0
    assert llm_meta[1].metadata["usage"]["prompt_tokens"] > 0
    assert llm_meta[1].metadata["usage"]["completion_tokens"] > 0
