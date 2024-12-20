import sycamore
from sycamore import ExecMode
from sycamore.data import Document
from sycamore.schema import Schema, SchemaField
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.extract_schema import LLMPropertyExtractor


def test_extract_properties_from_schema():
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

    schema = Schema(
        fields=[
            SchemaField(
                name="name",
                field_type="str",
                description="This is the name of an entity",
                examples=["Mark", "Ollie", "Winston"],
            ),
            SchemaField(name="age", field_type="int", default=999),
            SchemaField(name="date", field_type="str", description="Any date in the doc in YYYY-MM-DD format"),
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
    property_extractor = LLMPropertyExtractor(OpenAI(OpenAIModels.GPT_4O), schema=schema)

    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    docs = ctx.read.document(docs)
    docs = docs.extract_properties(property_extractor)

    taken = docs.take_all()

    assert taken[0].properties["entity"]["name"] == "Vinayak"
    assert taken[0].properties["entity"]["age"] == 74
    assert taken[0].properties["entity"]["from_location"] == "Honolulu, HI", "Invalid location extracted or formatted"
    assert taken[0].properties["entity"]["date"] == "1923-02-24"

    assert taken[1].properties["entity"]["name"] is None, "Default None value not being used correctly"
    assert taken[1].properties["entity"]["age"] == 999, "Default value not being used correctly"
    assert taken[1].properties["entity"]["from_location"] == "New Delhi"
    assert taken[1].properties["entity"]["date"] == "2014-01-11"
