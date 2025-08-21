from sycamore.transforms.property_extraction.prompts import format_schema_v2
from sycamore.schema import (
    IntProperty,
    SchemaV2,
    NamedProperty,
    ObjectProperty,
    ArrayProperty,
    StringProperty,
    ChoiceProperty,
    RegexValidator,
)

complicated_schema = SchemaV2(
    properties=[
        NamedProperty(name="a", type=StringProperty(extraction_instructions="extract the letter a")),
        NamedProperty(
            name="b",
            type=ObjectProperty(
                properties=[
                    NamedProperty(name="c", type=ChoiceProperty(choices=["1", "2", "3"])),
                    NamedProperty(
                        name="d",
                        type=ArrayProperty(item_type=StringProperty(required=True, description="thing in array")),
                    ),
                    NamedProperty(
                        name="e",
                        type=ArrayProperty(
                            item_type=ArrayProperty(item_type=IntProperty(required=True, examples=[1, 4, 9]))
                        ),
                    ),
                ]
            ),
        ),
        NamedProperty(name="f", type=StringProperty(validators=[RegexValidator(regex=r"[1-9][0-9]{0,9}")])),
    ]
)

expected_formatted_schema = """\
{
  a: { type: string, extraction_instructions: extract the letter a },
  b: object {
    c: enum { "1", "2", "3" },
    d: array [
      { type: string, description: thing in array }
    ],
    e: array [
      array [
        { type: int, examples: [1, 4, 9] }
      ]
    ]
  },
  f: { type: string, constraints: [ must match the regex: `[1-9][0-9]{0,9}` ] }
}"""


def test_format_schema():
    f = format_schema_v2(complicated_schema)
    assert f == expected_formatted_schema
