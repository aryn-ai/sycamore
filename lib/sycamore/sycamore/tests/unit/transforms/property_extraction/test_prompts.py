from sycamore.transforms.property_extraction.prompts import format_schema_v2
from sycamore.schema import (
    DateProperty,
    DateTimeProperty,
    IntProperty,
    SchemaV2,
    NamedProperty,
    ObjectProperty,
    ArrayProperty,
    StringProperty,
    ChoiceProperty,
    RegexValidator,
)
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.tests.unit.transforms.test_summarize import occurrences


complicated_schema = SchemaV2(
    properties=[
        NamedProperty(name="a", type=StringProperty(extraction_instructions="extract the letter a")),
        NamedProperty(
            name="b",
            type=ObjectProperty(
                description="description of object B",
                properties=[
                    NamedProperty(name="c", type=ChoiceProperty(choices=["1", "2", "3"])),
                    NamedProperty(
                        name="d",
                        type=ArrayProperty(
                            description="description of array D",
                            extraction_instructions="extract a table called D",
                            item_type=StringProperty(required=True, description="thing in array"),
                        ),
                    ),
                    NamedProperty(
                        name="e",
                        type=ArrayProperty(
                            description="Outer array of E",
                            examples=[[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]],
                            item_type=ArrayProperty(
                                description="Inner array of E",
                                examples=[[1, 2, 3], [4, 5, 6]],
                                item_type=IntProperty(required=True, examples=[1, 4, 9]),
                            ),
                        ),
                    ),
                ],
            ),
        ),
        NamedProperty(name="f", type=StringProperty(validators=[RegexValidator(regex=r"[1-9][0-9]{0,9}")])),
        NamedProperty(name="g", type=DateProperty()),
        NamedProperty(name="h", type=DateTimeProperty(required=True)),
    ]
)

expected_formatted_schema = """\
{

  // Extraction Instructions: extract the letter a
  a: string?

  // Description: description of object B
  b: object {

    c: enum { "1", "2", "3" }?

    // Description: description of array D
    // Extraction Instructions: extract a table called D
    d: array [

      // Description: thing in array
      string
    ]

    // Description: Outer array of E
    // Examples:
    //   - [[1, 2, 3], [4, 5, 6]]
    //   - [[2, 3, 4], [5, 6, 7]]
    e: array [

      // Description: Inner array of E
      // Examples:
      //   - [1, 2, 3]
      //   - [4, 5, 6]
      array [

        // Examples:
        //   - 1
        //   - 4
        //   - 9
        int
      ]
    ]
  }

  // Constraints:
  //   - must match the regex: `[1-9][0-9]{0,9}`
  // Invalid Guesses:
  //   - "0123"
  f: string?

  g: date? (YYYY-MM-DD)

  h: datetime (YYYY-MM-DDTHH:MM:SS)
}"""


def test_format_schema_v2():
    rpt = RichProperty.from_prediction({"f": "0123"})
    rpt.value["f"].invalid_guesses = ["0123"]
    f = format_schema_v2(complicated_schema, rpt)
    assert f == expected_formatted_schema


def test_format_schema_v2_zipwith_array():
    sch = SchemaV2(
        properties=[
            NamedProperty(
                name="a",
                type=ArrayProperty(
                    item_type=ObjectProperty(properties=[NamedProperty(name="b", type=StringProperty())])
                ),
            )
        ]
    )

    rpt = RichProperty.from_prediction({"a": [{"b": "b1"}, {"b": "b2"}]})

    f = format_schema_v2(sch, rpt)
    assert occurrences(f, "object") == 1
