from datetime import date
import pyarrow as pa

from sycamore.data import Document
from sycamore.utils.pyarrow import schema_to_pyarrow, docs_to_pyarrow
from sycamore.schema import SchemaV2, make_property, make_named_property


def test_scalar_schema_to_pyarrow():
    schema = SchemaV2(
        properties=[
            make_named_property(name="bool", type="bool"),
            make_named_property(name="int", type="int"),
            make_named_property(name="float", type="float"),
            make_named_property(name="string", type="string"),
            make_named_property(name="date", type="date"),
            make_named_property(name="datetime", type="datetime"),
        ]
    )

    pa_schema = schema_to_pyarrow(schema)
    assert pa_schema.types == [pa.bool_(), pa.int64(), pa.float64(), pa.string(), pa.date32(), pa.date64()]


def test_choice_schema_to_pyarrow():
    schema = SchemaV2(
        properties=[
            make_named_property(name="choice1", type="choice", choices=["choice1", "choice2"]),
            make_named_property(name="choice2", type="choice", choices=[1, 2]),
            make_named_property(name="choice3", type="choice", choices=[date(2023, 1, 1), date(2024, 1, 1)]),
        ]
    )

    pa_schema = schema_to_pyarrow(schema)
    assert pa_schema.types == [pa.string(), pa.int64(), pa.date32()]


def test_nested_schema_to_pyarrow():
    schema = SchemaV2(
        properties=[
            make_named_property(name="name", type="string", required=True),
            make_named_property(name="age", type="int"),
            make_named_property(
                name="children",
                type="array",
                item_type=make_property(
                    type="object",
                    properties=[
                        make_named_property(name="name", type="string", required=True),
                        make_named_property(name="age", type="int", required=True),
                    ],
                ),
            ),
        ]
    )

    pa_schema = schema_to_pyarrow(schema)
    types = pa_schema.types
    assert types[0] == pa.string()
    assert types[1] == pa.int64()
    assert types[2] == pa.list_(pa.struct([pa.field("name", pa.string()), pa.field("age", pa.int64())]))


def test_doc_to_pyarrow() -> None:
    schema = SchemaV2(
        properties=[
            make_named_property(name="name", type="string", required=True),
            make_named_property(name="age", type="int"),
            make_named_property(
                name="children",
                type="array",
                item_type=make_property(
                    type="object",
                    properties=[
                        make_named_property(name="name", type="string", required=True),
                        make_named_property(name="age", type="int", required=True),
                    ],
                ),
            ),
        ]
    )

    pa_schema = schema_to_pyarrow(schema)

    people = [
        {
            "name": "Alice",
            "age": 40,
            "children": [{"name": "Bob", "age": 10}, {"name": "Charlie", "age": 8}],
        },
        {
            "name": "Swathi",
            "age": 32,
            "children": [
                {"name": "Keshav", "age": 4},
            ],
        },
        {"name": "John", "age": 45, "children": None},
    ]

    docs = [Document({"properties": {"entity": person}}) for person in people]

    table = docs_to_pyarrow(docs, pa_schema)
    assert table.to_pylist() == people
