from sycamore.data import Document
from sycamore.utils.pyarrow import schema_to_pyarrow, docs_to_pyarrow
from sycamore.schema import SchemaV2, make_property, make_named_property

def test_schema_to_pyarrow() -> None:
    schema = SchemaV2(properties=[
        make_named_property(name="name", type="string", required=True),
        make_named_property(name="age", type="int"),
        make_named_property(name="children", type="array", item_type=make_property(
            type="array", item_type=make_property(
                type="object",
                properties=[
                    make_named_property(name="name", type="string", required=True),
                    make_named_property(name="age", type="int", required=True),
                ]
            )
        ))
    ])

    pa_schema = schema_to_pyarrow(schema)

    print(pa_schema)


    
def test_doc_to_pyarrow() -> None:
    schema = SchemaV2(properties=[
        make_named_property(name="name", type="string", required=True),
        make_named_property(name="age", type="int"),
        make_named_property(name="children", type="array", item_type=make_property(
            type="object",
            properties=[
                make_named_property(name="name", type="string", required=True),
                make_named_property(name="age", type="int", required=True),
            ]
        ))
    ])
                            
    pa_schema = schema_to_pyarrow(schema)
    print(pa_schema)
    
    docs = [
        Document({
            "properties": {
                "entity": {
                    "name": "Alice",
                    "age": 40,
                    "children": [
                        {"name": "Bob", "age": 10},
                        {"name": "Charlie", "age": 8}
                    ]
                }
            }
        }),
        Document({
            "properties": {
                "entity": {
                    "name": "Swathi",
                    "age": 32,
                    "children": [
                        {"name": "Keshav", "age": 4},
                    ]
                }
            }
        }),
        Document({
            "properties": {
                "entity": {
                    "name": "John",
                    "age": 45,
                    "children": NOne
                }
            }
        }),
    ]

    table = docs_to_pyarrow(docs, pa_schema)

    print(table.to_pylist())
