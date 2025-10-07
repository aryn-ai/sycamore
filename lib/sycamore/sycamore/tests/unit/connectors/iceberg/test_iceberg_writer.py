import os
import time

from pyiceberg.catalog import load_catalog


from sycamore.connectors.iceberg.iceberg_writer import IcebergWriter
from sycamore.data import Document
from sycamore.schema import SchemaV2, make_property, make_named_property


def test_iceberg_writer(tmp_path) -> None:
    catalog_options = {
        "type": "in-memory",
        "warehouse": f"file://{str(tmp_path / 'warehouse2')}"
    }

    catalog_options = {
        "type": "rest",
        "uri": "https://dbc-75b31638-df31.cloud.databricks.com/api/2.1/unity-catalog/iceberg-rest",
        "warehouse": "aryn-databricks",
        "token": os.environ["DATABRICKS_TOKEN"],
    }
    


    schema = SchemaV2(properties=[
        make_named_property(name="name", type="string", required=True),
        make_named_property(name="age", type="int"),
        make_named_property(name="children", type="array", item_type=make_property(
            type="object",
            properties=[
                make_named_property(name="name", type="string"),
                make_named_property(name="age", type="int"),
            ]
        ))
    ])
    

    doc = Document({
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
    })

    
    
    writer = IcebergWriter(
        child = None,
        catalog_kwargs=catalog_options,
        schema=schema,
        table_identifier="test_namespace.my_table",
    )

    res = writer.local_execute([doc])
    
    
    # print(tmp_path)
    # time.sleep(5)  wait for the table to be created

    # catalog = load_catalog(**catalog_options)
    # print(catalog.list_namespaces())

    table = writer.table
    #catalog.load_table("test_namespace.my_table")
    pyarrow_table = table.scan().to_arrow()
    out = pyarrow_table.to_pydict()
    print(out)
