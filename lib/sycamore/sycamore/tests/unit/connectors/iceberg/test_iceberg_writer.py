from pyiceberg.catalog import load_catalog
import pytest

import sycamore
from sycamore.connectors.iceberg.iceberg_writer import IcebergWriter
from sycamore.data import Document
from sycamore.schema import SchemaV2, make_named_property


@pytest.mark.parametrize("mode", [sycamore.EXEC_LOCAL, sycamore.EXEC_RAY])
def test_iceberg_writer_ray(mode, tmp_path) -> None:
    catalog_options = {
        "type": "sql",
        "uri": f"sqlite:////{str(tmp_path / 'iceberg_test.db')}",
    }

    schema = SchemaV2(
        properties=[
            make_named_property(name="field1", type="string"),
            make_named_property(name="field2", type="int"),
        ]
    )

    docs = [
        Document(properties={"entity": {"field1": "value1", "field2": 123}}),
        Document(properties={"entity": {"field1": "value2", "field2": 456}}),
    ]
    table_id = "test_namespace.simple_table"

    ctx = sycamore.init(exec_mode=mode)
    (
        ctx.read.document(docs)
        .transform(
            IcebergWriter,
            catalog_kwargs=catalog_options,
            schema=schema,
            table_identifier="test_namespace.simple_table",
            location=str(tmp_path),
        )
        .execute()
    )

    table = load_catalog(**catalog_options).load_table(table_id)
    table_dict = table.scan().to_arrow().to_pydict()

    assert table_dict == {"field1": ["value1", "value2"], "field2": [123, 456]}
