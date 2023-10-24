import pytest
from pandas import DataFrame
from pyarrow import Table

from sycamore.scans import ArrowScan, DocScan, PandasScan
from sycamore.data import Document


class TestMaterializedScan:
    dicts = [{"doc_id": 1, "type": "hello, world!"}, {"doc_id": 2, "type": "你好，世界！"}]

    @pytest.mark.parametrize(
        "scanner",
        [ArrowScan(Table.from_pylist(dicts)), DocScan([Document(d) for d in dicts]), PandasScan(DataFrame(dicts))],
    )
    def test_materialized_scan(self, scanner):
        ds = scanner.execute()
        assert ds.schema().names == ["doc"]
