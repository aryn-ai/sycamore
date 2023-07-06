import pytest
from pandas import DataFrame
from pyarrow import Table
from shannon.execution.scans import (ArrowScan, DocScan, PandasScan)
from shannon.data import Document


class TestMaterializedScan:

    dicts = [
        {'int': 1, 'float': 3.14, 'str': 'hello, world!'},
        {'int': 2, 'float': 1.61, 'str': '你好，世界！'}]

    @pytest.mark.parametrize("scanner", [
        ArrowScan(Table.from_pylist(dicts)),
        DocScan([Document(d) for d in dicts]),
        PandasScan(DataFrame(dicts))
    ])
    def test_materialized_scan(self, scanner):
        ds = scanner.execute()
        assert (ds.schema().names == ['int', 'float', 'str'])
