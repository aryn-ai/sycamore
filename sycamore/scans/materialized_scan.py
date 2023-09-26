from typing import Union

from pandas import DataFrame
from pyarrow import Table
from ray.data import Dataset, from_arrow, from_items, from_pandas

from sycamore.plan_nodes import Scan
from sycamore.data import Document


class MaterializedScan(Scan):
    """A base scan class for materialized data
    e.g. arrow table, pandas dataframe, python dict list or even spark
    dataset
    """

    def __init__(self, **resource_args):
        super().__init__(**resource_args)


class ArrowScan(MaterializedScan):
    def __init__(self, tables: Union["Table", bytes, list[Union[Table, bytes]]], **resource_args):
        super().__init__(**resource_args)
        self._tables = tables

    def execute(self) -> Dataset:
        return from_arrow(tables=self._tables).map(lambda dict: Document(dict))

    def format(self):
        return "arrow"


class DocScan(MaterializedScan):
    def __init__(self, docs: list[Document], **resource_args):
        super().__init__(**resource_args)
        self._dicts = docs

    def execute(self) -> Dataset:
        return from_items(items=self._dicts)

    def format(self):
        return "document"


class PandasScan(MaterializedScan):
    def __init__(self, dfs: Union[DataFrame, list[DataFrame]], **resource_args):
        super().__init__(**resource_args)
        self._dfs = dfs

    def execute(self) -> Dataset:
        return from_pandas(dfs=self._dfs).map(lambda dict: Document(dict))

    def format(self):
        return "pandas"
