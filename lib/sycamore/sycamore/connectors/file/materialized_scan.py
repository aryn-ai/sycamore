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

    def execute(self, **kwargs) -> Dataset:
        return from_arrow(tables=self._tables).map(lambda dict: {"doc": Document(dict).serialize()})

    def format(self):
        return "arrow"


class DocScan(MaterializedScan):
    def __init__(self, docs: list[Document], **resource_args):
        super().__init__(**resource_args)
        if not isinstance(docs, list):
            raise ValueError("docs should be a list")
        for d in docs:
            if not isinstance(d, Document):
                raise ValueError("each entry in list should be a document")
        self._docs = docs

    def execute(self, **kwargs) -> Dataset:
        return from_items(items=[{"doc": doc.serialize()} for doc in self._docs])

    def local_source(self) -> list[Document]:
        return self._docs

    def format(self):
        return "document"


class PandasScan(MaterializedScan):
    def __init__(self, dfs: Union[DataFrame, list[DataFrame]], **resource_args):
        super().__init__(**resource_args)
        self._dfs = dfs

    def execute(self, **kwargs) -> Dataset:
        return from_pandas(dfs=self._dfs).map(lambda dict: {"doc": Document(dict).serialize()})

    def format(self):
        return "pandas"
