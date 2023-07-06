import ray.data
from pandas import DataFrame
from pyarrow import Table
from ray.data import Dataset
from shannon.execution import Scan
from shannon.data import Document
from typing import (List, Union)


class MaterializedScan(Scan):
    """A base scan class for materialized data
     e.g. arrow table, pandas dataframe, python dict list or even spark
     dataset
    """
    def __init__(self, **resource_args):
        super().__init__(**resource_args)


class ArrowScan(MaterializedScan):
    def __init__(
            self,
            tables: Union["Table", bytes, List[Union["Table", bytes]]],
            **resource_args):
        super().__init__(**resource_args)
        self._tables = tables

    def execute(self) -> "Dataset":
        return ray.data.from_arrow(tables=self._tables)

    def format(self):
        return "arrow"


class DocScan(MaterializedScan):
    def __init__(self, docs: List[Document], **resource_args):
        super().__init__(**resource_args)
        self._dicts = docs

    def execute(self) -> "Dataset":
        return ray.data.from_items(items=self._dicts)

    def format(self):
        return "document"


class PandasScan(MaterializedScan):
    def __init__(
            self,
            dfs: Union["DataFrame", List["DataFrame"]],
            **resource_args):
        super().__init__(**resource_args)
        self._dfs = dfs

    def execute(self) -> "Dataset":
        return ray.data.from_pandas(dfs=self._dfs)

    def format(self):
        return "pandas"
