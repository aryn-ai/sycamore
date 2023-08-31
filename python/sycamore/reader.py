from typing import (List, Union)

from pandas import DataFrame
from pyarrow import Table

from sycamore import (Context, DocSet)
from sycamore.data import Document
from sycamore.execution.scans import (
    ArrowScan, BinaryScan, DocScan, PandasScan)


class DocSetReader:
    def __init__(self, context: Context):
        self._context = context

    def binary(
            self,
            paths: Union[str, List[str]],
            binary_format: str) -> DocSet:
        scan = BinaryScan(paths, binary_format=binary_format)
        return DocSet(self._context, scan)

    def arrow(
            self,
            tables: Union["Table", bytes, List[Union["Table", bytes]]]) \
            -> DocSet:
        scan = ArrowScan(tables)
        return DocSet(self._context, scan)

    def document(self, docs: List[Document]) -> DocSet:
        scan = DocScan(docs)
        return DocSet(self._context, scan)

    def pandas(self, dfs: Union["DataFrame", List["DataFrame"]]) -> DocSet:
        scan = PandasScan(dfs)
        return DocSet(self._context, scan)
