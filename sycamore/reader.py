from typing import Optional, Union

from pandas import DataFrame
from pyarrow import Table
from pyarrow.filesystem import FileSystem

from sycamore import Context, DocSet
from sycamore.data import Document
from sycamore.scans import ArrowScan, BinaryScan, DocScan, PandasScan
from sycamore.scans.file_scan import FileMetadataProvider


class DocSetReader:
    def __init__(self, context: Context):
        self._context = context

    def binary(
        self,
        paths: Union[str, list[str]],
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        **resource_args
    ) -> DocSet:
        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **resource_args
        )
        return DocSet(self._context, scan)

    def arrow(self, tables: Union[Table, bytes, list[Union[Table, bytes]]]) -> DocSet:
        scan = ArrowScan(tables)
        return DocSet(self._context, scan)

    def document(self, docs: list[Document]) -> DocSet:
        scan = DocScan(docs)
        return DocSet(self._context, scan)

    def pandas(self, dfs: Union[DataFrame, list[DataFrame]]) -> DocSet:
        scan = PandasScan(dfs)
        return DocSet(self._context, scan)
