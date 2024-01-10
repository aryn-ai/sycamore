from typing import Optional, Union

from pandas import DataFrame
from pyarrow import Table
from pyarrow.filesystem import FileSystem

from sycamore import Context, DocSet
from sycamore.data import Document
from sycamore.scans.materialized_scan import ArrowScan, DocScan, PandasScan
from sycamore.scans.file_scan import BinaryScan, FileMetadataProvider, JsonScan


class DocSetReader:
    """
    Read data from different kinds of sources into DocSet.

    DocSetReader is exposed through sycamore context read API.
    """

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
        """
        Scan data file into raw bytes

        For each file, BinaryScan creates one Document, we use BinaryScan to process
        unstructured data format like PDF or HTML.

        Examples:
            >>> import sycamore
            >>> import pyarrow as pa
            >>> context = sycamore.init()
            >>> docset = context.read.binary("s3://bucket/prefix", "pdf")
        """
        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **resource_args
        )
        return DocSet(self._context, scan)

    # TODO: Support including the metadata attributes in the manifest file directly
    def manifest(
        self,
        metadata_provider: FileMetadataProvider,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        **resource_args
    ) -> DocSet:
        paths = metadata_provider.get_paths()
        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **resource_args
        )
        return DocSet(self._context, scan)

    def json(
        self,
        paths: Union[str, list[str]],
        properties: Optional[Union[str, list[str]]] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        document_body_field: Optional[str] = None,
        **resource_args
    ) -> DocSet:
        """
        Scan JSON or JSONL data file into DocSet

        We currently handle each JSON file by reading binary and then parsing it into Document.
        Examples:
            >>> import sycamore
            >>> import pyarrow as pa
            >>> context = sycamore.init()
            >>> docset = context.read.json("s3://bucket/prefix")
        """
        json_scan = JsonScan(
            paths,
            properties=properties,
            metadata_provider=metadata_provider,
            document_body_field=document_body_field,
            **resource_args
        )
        return DocSet(self._context, json_scan)

    def arrow(self, tables: Union[Table, bytes, list[Union[Table, bytes]]]) -> DocSet:
        """
        Scan arrow data into a DocSet

        Examples:
            >>> import sycamore
            >>> import pyarrow as pa
            >>> context = sycamore.init()
            >>> table = pa.table({"x": [1]})
            >>> docset = context.read.arrow(table)
        """
        scan = ArrowScan(tables)
        return DocSet(self._context, scan)

    def document(self, docs: list[Document]) -> DocSet:
        """
        Scan a list of Documents into a DocSet

        Examples:
            >>> import sycamore
            >>> from sycamore.data import Document
            >>> context = sycamore.init()
            >>> documents = [Document()]
            >>> docset = context.read.document(documents)
        """
        scan = DocScan(docs)
        return DocSet(self._context, scan)

    def pandas(self, dfs: Union[DataFrame, list[DataFrame]]) -> DocSet:
        """
        Scan a list of Documents into a DocSet

        Examples:
            >>> import sycamore
            >>> from pandas import DataFrame
            >>> context = sycamore.init()
            >>> df = DataFrame({"doc_id": 1, "type": "hello, world!"})
            >>> docset = context.read.pandas(df)
        """
        scan = PandasScan(dfs)
        return DocSet(self._context, scan)
