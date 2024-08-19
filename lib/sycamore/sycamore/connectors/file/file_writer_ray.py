from typing import Any, Callable, Iterable, Optional

import posixpath
from pyarrow.fs import FileSystem, FileType
from pyarrow import NativeFile
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import Datasink, FilenameProvider, BlockBasedFileDatasink
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from ray.data._internal.execution.interfaces import TaskContext
from urllib.parse import urlparse

from sycamore.connectors.file.file_writer import default_filename, default_doc_to_bytes, document_to_bytes
from sycamore.data import Document, MetadataDocument
from sycamore.utils.time_trace import TimeTrace


class _FileDataSink(Datasink):
    def __init__(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        makedirs: bool = True,
    ):
        (paths, self._filesystem) = _resolve_paths_and_filesystem(path, filesystem)
        self._root = paths[0]
        if self._root == "":
            self._root = "./"
        self._filename_fn = filename_fn
        self._doc_to_bytes_fn = doc_to_bytes_fn
        self._makedirs = makedirs

    def on_write_start(self) -> None:
        if not self._makedirs:
            return

        # This follows Ray logic to skip attempting to
        # create "directories" for s3 filesystems.
        parsed_uri = urlparse(self._root)
        is_s3_uri = parsed_uri.scheme == "s3"

        if not is_s3_uri and self._filesystem.get_file_info(self._root).type is FileType.NotFound:
            self._filesystem.create_dir(self._root, recursive=True)

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        for block in blocks:
            b = BlockAccessor.for_block(block).to_arrow().to_pylist()
            for _, row in enumerate(b):
                doc = Document.from_row(row)
                if isinstance(doc, MetadataDocument):
                    continue
                bytes = self._doc_to_bytes_fn(doc)
                path = posixpath.join(self._root, self._filename_fn(doc))
                with self._filesystem.open_output_stream(path) as file:
                    file.write(bytes)


class _JsonBlockDataSink(BlockBasedFileDatasink):
    def __init__(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
    ) -> None:
        class BlockFilenameProvider(FilenameProvider):
            def get_filename_for_block(self, block: Block, task_index: int, block_index: int) -> str:
                return f"block_{block_index}_{task_index}.jsonl"

        super().__init__(path, filesystem=filesystem, filename_provider=BlockFilenameProvider())

    def write_block_to_file(self, block: BlockAccessor, file: NativeFile) -> None:
        with TimeTrace("jsonSink"):
            for row in block.iter_rows(True):  # type: ignore[var-annotated]
                doc = Document.from_row(row)
                if isinstance(doc, MetadataDocument):
                    continue
                del doc.binary_representation  # Doesn't make sense in JSON
                binary = document_to_bytes(doc)
                file.write(binary)
