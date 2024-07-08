from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node, Write

from pyarrow.fs import FileSystem
from pyarrow import NativeFile

from ray.data import Dataset
from ray.data.datasource import FilenameProvider, Datasink, BlockBasedFileDatasink
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from ray.data.block import Block, BlockAccessor
from ray.data._internal.execution.interfaces import TaskContext

from collections import UserDict
from io import StringIO
import json
import logging
from pathlib import Path
import posixpath
import uuid
from typing import Any, Callable, Optional, Iterable
from sycamore.utils.time_trace import TimeTrace

logger = logging.getLogger(__name__)


class JSONEncodeWithUserDict(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UserDict):
            return obj.data
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        else:
            return json.JSONEncoder.default(self, obj)


def default_filename(doc: Document, extension: Optional[str] = None) -> str:
    """Returns a default filename based on document_id and extension.

    If the doc_id is not set, a new uuid is generated.

    Args:
        doc: A sycamore.data.Document instance.
        extension: An optional extension that will be appended to the name following a '.'.
    """
    if doc.doc_id is None:
        base_name = str(uuid.uuid4())
    else:
        base_name = str(doc.doc_id)

    if extension is not None:
        return f"{base_name}.{extension.lstrip('.')}"
    return base_name


def doc_path_filename(extension: str, suffix: str) -> Callable[[Document], str]:
    """Returns a function that takes a doc and returns a filename based on path.

    The the filename is extracted from the 'path' property and is used with the
    provided suffix and extension. For example, if the input path is
    my_dataset/my_directory/file.json, then the returned filename would be
    'file_{suffix}.{extension}'

    Args:
        extension: Extension to use for the file.
        suffix: Filename suffix to place before the extension.
    """

    def fn(doc: Document):
        path = Path(doc.properties["path"])
        base_name = ".".join(path.name.split(".")[0:-1])
        return f"{base_name}_{suffix}.{extension}"

    return fn


def default_doc_to_bytes(doc: Document) -> bytes:
    """Returns the text_representation of the document if available or the binary representation if not.

    Args:
        doc: A sycamore.data.Document instance.
    """
    if doc.text_representation is not None:
        return doc.text_representation.encode("utf-8")
    elif doc.binary_representation is not None:
        return doc.binary_representation
    else:
        raise RuntimeError(f"No default content representation for Document {doc}")


def json_properties_content(doc: Document) -> bytes:
    """Return just the properties of the document as a json object"""
    return json.dumps(doc.properties).encode("utf-8")


def elements_to_bytes(doc: Document) -> bytes:
    """Returns a utf-8 encoded json string containing the elements of the document.

    The elements are line-delimited.
    """

    out = StringIO()
    for element in doc.elements:
        json.dump(element, out, cls=JSONEncodeWithUserDict)
        out.write("\n")
    return out.getvalue().encode("utf-8")


def document_to_bytes(doc: Document) -> bytes:
    """
    Returns a UTF-8 encoded json string of the document.  Adds newline.
    Beware this will try to interpret binary_representation as UTF-8.
    """

    out = StringIO()
    json.dump(doc, out, cls=JSONEncodeWithUserDict)
    out.write("\n")
    return out.getvalue().encode("utf-8")


class FileWriter(Write):
    """Sycamore Write implementation that writes out binary or text representation.

    Supports writting files to any FileSystem supported by Ray (e.g. arrow.fs.FileSystem).
    Each document is written to a separate file.
    """

    def __init__(
        self,
        plan: Node,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        **ray_remote_args,
    ):
        """Initializes a FileWriter instance.

        Args:
            plan: A Sycamore plan representing the DocSet to write out.
            path: The path prefix to write to. Should include the scheme.
            filesystem: The pyarrow.fs FileSystem to use.
            filename_fn: A function for generating a file name. Takes a Document
                and returns a unique name that will be appended to path.
            doc_to_bytes_fn: A function from a Document to bytes for generating the data to write.
                Defaults to using text_representation if available, or binary_representation
                if not.
            ray_remote_args: Arguments to pass to the underlying execution environment.
        """

        super().__init__(plan, **ray_remote_args)
        self.path = path
        self.filesystem = filesystem
        self.filename_fn = filename_fn
        self.doc_to_bytes_fn = doc_to_bytes_fn
        self.ray_remote_args = ray_remote_args

    def execute(self, **kwargs) -> Dataset:
        dataset = self.child().execute()

        dataset.write_datasink(
            _FileDataSink(
                self.path,
                filesystem=self.filesystem,
                filename_fn=self.filename_fn,
                doc_to_bytes_fn=self.doc_to_bytes_fn,
            ),
            ray_remote_args=self.ray_remote_args,
        )

        return dataset


class JsonWriter(Write):
    """
    Sycamore Write implementation that writes blocks of Documents to JSONL
    files.  Supports output to any Ray-supported filesystem.  Typically
    each source document (such as a PDF) ends up as a block.  After an
    explode(), there will be multiple documents in the block.
    """

    def __init__(
        self,
        plan: Node,
        path: str,
        filesystem: Optional[FileSystem] = None,
        **ray_remote_args,
    ) -> None:
        """
        Construct a JsonWriter instance.

        Args:
            plan: A Sycamore plan representing the DocSet to write out.
            path: The path prefix to write to. Should include the scheme.
            filesystem: The pyarrow.fs FileSystem to use.
            ray_remote_args: Arguments to pass to the underlying execution environment.
        """

        super().__init__(plan, **ray_remote_args)
        self.path = path
        self.filesystem = filesystem
        self.ray_remote_args = ray_remote_args

    def execute(self, **kwargs) -> Dataset:
        ds = self.child().execute()
        sink = _JsonBlockDataSink(self.path, filesystem=self.filesystem)
        ds.write_datasink(sink, ray_remote_args=self.ray_remote_args)
        return ds


class DocToRowFilenameProvider(FilenameProvider):
    def __init__(self, filename_fn: Callable[[Document], str]):
        self._filename_fn = filename_fn

    def get_filename_for_row(self, row: dict[str, Any], task_index: int, block_index: int, row_index: int) -> str:
        return self._filename_fn(Document.from_row(row))


class BlockFilenameProvider(FilenameProvider):
    def get_filename_for_block(self, block: Block, task_index: int, block_index: int) -> str:
        return f"block_{block_index}_{task_index}.jsonl"


class _FileDataSink(Datasink):
    def __init__(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        makedirs: bool = False,
    ):
        (paths, self._filesystem) = _resolve_paths_and_filesystem(path, filesystem)
        self._root = paths[0]
        self._filename_fn = filename_fn
        self._doc_to_bytes_fn = doc_to_bytes_fn

        if makedirs:
            self._filesystem.create_dir(path)

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
