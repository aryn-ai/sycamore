from sycamore.data import Document
from sycamore.plan_nodes import Node, Write

from pyarrow.fs import FileSystem

from ray.data import Dataset
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import FileBasedDatasource, WriteResult
from ray.data._internal.execution.interfaces import TaskContext

from collections import UserDict
from io import StringIO
import json
import logging
import os
from pathlib import Path
import uuid
from typing import Callable, Iterable, Optional, no_type_check

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

    def execute(self) -> Dataset:
        dataset = self.child().execute()

        dataset.write_datasource(
            _WritableFilePerRowDataSource(),
            path=self.path,
            dataset_uuid=uuid.uuid4(),
            filesystem=self.filesystem,
            filename_fn=self.filename_fn,
            doc_to_bytes_fn=self.doc_to_bytes_fn,
            ray_remote_args=self.ray_remote_args,
        )

        return dataset


# Some of this code is taken from Ray's FileBasedDatasource. We should switch to using that
# Once it supports using a custom filename.
class _WritableFilePerRowDataSource(FileBasedDatasource):
    _WRITE_FILE_PER_ROW = True

    # This will not typecheck correctly because the parameters don't match the superclass.
    # It turns out this is a problem in ray itself -- the parameters in
    # FileBasedDatasource::write don't match the parameters of Datasource::write. This
    # isn't a correctness problem, but it means there is no way to get mypy to check this.
    @no_type_check
    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        prefer_text: bool = True,
        **write_args,
    ) -> WriteResult:
        from ray.data.datasource.file_based_datasource import (
            _open_file_with_retry,
            _resolve_paths_and_filesystem,
            _unwrap_s3_serialization_workaround,
        )

        path, filesystem = _resolve_paths_and_filesystem(path, filesystem)
        path = path[0]

        for block in blocks:
            block = BlockAccessor.for_block(block)  # .to_arrow().to_pylist()

            fs = _unwrap_s3_serialization_workaround(filesystem)

            for row in block.iter_rows(public_row_format=True):
                doc = Document.from_row(row)
                filename = filename_fn(doc)

                write_path = os.path.join(path, filename)
                logger.debug(f"Writing file at {write_path}")

                # with fs.open_output_stream(write_path) as outfile:
                with _open_file_with_retry(write_path, lambda: fs.open_output_stream(write_path)) as outfile:
                    outfile.write(doc_to_bytes_fn(doc))

        return "ok"
