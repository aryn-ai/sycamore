from sycamore.data import Document
from sycamore.plan_nodes import Node, Write

from pyarrow.fs import FileSystem

from collections import UserDict
from io import StringIO
import json
import logging
from pathlib import Path
import posixpath
import uuid
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data import Dataset


logger = logging.getLogger(__name__)


class JSONEncodeWithUserDict(json.JSONEncoder):
    def default(self, obj):
        from sycamore.data.bbox import BoundingBox

        if isinstance(obj, UserDict):
            return obj.data
        elif isinstance(obj, BoundingBox):
            return {"x1": obj.x1, "y1": obj.y1, "x2": obj.x2, "y2": obj.y2}
        elif isinstance(obj, bytes):
            import base64

            return base64.b64encode(obj).decode("utf-8")
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


def document_to_json_bytes(doc: Document) -> bytes:
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

    def execute(self, **kwargs) -> "Dataset":
        from sycamore.connectors.file.file_writer_ray import _FileDataSink

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

    def local_execute(self, all_docs: list[Document]) -> list[Document]:
        from sycamore.utils.pyarrow import cross_check_infer_fs
        from sycamore.data import MetadataDocument

        (filesystem, path) = cross_check_infer_fs(self.filesystem, self.path)

        for d in all_docs:
            if isinstance(d, MetadataDocument):
                continue
            bytes = self.doc_to_bytes_fn(d)
            file_path = posixpath.join(path, self.filename_fn(d))
            with filesystem.open_output_stream(str(file_path)) as file:
                file.write(bytes)

        return all_docs


class JsonWriter(FileWriter):
    """
    Sycamore Write implementation that writes blocks of Documents to JSONL
    files.  Supports output to any Ray-supported filesystem.  Typically
    each source document (such as a PDF) ends up as a block.  After an
    explode(), there will be multiple documents in the block.

    Warning: JSON writing is not reversable with JSON reading. You will get
    a slightly different document back.
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

        super().__init__(
            plan, path=path, filesystem=filesystem, doc_to_bytes_fn=document_to_json_bytes, **ray_remote_args
        )

    def execute(self, **kwargs) -> "Dataset":
        ds = self.child().execute()
        from sycamore.connectors.file.file_writer_ray import _JsonBlockDataSink

        sink = _JsonBlockDataSink(self.path, filesystem=self.filesystem)
        ds.write_datasink(sink, ray_remote_args=self.ray_remote_args)
        return ds
