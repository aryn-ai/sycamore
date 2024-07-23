import pickle
import posixpath
from typing import Optional, Iterable, Any

from pyarrow.filesystem import FileSystem
from ray.data import Datasink
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from sycamore.data import Document, MetadataDocument

import structlog

log = structlog.get_logger()


def pickle_doc(doc: Document) -> bytes:
    return pickle.dumps(doc)


def pickle_name(doc: Document, extension="pickle"):
    return f"{doc.doc_id}.{extension}"


class LunaLogger(Datasink):
    """
    Custom data sink for Sycamore nodes. It emits metrics using the structlog interface and attaches query and
    node information along with each record. Supports a verbose mode that writes raw documents as well.
    """

    def __init__(
        self,
        query_id: str,
        node_id: str,
        path: str = None,
        filesystem: Optional[FileSystem] = None,
        makedirs: bool = False,
        verbose: bool = False,
    ):
        self._query_id = query_id
        self._node_id = node_id
        self._verbose = verbose
        self._path = path

        if verbose:
            assert path is not None
            (paths, self._filesystem) = _resolve_paths_and_filesystem(path, filesystem)
            self._root = paths[0]

        if makedirs:
            self._filesystem.create_dir(path)

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        for block in blocks:
            b = BlockAccessor.for_block(block).to_arrow().to_pylist()
            for _, row in enumerate(b):
                doc = Document.from_row(row)
                if isinstance(doc, MetadataDocument):
                    continue
                log.info("Processed record ", query_id=self._query_id, doc_id=doc.doc_id, node_id=self._node_id)
                if self._verbose:
                    assert self._root is not None
                    assert self._filesystem is not None
                    doc_bytes = pickle_doc(doc)
                    path = posixpath.join(self._root, pickle_name(doc))
                    with self._filesystem.open_output_stream(path) as file:
                        file.write(doc_bytes)
