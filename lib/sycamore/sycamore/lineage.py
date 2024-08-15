from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import uuid

from sycamore.context import Context
from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node, UnaryNode
from sycamore.transforms.base import rename

if TYPE_CHECKING:
    from ray import Dataset
    import pyarrow


class Materialize(UnaryNode):
    def __init__(self, child: Node, context: Context, path: Optional[Union[Path, str, dict]], **kwargs):
        assert isinstance(child, Node)

        self._root = None
        if path is None:
            pass
        elif isinstance(path, str) or isinstance(path, Path):
            (self._fs, self._root) = self.infer_fs(str(path))
            self._doc_to_name = self.doc_to_name
            self._doc_to_binary = Document.serialize
            self._clean_root = True
        elif isinstance(path, dict):
            assert "root" in path, "Need to specify root in materialize(path={})"
            self._root = Path(path["root"])
            if "fs" in path:
                self._fs = path["fs"]
            else:
                (self._fs, self._root) = self.infer_fs(str(self._root))
            self._doc_to_name = path.get("name", self.doc_to_name)
            self._doc_to_binary = path.get("tobin", Document.serialize)
            assert callable(self._doc_to_name)
            self._clean_root = path.get("clean", True)
        else:
            assert False, f"unsupported type ({type(path)}) for path argument, expected str, Path, or dict"

        super().__init__(child, **kwargs)

    def execute(self, **kwargs) -> "Dataset":
        # right now, the only thing we can do is save data, so do it in parallel.  once we support
        # validation to support retries we won't be able to run the validation in parallel.
        # non-shared filesystems will also eventually be a problem but we can put it off for now.
        input_dataset = self.child().execute(**kwargs)
        if self._root is not None:
            import numpy

            self.cleanup()

            @rename("lineage-materialize")
            def ray_callable(ray_input: dict[str, numpy.ndarray]) -> dict[str, numpy.ndarray]:
                for s in ray_input.get("doc", []):
                    self.save(Document.deserialize(s))
                return ray_input

            return input_dataset.map_batches(ray_callable)

        return input_dataset

    def local_execute(self, docs: list[Document]) -> list[Document]:
        if self._root is not None:
            self.cleanup()
            for d in docs:
                self.save(d)
        [d for d in docs if isinstance(d, MetadataDocument)]
        # logging.info(f"Found {len(md)} md documents")
        # logging.info(f"\n{pprint.pformat(md)}")
        return docs

    @staticmethod
    def infer_fs(path: str) -> "pyarrow.FileSystem":
        from pyarrow import fs

        (fs, root) = fs.FileSystem.from_uri(path)
        return (fs, Path(root))

    def save(self, doc: Document) -> None:
        bin = self._doc_to_binary(doc)
        if bin is None:
            return
        assert isinstance(bin, bytes), f"tobin function returned {type(bin)} not bytes"
        assert self._root is not None
        name = self._doc_to_name(doc)
        path = self._root / name
        with self._fs.open_output_stream(str(path)) as out:
            out.write(bin)

    def cleanup(self) -> None:
        if not self._clean_root:
            return

        import shutil

        if self._root is None:
            return
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def doc_to_name(doc: Document) -> str:
        if isinstance(doc, MetadataDocument):
            return "md-" + str(uuid.uuid4())

        assert isinstance(doc, Document)
        return doc.doc_id or str(uuid.uuid4())
