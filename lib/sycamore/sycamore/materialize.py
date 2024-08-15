import logging
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import uuid

from sycamore.context import Context
from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node, UnaryNode, NodeTraverse
from sycamore.transforms.base import rename

if TYPE_CHECKING:
    from ray import Dataset
    import pyarrow


logger = logging.getLogger("__name__")


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
        logging.error("Materialize execute")
        # right now, the only thing we can do is save data, so do it in parallel.  once we support
        # validation to support retries we won't be able to run the validation in parallel.
        # non-shared filesystems will also eventually be a problem but we can put it off for now.
        input_dataset = self.child().execute(**kwargs)
        if self._root is not None:
            import numpy

            self.cleanup()

            @rename("materialize")
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


class AutoMaterialize(NodeTraverse):
    """Automatically add materialize nodes after every node in an execution.

    Usage:
       from sycamore.materialize import AutoMaterialize
       ctx = sycamore.init()
       ctx.rewrite_rules.append(AutoMaterialize())

       # override base directory for materialization
       ctx.rewrite_rules.append(AutoMaterialize("/home/example/subdir"))

       # override multiple parameters, root is the root directory for the materialize operations
       # the remaining parameters work the same as docset.materialize(), and are simply passed through.
       a = AutoMaterialize({"root":Path|str, fs=pyarrow.fs, name=fn, clean=bool, tobin=fn})
       ctx.rewrite_rules.append(a)

    Nodes in the plan will automatically be named. You can specify a name by defining it for the node:
       ctx = sycamore.init()
       ds = ctx.read.document(docs, materialize={"name": "reader"}).map(noop_fn, materialize={"name": "noop"})
       # NOTE: names in a single execution must be unique. This is guaranteed by auto naming
       # NOTE: automatic names are not guaranteed to be stable
    """

    def __init__(self, path: Union[str, Path, dict] = {}):
        super().__init__()
        if isinstance(path, str) or isinstance(path, Path):
            path = {"root": path}
        if "clean" not in path:
            path["clean"] = True

        self._path = path
        self.directory = path.pop("root", None)
        self._basename_to_count: dict[str, int] = {}

    def once(self, context, node):
        self._name_unique = set()
        node = node.traverse(after=self._naming_pass())

        self._setup_directory()

        node = node.traverse(visit=self._cleanup_pass())
        node = node.traverse(before=self._wrap_pass(context))

        return node

    def _naming_pass(self):
        def after(node):
            if isinstance(node, Materialize):
                return node

            if "materialize" not in node.properties:
                node.properties["materialize"] = {}

            materialize = node.properties["materialize"]

            if "name" not in materialize:
                basename = node.__class__.__name__
                count = self._basename_to_count.get(basename, 0)
                materialize["name"] = f"{basename}.{count}"
                self._basename_to_count[basename] = count + 1

            name = materialize["name"]
            assert name not in self._name_unique, f"Duplicate name {name} found in nodes"
            self._name_unique.add(name)

            return node

        return after

    def _cleanup_pass(self):
        def visit(node):
            if isinstance(node, Materialize):
                return
            materialize = node.properties["materialize"]
            materialize.pop("mark", None)

            path = self.directory / materialize["name"]

            if path.exists():
                if not self._path["clean"]:
                    return
                # auto-materialize dirs should be non-hierarchical, minimize the chance that
                # mis-use will delete unexpected files.
                for p in path.iterdir():
                    assert not p.is_dir()

                for p in path.iterdir():
                    p.unlink()
            else:
                path.mkdir(parents=True)

        return visit

    def _wrap_pass(self, context):
        def before(node):
            if isinstance(node, Materialize):
                assert len(node.children) == 1, f"Materialize node {node.__name__} should have exactly one child"
                child = node.children[0]
                if isinstance(child, Materialize):
                    logger.warning(f"Found two materialize nodes in a row: {node.__name__} and {child.__name__}")
                    return
                child.properties["materialize"]["mark"] = True
                return node

            materialize = node.properties["materialize"]
            if materialize.get("mark", False):
                return node

            path = self._path.copy()
            path["root"] = self.directory / materialize["name"]
            materialize["mark"] = True
            materialize["count"] = materialize.get("count", 0) + 1
            return Materialize(node, context, path=path)

        return before

    def _setup_directory(self):
        from pathlib import Path

        if self.directory is None:
            from datetime import datetime
            import tempfile

            now = datetime.now().replace(microsecond=0)
            dir = Path(tempfile.gettempdir()) / f"materialize.{now.isoformat()}"
            self.directory = dir
            logger.info(f"Materialize directory was not specified. Used {dir}")

        if not isinstance(self.directory, Path):
            self.directory = Path(self.directory)
