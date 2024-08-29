import logging
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING
import uuid

from sycamore.context import Context
from sycamore.data import Document, MetadataDocument
from sycamore.materialize_config import MaterializeSourceMode
from sycamore.plan_nodes import Node, UnaryNode, NodeTraverse
from sycamore.transforms.base import rename

if TYPE_CHECKING:
    from ray import Dataset
    import pyarrow


logger = logging.getLogger("__name__")


class _PyArrowFsHelper:
    def __init__(self, fs: "pyarrow.FileSystem"):
        self._fs = fs

    def list_files(self, path):
        from pyarrow.fs import FileSelector

        return self._fs.get_file_info(FileSelector(str(path), allow_not_found=True, recursive=True))

    def file_exists(self, path: Path) -> bool:
        from pyarrow.fs import FileType

        info = self._fs.get_file_info(str(path))
        return info.type == FileType.File

    def safe_cleanup(self, path) -> None:
        # materialize dirs should be non-hierarchical, minimize the chance that
        # mis-use will delete unexpected files.
        plen = len(str(path)) + 1
        for fi in self.list_files(path):
            assert "/" not in fi.path[plen:], f"Refusing to clean {path}. Found unexpected hierarchical file {fi.path}"

        logging.info(f"Cleaning up any materialized files in {path}")
        self._fs.delete_dir_contents(str(path), missing_dir_ok=True)
        self._fs.create_dir(str(path))


class Materialize(UnaryNode):
    def __init__(
        self,
        child: Optional[Node],
        context: Context,
        path: Optional[Union[Path, str, dict]] = None,
        source_mode: MaterializeSourceMode = MaterializeSourceMode.OFF,
        **kwargs,
    ):
        assert child is None or isinstance(child, Node)

        self._root = None
        if path is None:
            pass
        elif isinstance(path, str) or isinstance(path, Path):
            (self._fs, self._root) = self.infer_fs(str(path))
            self._fshelper = _PyArrowFsHelper(self._fs)
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
            self._fshelper = _PyArrowFsHelper(self._fs)
            self._doc_to_name = path.get("name", self.doc_to_name)
            self._doc_to_binary = path.get("tobin", Document.serialize)
            assert callable(self._doc_to_name)
            self._clean_root = path.get("clean", True)
        else:
            assert False, f"unsupported type ({type(path)}) for path argument, expected str, Path, or dict"

        if source_mode != MaterializeSourceMode.OFF:
            assert path is not None
            assert (
                self._doc_to_binary == Document.serialize
            ), "Using materialize in source mode requires default serialization"
            assert self._clean_root, "Using materialize in source mode requires cleaning the root"

        self._source_mode = source_mode
        self._executed_child = False

        super().__init__(child, **kwargs)

    def execute(self, **kwargs) -> "Dataset":
        logger.debug("Materialize execute")
        if self._source_mode == MaterializeSourceMode.IF_PRESENT:
            success = self._fshelper.file_exists(self._success_path())
            if success or len(self.children) == 0:
                logger.info(f"Using {self._root} as cached source of data")
                self._executed_child = False
                if not success:
                    self._verify_has_files()
                    logging.warning(f"materialize.success not found in {self._root}. Returning partial data")

                from ray.data import read_binary_files

                files = read_binary_files(self._root, filesystem=self._fs, file_extensions=["pickle"])

                return files.map(self._ray_to_document)

        self._executed_child = True
        # right now, no validation happens, so save data in parallel. Once we support validation
        # to support retries we won't be able to run the validation in parallel.  non-shared
        # filesystems will also eventually be a problem but we can put it off for now.
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

    def _verify_has_files(self) -> None:

        assert self._root is not None
        assert self._fs is not None

        files = self._fshelper.list_files(self._root)
        for n in files:
            if n.path.endswith(".pickle"):
                return

        raise ValueError(f"Materialize root {self._root} has no .pickle files")

    def _ray_to_document(self, dict: dict[str, Any]) -> dict[str, bytes]:
        return {"doc": dict["bytes"]}

    def _will_be_source(self) -> bool:
        return self._source_mode == MaterializeSourceMode.IF_PRESENT and self._fshelper.file_exists(
            self._success_path()
        )

    def local_execute(self, docs: list[Document]) -> list[Document]:
        if self._source_mode == MaterializeSourceMode.IF_PRESENT:
            if self._fshelper.file_exists(self._success_path()):
                self._executed_child = False
                logger.info(f"Using {self._root} as cached source of data")

                return self.local_source()

        if self._root is not None:
            self.cleanup()
            for d in docs:
                self.save(d)
            self._executed_child = True

        return docs

    def local_source(self) -> list[Document]:
        assert self._root is not None
        self._verify_has_files()
        logger.info(f"Using {self._root} as cached source of data")
        if not self._fshelper.file_exists(self._success_path()):
            logging.warning(f"materialize.success not found in {self._root}. Returning partial data")
        ret = []
        for fi in self._fshelper.list_files(self._root):
            n = Path(fi.path)
            if n.suffix == ".pickle":
                f = self._fs.open_input_stream(str(n))
                ret.append(Document.deserialize(f.read()))
                f.close()

        return ret

    def _success_path(self):
        return self._root / "materialize.success"

    def finalize(self):
        if not self._executed_child:
            return
        if self._root is not None:
            self._fs.open_output_stream(str(self._success_path())).close()
            assert self._fshelper.file_exists(self._success_path())

    @staticmethod
    def infer_fs(path: str) -> "pyarrow.FileSystem":
        import re

        if not re.match("^[a-z0-9]+://.", path):
            # pyarrow expects URIs, accepts /dir/path, but rejects ./dir/path
            # normalize everything to a URI.
            p = Path(path)
            if p.is_absolute():
                path = p.as_uri()
            else:
                path = p.absolute().as_uri()

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
        if self._clean_root and self._fshelper.file_exists(path):
            raise ValueError(f"Duplicate name {path} generated for clean root")
        with self._fs.open_output_stream(str(path)) as out:
            out.write(bin)

    def cleanup(self) -> None:
        if self._root is None:
            return

        if not self._clean_root:
            self._fs.create_dir(str(self._root))
            return

        self._fshelper.safe_cleanup(self._root)

    @staticmethod
    def doc_to_name(doc: Document) -> str:
        if isinstance(doc, MetadataDocument):
            return "md-" + str(uuid.uuid4()) + ".pickle"

        assert isinstance(doc, Document)
        return (doc.doc_id or str(uuid.uuid4())) + ".pickle"


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
        else:
            path = path.copy()
        if "clean" not in path:
            path["clean"] = True

        self._choose_directory(path)
        self._basename_to_count: dict[str, int] = {}

    def once(self, context, node):
        self._name_unique = set()
        node = node.traverse(after=self._naming_pass())
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

            path = self._directory / materialize["name"]

            if not self._path["clean"]:
                return

            self._fshelper.safe_cleanup(path)

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
            path["root"] = self._directory / materialize["name"]
            materialize["mark"] = True
            materialize["count"] = materialize.get("count", 0) + 1
            return Materialize(node, context, path=path)

        return before

    def _choose_directory(self, path):
        from pathlib import Path

        directory = path.pop("root", None)
        self._path = path

        if directory is None:
            from datetime import datetime
            import tempfile

            now = datetime.now().replace(microsecond=0)
            dir = Path(tempfile.gettempdir()) / f"materialize.{now.isoformat()}"
            directory = str(dir)
            logger.info(f"Materialize directory was not specified. Used {dir}")

        (self._fs, self._directory) = Materialize.infer_fs(directory)
        if "fs" in self._path:
            self._fs = self._path["fs"]
        self._fshelper = _PyArrowFsHelper(self._fs)
