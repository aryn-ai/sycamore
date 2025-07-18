import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING, cast
import inspect

from sycamore.context import Context
from sycamore.data import Document, MetadataDocument
from sycamore.materialize_config import MaterializeSourceMode, RandomNameGroup, MRRNameGroup, MaterializeNameGroup
from sycamore.plan_nodes import Node, UnaryNode, NodeTraverse
from sycamore.transforms.base import rename

if TYPE_CHECKING:
    from ray.data import Dataset
    import pyarrow


logger = logging.getLogger(__name__)


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


class MaterializeReadReliability(NodeTraverse):
    """
    A node traversal rule that implements reliable materialization for document pipelines.
    This class handles batch processing, automatic retries, and progress tracking to ensure
    robust data materialization even in the presence of failures.

    Args:
        max_batch (int): Maximum number of documents to process in a single batch. Default: 200
        max_retries (int): Maximum number of retry attempts for failed batches. Default: 20

    # Add to context rewrite rules
    ctx.rewrite_rules.append(MaterializeReadReliabliity([batch_size=200]))

    Warning:
        This class enforces specific constraints on pipeline structure:
        - Pipeline must have exactly one output materialization point
        - MRR is only compatible with docset.execute(), not docset.take_all()
        - The reliability pipeline requires proper document ID formatting (doc-path-sha256-*)
    """

    def __init__(self, max_batch: int = 200, max_retries: int = 20):

        super().__init__()

        self.max_batch = max_batch
        self.current_batch = 0
        self.retries_count = 0
        # Need for refresh_seen_files
        self.prev_seen = -1
        self.max_retries = max_retries
        self.cycle_error: Optional[Union[str, Exception]] = ""
        self.iteration = 0
        self._name_group = MRRNameGroup

    def reinit(self, out_mat_path, max_batch, max_retries):

        (fs, path) = Materialize.infer_fs(str(out_mat_path))
        logger.info(f"Fetching files from {out_mat_path}")
        self.fs = fs
        self.path = str(path)
        self.__init__(max_batch=max_batch, max_retries=max_retries)

        # Initialize seen files
        self._refresh_seen_files()
        self.prev_seen = len(self.seen)

    @staticmethod
    def maybe_execute_reliably(docset) -> bool:
        """
        Determines if the execution should use reliability mode.

        Returns:
            Tuple of (should_use_reliability, mrr_instance)
        """
        plan, context = docset.plan, docset.context
        if not isinstance(plan, Materialize):
            return False

        mrr = next((rule for rule in context.rewrite_rules if isinstance(rule, MaterializeReadReliability)), None)
        if not mrr:
            return False

        if isinstance(plan._orig_path, str) or isinstance(plan._orig_path, Path):
            destination_path = plan._orig_path
        elif isinstance(plan._orig_path, dict):
            destination_path = plan._orig_path["root"]
        else:
            return False

        mrr.reinit(out_mat_path=destination_path, max_batch=mrr.max_batch, max_retries=mrr.max_retries)
        MaterializeReadReliability.execute_reliably(context, plan, mrr)

        return True

    @staticmethod
    def execute_reliably(context, plan, mrr: "MaterializeReadReliability", **kwargs) -> None:
        """
        Executes the plan with reliability guarantees.
        Handles batching, retries, and error reporting.
        """
        from sycamore.executor import Execution
        import traceback

        while True:
            mrr.clear_console()
            try:
                for doc in Execution(context).execute_iter(plan, **kwargs):
                    pass

                if mrr.current_batch == 0:
                    mrr.reset_batch()
                    logger.info(f"\nProcessed {len(mrr.seen)} docs.")
                    break

            except AssertionError:
                raise
            except Exception as e:
                mrr.cycle_error = e
                logger.info(f"Retrying batch job because of {e}.\n" f"Processed {len(mrr.seen)} docs at present.")
                detailed_cycle_error = traceback.format_exc()
                print(f"Detailed Trace:\n{detailed_cycle_error}")
            mrr.reset_batch()

            if mrr.retries_count > mrr.max_retries:
                logger.info(
                    f"\nGiving up after retrying {mrr.retries_count} times. " f"Processed {len(mrr.seen)} docs."
                )
                break

    def once(self, context, node):
        for rule in context.rewrite_rules:
            if isinstance(rule, MaterializeReadReliability):
                mrr = rule
        self.count = 0
        node = node.traverse(visit=self.propagate_mrr(mrr))
        return node

    def propagate_mrr(self, mrr):

        from sycamore.connectors.file.file_scan import BinaryScan

        def visit(node):
            if self.count == 0:
                if len(node.children) > 0:
                    assert isinstance(
                        node, Materialize
                    ), "The last node should be a materialize node to ensure reliability"
                    logger.info("Overriding doc_to_name, doc_to_binary, clean_root for reliability pipeline")
                    node._doc_to_name = self._name_group.doc_to_materialize_name
                    node._name_group = self._name_group
                    node._clean_root = False
                    node._source_mode = MaterializeSourceMode.RECOMPUTE
            elif isinstance(node, BinaryScan):
                assert len(node.children) == 0, "Binary Scan should be the first node in the reliability pipeline"
                node._path_filter = mrr.filter
            elif isinstance(node, Materialize):
                assert (
                    len(node.children) == 0
                ), "Only first and last node should be materialize nodes to maintain reliability."
                node._name_group = self._name_group
                node._doc_to_name = self._name_group.doc_to_materialize_name
                # If there is already a filter on the materialize, we need to
                # AND it with the MRR filter; which means caching a copy of the
                # original filter to not end up with recursive expansions ((orig AND mrr) AND mrr)...
                # -> in the case of no filter on the materialize we cache a noop
                # filter to act as the original filter
                if node._path_filter is not None:
                    if hasattr(node._path_filter, "name_group"):
                        node._path_filter.name_group = self._name_group
                    if not hasattr(node, "_original_filter"):
                        node._original_filter = node._path_filter
                    node._path_filter = lambda p: (node._original_filter(p) and mrr.filter(p))  # type: ignore
                else:
                    node._path_filter = mrr.filter
                    node._original_filter = lambda p: True
            else:
                assert (
                    len(node.children) != 0
                ), f"""Reliability pipeline cannot have node {type(node)} as first node.\n
                Only BinaryScan and Materialize nodes are allowed."""

            assert len(node.children) < 2, "Reliablity pipeline should only have one/zero child"

            self.count += 1

        return visit

    def _refresh_seen_files(self):
        """Refresh the list of already processed files"""
        from pyarrow.fs import FileSelector

        files = self.fs.get_file_info(FileSelector(self.path, allow_not_found=True))
        self.seen = {
            self._name_group.materialize_name_to_docid_safe(str(f.path)): f.mtime
            for f in files
            if self._name_group.materialize_name_to_docid_safe(str(f.path)) is not None
            and not self._name_group.is_metadata_materialize_name(str(f.path))
        }
        logger.info(f"Found {len(self.seen)} already materialized outputs")
        if len(self.seen) == self.prev_seen:
            self.retries_count += 1
        else:
            self.retries_count = 0

    def filter(self, p: str, read_binary: bool = False) -> bool:
        """Filter files for processing, respecting batch size"""
        if self.current_batch >= self.max_batch:
            print(" - False: over batch size")
            return False
        if not read_binary:
            id = self._name_group.materialize_name_to_docid_safe(p)
        else:
            id = self._name_group.docpath_to_docid(str(p))
        if id is None:
            logger.debug(f"Got path {p} not in proper format")
            return False

        if id in self.seen:
            print(" - False: already seen")
            return False

        if not self._name_group.is_metadata_materialize_name(p):
            self.current_batch += 1
            print(" - True: new, non-metadata")
            return True
        print(" - True: metadata")
        return True

    def reset_batch(self) -> None:
        """Reset the current batch counter and refresh seen files"""
        self.current_batch = 0
        self.prev_seen = len(self.seen)
        self._refresh_seen_files()

    def clear_console(self) -> None:
        """Hook to clear output and print status before each iteration."""
        from IPython.display import clear_output

        clear_output(wait=True)
        self.iteration += 1
        print(f"Starting iteration: {self.iteration}")
        if self.cycle_error != "":
            print(f"Previous batch error: {self.cycle_error}. \nProcessed {len(self.seen)} docs until now.")
            self.cycle_error = ""
        else:
            print(f"No errors in previous batch. \nProcessed {len(self.seen)} docs at present.")


def _success_path(base_path: Path) -> Path:
    return base_path / "materialize.success"


class Materialize(UnaryNode):
    def __init__(
        self,
        child: Optional[Node],
        context: Context,
        path: Optional[Union[Path, str, dict]] = None,
        source_mode: MaterializeSourceMode = MaterializeSourceMode.RECOMPUTE,
        tolerate_input_errors: bool = False,
        **kwargs,
    ):
        assert child is None or isinstance(child, Node)
        self._orig_path = path
        self._root = None
        self._path_filter = None
        self._tolerate_input_errors = tolerate_input_errors
        self._name_group: type[MaterializeNameGroup] = RandomNameGroup
        if path is None:
            pass
        elif isinstance(path, str) or isinstance(path, Path):
            (self._fs, self._root) = self.infer_fs(str(path))
            self._fshelper = _PyArrowFsHelper(self._fs)
            self._doc_to_name = self._name_group.doc_to_materialize_name
            self._doc_to_binary = Document.serialize
            self._clean_root = True
        elif isinstance(path, dict):
            assert "root" in path, "Need to specify root in materialize(path={})"
            if "fs" in path:
                self._fs = path["fs"]
                self._root = Path(path["root"])
            else:
                (self._fs, self._root) = self.infer_fs(str(path["root"]))
            self._fshelper = _PyArrowFsHelper(self._fs)
            namer = path.get("name", None)
            if inspect.isclass(namer) and issubclass(namer, MaterializeNameGroup):
                self._name_group = namer
                self._doc_to_name = namer.doc_to_materialize_name
            elif callable(namer):
                self._doc_to_name = namer
                logger.warn(
                    "Found floating materialize-file naming function. "
                    "Some operations (MRR, materialize filter) may not work."
                )
            else:
                assert namer is None, f"Found unexpected value for name field: {namer}"
                self._doc_to_name = self._name_group.doc_to_materialize_name
            assert callable(self._doc_to_name)
            self._doc_to_binary = path.get("tobin", Document.serialize)
            self._clean_root = path.get("clean", True)
            self._path_filter = path.get("filter", None)
            if hasattr(self._path_filter, "name_group"):
                self._path_filter.name_group = self._name_group  # type: ignore # doesn't understand hasattr
        else:
            assert False, f"unsupported type ({type(path)}) for path argument, expected str, Path, or dict"

        if source_mode != MaterializeSourceMode.RECOMPUTE:
            assert path is not None
            assert (
                self._doc_to_binary == Document.serialize
            ), "Using materialize in source mode requires default serialization"
            assert self._clean_root, "Using materialize in source mode requires cleaning the root"

        self._source_mode = source_mode
        self._executed_child = False

        super().__init__(child, **kwargs)

        self._maybe_anonymous()

    def _maybe_anonymous(self) -> None:
        if self._root is None:
            return
        from pyarrow.fs import S3FileSystem

        if not isinstance(self._fs, S3FileSystem):
            return

        try:
            self._fs.get_file_info(str(self._root))
            return
        except OSError as e:
            logging.warning(f"Got error {e} trying to get file info on {self._root}, trying again in anonymous mode")

        fs = S3FileSystem(anonymous=True)
        try:
            fs.get_file_info(str(self._root))
            self._fs = fs
            self._fshelper = _PyArrowFsHelper(self._fs)
            logger.info(f"Successfully read path {self._root} with anonymous S3")
            return
        except OSError as e:
            logging.warning(
                f"Got error {e} trying to anonymously get file info on {self._root}. Likely to fail shortly."
            )
            return

    def prepare(self):
        """
        Clean up the materialize location if necessary.
        Validate that cleaning worked, but only once all materializes have finished cleaning.
        This protects against multiple materializes pointing to the same location.
        """

        if self._root is None:
            return

        if self._will_be_source():
            return

        if not self._clean_root:
            self._fs.create_dir(str(self._root))
            return

        self._fshelper.safe_cleanup(self._root)

        def check_clean():
            clean_path = self._root / "materialize.clean"
            if self._fshelper.file_exists(clean_path):
                raise ValueError(
                    f"path {clean_path} already exists despite cleaning the root directory."
                    + " Most likely there two materialize nodes are using the same path"
                )
            self._fs.open_output_stream(str(clean_path)).close()
            assert self._fshelper.file_exists(
                clean_path
            ), f"{clean_path} was just created in {self._fs}, but does not exist?!"

        return check_clean

    def execute(self, **kwargs) -> "Dataset":
        logger.debug("Materialize execute")
        if self._source_mode == MaterializeSourceMode.USE_STORED:
            success = self._fshelper.file_exists(self._success_path())
            if success or len(self.children) == 0:
                logger.info(f"Using {self._orig_path} as the cached source of data")

                self._executed_child = False
                if not success:
                    self._verify_has_files()
                    logging.warning(f"materialize.success not found in {self._orig_path}. Returning partial data")

                from ray.data import read_binary_files
                from ray.data.datasource import PathPartitionFilter, PathPartitionParser

                partition_filter = None
                if self._path_filter is not None:
                    partition_filter = PathPartitionFilter(
                        cast(PathPartitionParser, RayPathParser()), self._path_filter
                    )
                shuffle = None if partition_filter is None else "files"

                try:
                    files = read_binary_files(
                        self._root,
                        filesystem=self._fs,
                        file_extensions=["pickle"],
                        partition_filter=partition_filter,
                        shuffle=shuffle,
                    )

                    return files.flat_map(self._ray_to_document)
                except ValueError as e:
                    from ray.data import from_items

                    if "No input files found to read." in str(e):
                        logger.warning("No more files found during reliability step.")
                        return from_items(items=[])

                    if "No input files found to read with the following file extensions" not in str(e):
                        raise
                logger.warning(
                    f"Unable to find any .pickle files in {self._root}, but either"
                    " there is a materialize.success or this is a start node."
                )

                return from_items(items=[])

        self._executed_child = True
        # right now, no validation happens, so save data in parallel. Once we support validation
        # to support retries we won't be able to run the validation in parallel.  non-shared
        # filesystems will also eventually be a problem but we can put it off for now.

        input_dataset = self.child().execute(**kwargs)
        if self._root is not None:
            import numpy

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

        raise ValueError(
            f"""Materialize root {self._orig_path} has no .pickle files.
            If using reliability, make sure to write doc ids using 'docid_from_path'."""
        )

    def _ray_to_document(self, dict: dict[str, Any]) -> list[dict[str, bytes]]:
        b = dict["bytes"]
        if len(b) == 0:
            if self._tolerate_input_errors:
                logger.info("Dropping empty doc cause of tolerate_input_errors")
                return []
            else:
                logger.warning("Found empty input doc, pipeline is gonna fail")
        return [{"doc": b}]

    def _will_be_source(self) -> bool:
        if len(self.children) == 0:
            return True
        return self._source_mode == MaterializeSourceMode.USE_STORED and self._fshelper.file_exists(
            self._success_path()
        )

    def local_execute(self, docs: list[Document]) -> list[Document]:
        if self._source_mode == MaterializeSourceMode.USE_STORED:
            if self._fshelper.file_exists(self._success_path()):
                self._executed_child = False
                logger.info(f"Using {self._orig_path} as cached source of data")

                return self.local_source()

        if self._root is not None:
            for d in docs:
                self.save(d)
            self._executed_child = True
        return docs

    def local_source(self) -> list[Document]:
        assert self._root is not None
        self._verify_has_files()
        logger.info(f"Using {self._orig_path} as cached source of data")
        if not self._fshelper.file_exists(self._success_path()):
            logging.warning(f"materialize.success not found in {self._orig_path}. Returning partial data")
        from sycamore.utils.sycamore_logger import RateLimitLogger

        limited_logger = logging.getLogger(__name__ + ".limited_local_source")
        limited_logger.addFilter(RateLimitLogger())
        ret = []
        count = 0
        for fi in self._fshelper.list_files(self._root):
            if fi.size == 0:
                continue
            if self._path_filter is not None and not self._path_filter(fi.path):
                continue
            n = Path(fi.path)
            if n.suffix == ".pickle":
                limited_logger.info(f"  reading file {count} from {str(n)}")
                count = count + 1
                f = self._fs.open_input_stream(str(n))
                ret.append(Document.deserialize(f.read()))
                f.close()
        logger.info(f"  read {count} total files")

        return ret

    def load_metadata(self) -> list[MetadataDocument]:
        self._verify_has_files()
        if not self._fshelper.file_exists(self._success_path()):
            logging.warning(f"materialize.success not found in {self._orig_path}. Returning partial data")
        from sycamore.utils.sycamore_logger import RateLimitLogger

        limited_logger = logging.getLogger(__name__ + ".load_metadata")
        limited_logger.addFilter(RateLimitLogger())
        ret = []
        count = 0
        for fi in self._fshelper.list_files(self._root):
            if self._path_filter is not None and not self._path_filter(fi.path):
                continue
            n = Path(fi.path)
            if n.name.startswith("md-") and n.suffix == ".pickle":
                limited_logger.info(f"  reading file {count} from {str(n)}")
                count += 1
                f = self._fs.open_input_stream(str(n))
                try:
                    doc = Document.deserialize(f.read())
                finally:
                    f.close()
                assert isinstance(doc, MetadataDocument), f"md-*.pickle file has wrong type {doc}"
                ret.append(doc)

        logger.info(f"  read {count} total files")

        return ret

    def _success_path(self):
        return _success_path(self._root)

    def finalize(self):
        if not self._executed_child:
            return
        if self._root is not None:
            self._fs.open_output_stream(str(self._success_path())).close()
            assert self._fshelper.file_exists(self._success_path())

    @staticmethod
    def infer_fs(path: str) -> Tuple["pyarrow.FileSystem", Path]:
        from sycamore.utils.pyarrow import infer_fs as util_infer_fs

        (fs, path) = util_infer_fs(path)
        return (fs, Path(path))

    def save(self, doc: Document) -> None:
        bin = self._doc_to_binary(doc)
        if bin is None:
            return
        assert isinstance(bin, bytes), f"tobin function returned {type(bin)} not bytes"
        assert self._root is not None
        name = self._doc_to_name(doc, bin)
        assert isinstance(name, str) or isinstance(
            name, Path
        ), f"doc_to_name function turned docid {doc.doc_id} into {name} -- should be string or Path"
        path = self._root / name

        if self._clean_root and self._fshelper.file_exists(path):
            if self._doc_to_name != RandomNameGroup.doc_to_materialize_name:
                # default doc_to_name includes a content based hash, so "duplicate" entries
                # should only be possible if ray executes the save operation multiple times on
                # the same content.
                logger.warn(
                    f"Duplicate name {path} generated for clean root;"
                    + " this could be ray re-execution or fault tolerance; first written data kept"
                )

            return
        with self._fs.open_output_stream(str(path)) as out:
            out.write(bin)


class RayPathParser:
    def __call__(self, path: str) -> Path:
        return Path(path)


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

       # The source_mode can be passed to AutoMaterialize, which will in turn be passed to all
       # created materialize nodes. To use each materialized node as a source:
       ctx.rewrite_rules.append(AutoMaterialize(source_mode=sycamore.MATERIALIZED_USE_STORED)

    Nodes in the plan will automatically be named. You can specify a name by defining it for the node:
       ctx = sycamore.init()
       ds = ctx.read.document(docs, materialize={"name": "reader"}).map(noop_fn, materialize={"name": "noop"})
       # NOTE: names in a single execution must be unique. This is guaranteed by auto naming
       # NOTE: automatic names are not guaranteed to be stable
    """

    def __init__(self, path: Union[str, Path, dict] = {}, source_mode=MaterializeSourceMode.RECOMPUTE):
        super().__init__()
        if isinstance(path, str) or isinstance(path, Path):
            path = {"root": path}
        else:
            path = path.copy()
        if "clean" not in path:
            path["clean"] = True

        self._choose_directory(path)
        self._basename_to_count: dict[str, int] = {}
        self._source_mode = source_mode

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

            if self._source_mode == MaterializeSourceMode.USE_STORED and self._fshelper.file_exists(
                _success_path(path)
            ):
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
            return Materialize(node, context, path=path, source_mode=self._source_mode)

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


def clear_materialize(plan: Node, *, path: Optional[Union[Path, str]], clear_non_local: bool):
    """See docset.clear_materialize() for documentation"""
    from pyarrow.fs import LocalFileSystem

    if isinstance(path, Path):
        path = str(path)  # pathlib.PurePath.match requires the match to be a string
    if path is None:
        path = "*"

    def clean_dir(n: Node):
        if not isinstance(n, Materialize):
            return
        if n._root is None:
            return
        if not (isinstance(n._fs, LocalFileSystem) or clear_non_local):
            logger.info(f"Skipping clearing non-local path {n._orig_path}")
            return
        if not n._root.match(path):
            return
        # safe_cleanup logs
        n._fshelper.safe_cleanup(n._root)

    plan.traverse(visit=clean_dir)


# This is a class so Materialize can change the name group post initialize
class DocIdFilter:
    """
    Filter docids in a materialize step. Useful for debugging. Use like so:

     .. code-block::python

        doc_ids = ["list", "of", "docids"]
        ds = ctx.read.materialize(path={"root": "materializedir", "filter": DocIdFilter(doc_ids)})

    Args:
        doc_ids: The list of doc ids to read from materialize
        name_group: The naming scheme for materialize filenames and doc_ids. If using
            defaults in materialize, this defaults to the correct thing.
    """

    def __init__(self, doc_ids: list[str], name_group: type[MaterializeNameGroup] = RandomNameGroup):
        self.doc_id_set = set(doc_ids)
        self.name_group = name_group

    def filter(self, p: str) -> bool:
        logger.info(p)
        did = self.name_group.materialize_name_to_docid_safe(p)
        if did is None:
            return False
        if self.name_group is RandomNameGroup:
            candidates = [
                did,
                did[len("aryn:") :],
                did[len("d-") :],
                did[len("aryn:d-") :],
            ]
        else:
            candidates = [did]
        logger.info(candidates)
        return any(c in self.doc_id_set for c in candidates)

    def __call__(self, p: str) -> bool:
        return self.filter(p)
