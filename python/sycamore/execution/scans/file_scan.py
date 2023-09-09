from typing import Any, Dict, List, Optional, Union

from pyarrow.filesystem import FileSystem
from ray.data import Dataset, read_binary_files, read_json
from ray.data.datasource import FileExtensionFilter

from sycamore.data import Document
from sycamore.execution import Scan


def _set_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    import uuid

    doc["doc_id"] = str(uuid.uuid1())
    return doc


class FileScan(Scan):
    """A base scan class for file based data"""

    def __init__(self, paths: Union[str, List[str]], *, parallelism: Optional[int] = None, **resource_args):
        super().__init__(**resource_args)
        self._paths = paths
        self.parallelism = parallelism


class BinaryScan(FileScan):
    """Scan data file into raw bytes

    For each file, BinaryScan creates one Document in the form of
    {"doc_id": uuid,
     "content": {"binary": xxx, "text": None},
      "properties": {"path": xxx}}.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        *,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional["FileSystem"] = None,
        **resource_args
    ):
        super().__init__(paths, parallelism=parallelism, **resource_args)
        self._paths = paths
        self.parallelism = -1 if parallelism is None else parallelism
        self._binary_format = binary_format
        self._filesystem = filesystem

    def _to_document(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document()
        import uuid

        document.doc_id = str(uuid.uuid1())
        document.type = self._binary_format
        document.content = dict["bytes"]
        document.properties.update({"path": dict["path"]})
        return document.data

    def _is_s3_scheme(self):
        if isinstance(self._paths, str):
            return self._paths.startswith("s3:")
        else:
            return all(path.startswith("s3:") for path in self._paths)

    def execute(self) -> "Dataset":
        partition_filter = FileExtensionFilter(self.format())
        files = read_binary_files(
            self._paths,
            include_paths=True,
            filesystem=self._filesystem,
            parallelism=self.parallelism,
            partition_filter=partition_filter,
            ray_remote_args=self.resource_args,
        )

        def prepend_scheme(file):
            file["path"] = "s3://" + file["path"]
            return file

        if self._is_s3_scheme():
            files = files.map(prepend_scheme, **self.resource_args)

        return files.map(self._to_document, **self.resource_args)

    def format(self):
        return self._binary_format


class JsonScan(FileScan):
    def __init__(self, paths: Union[str, List[str]], *, parallelism: Optional[int] = None, **resource_args):
        super().__init__(paths, parallelism=parallelism, **resource_args)
        self.parallelism = -1 if parallelism is None else parallelism

    def execute(self) -> "Dataset":
        json = read_json(paths=self._paths, parallelism=self.parallelism, **self.resource_args)
        return json

    def format(self):
        return "json"
