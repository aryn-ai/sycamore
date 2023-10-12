import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from pyarrow.filesystem import FileSystem
from ray.data import Dataset, read_binary_files, read_json
from ray.data.datasource import FileExtensionFilter

from sycamore.data import Document
from sycamore.plan_nodes import Scan


def _set_id(doc: dict[str, Any]) -> dict[str, Any]:
    import uuid

    doc["doc_id"] = str(uuid.uuid1())
    return doc


class FileMetadataProvider(ABC):
    @abstractmethod
    def get_metadata(self, file_path: str) -> dict[str, Any]:
        pass


class JsonManifestMetadataProvider(FileMetadataProvider):
    def __init__(self, manifest_path: str) -> None:
        super().__init__()
        self._manifest_path = manifest_path
        self._path_to_metadata_map = self._load_json_manifest()

    def get_metadata(self, file_path: str) -> dict[str, Any]:
        return self._path_to_metadata_map.get(file_path, {})

    def _load_json_manifest(self) -> dict[str, Any]:
        try:
            with open(self._manifest_path, "r") as manifest_file:
                metadata_map = json.load(manifest_file)
            return metadata_map
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON manifest file not found at '{self._manifest_path}'")


class FileScan(Scan):
    """A base scan class for file based data"""

    def __init__(self, paths: Union[str, list[str]], *, parallelism: Optional[int] = None, **resource_args):
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
        paths: Union[str, list[str]],
        *,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional["FileSystem"] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        **resource_args,
    ):
        super().__init__(paths, parallelism=parallelism, **resource_args)
        self._paths = paths
        self.parallelism = -1 if parallelism is None else parallelism
        self._binary_format = binary_format
        self._filesystem = filesystem
        self._metadata_provider = metadata_provider

    def _is_s3_scheme(self):
        if isinstance(self._paths, str):
            return self._paths.startswith("s3:")
        else:
            return all(path.startswith("s3:") for path in self._paths)

    def _to_document(self, dict: dict[str, Any]) -> dict[str, bytes]:
        document = Document()
        import uuid

        document.doc_id = str(uuid.uuid1())
        document.type = self._binary_format
        document.binary_representation = dict["bytes"]

        properties = document.properties
        if self._is_s3_scheme():
            dict["path"] = "s3://" + dict["path"]
        properties.update({"path": dict["path"]})
        if self._metadata_provider:
            properties.update(self._metadata_provider.get_metadata(dict["path"]))
        document.properties = properties

        return {"doc": document.serialize()}

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

        return files.map(self._to_document, **self.resource_args)

    def format(self):
        return self._binary_format


class JsonScan(FileScan):
    def __init__(self, paths: Union[str, list[str]], *, parallelism: Optional[int] = None, **resource_args):
        super().__init__(paths, parallelism=parallelism, **resource_args)
        self.parallelism = -1 if parallelism is None else parallelism

    def execute(self) -> "Dataset":
        json = read_json(paths=self._paths, parallelism=self.parallelism, **self.resource_args)
        return json

    def format(self):
        return "json"
