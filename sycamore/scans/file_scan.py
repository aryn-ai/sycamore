import json
from abc import ABC, abstractmethod
import boto3
from typing import Any, Optional, Union, Tuple, Callable
import uuid

from pyarrow.filesystem import FileSystem
from ray.data import Dataset, read_binary_files, read_json
from ray.data.datasource import FileExtensionFilter

from sycamore.data import Document
from sycamore.plan_nodes import Scan


def _set_id(doc: dict[str, Any]) -> dict[str, Any]:
    doc["doc_id"] = str(uuid.uuid1())
    return doc


class FileMetadataProvider(ABC):
    @abstractmethod
    def get_metadata(self, file_path: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_paths(self) -> list[str]:
        pass


class JsonManifestMetadataProvider(FileMetadataProvider):
    def __init__(self, manifest_path: str) -> None:
        super().__init__()
        self._manifest_path = manifest_path
        self._path_to_metadata_map = self._load_json_manifest()

    def get_metadata(self, file_path: str) -> dict[str, Any]:
        return self._path_to_metadata_map.get(file_path, {})

    def get_paths(self) -> list[str]:
        return list(self._path_to_metadata_map.keys())

    def _load_json_manifest(self) -> dict[str, Any]:
        if self._manifest_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket_name, key = self._parse_s3_path(self._manifest_path)
            response = s3.get_object(Bucket=bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            metadata_map = json.loads(content)
            return metadata_map
        else:
            try:
                with open(self._manifest_path, "r") as manifest_file:
                    metadata_map = json.load(manifest_file)
                return metadata_map
            except FileNotFoundError:
                raise FileNotFoundError(f"JSON manifest file not found at '{self._manifest_path}'")

    @staticmethod
    def _parse_s3_path(s3_path: str) -> Tuple[str, str]:
        parts = s3_path[len("s3://") :].split("/", 1)
        bucket_name = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket_name, key


class FileScan(Scan):
    """A base scan class for file based data"""

    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        filesystem: Optional[FileSystem] = None,
        parallelism: Optional[int] = None,
        **resource_args,
    ):
        super().__init__(**resource_args)
        self._paths = paths
        self._filesystem = filesystem
        self.parallelism = parallelism

    def _is_s3_scheme(self) -> bool:
        if isinstance(self._paths, str):
            return self._paths.startswith("s3:")
        else:
            return all(path.startswith("s3:") for path in self._paths)


class BinaryScan(FileScan):
    """Scan data file into raw bytes

    For each file, BinaryScan creates one Document in the form of
    {"doc_id": uuid,
     "content": {"binary": xxx, "text": None},
      "properties": {"path": xxx}}.

    Note: if you specify filter_paths_by_extension = False, you need to make sure
    all the files that are scanned can be processed by the pipeline. Many pipelines
    include file-type specific steps.
    """

    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        filter_paths_by_extension: bool = True,
        **resource_args,
    ):
        super().__init__(paths, parallelism=parallelism, filesystem=filesystem, **resource_args)
        self._paths = paths
        self.parallelism = -1 if parallelism is None else parallelism
        self._binary_format = binary_format
        self._metadata_provider = metadata_provider
        self._filter_paths_by_extension = filter_paths_by_extension

    def _to_document(self, dict: dict[str, Any]) -> dict[str, bytes]:
        document = Document()

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
        if self._filter_paths_by_extension:
            partition_filter = FileExtensionFilter(self.format())
        else:
            partition_filter = None
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
    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        properties: Optional[Union[str, list[str]]] = None,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        document_body_field: Optional[str] = None,
        doc_extractor: Optional[Callable] = None,
        **resource_args,
    ):
        super().__init__(paths, parallelism=parallelism, filesystem=filesystem, **resource_args)
        self._properties = properties
        self.parallelism = -1 if parallelism is None else parallelism
        self._metadata_provider = metadata_provider
        self._document_body_field = document_body_field
        self._doc_extractor = doc_extractor

    def _to_document(self, json_dict: dict[str, Any]) -> list[dict[str, Any]]:
        document = Document()

        document.doc_id = str(uuid.uuid1())
        document.type = "json"

        if self._document_body_field is not None:
            body = json_dict.pop(self._document_body_field, None)
        else:
            body = json.dumps(json_dict)

        if body is not None:
            document.text_representation = body
            document.binary_representation = body.encode("utf-8")

        document.properties = self._extract_properties(json_dict)

        # TODO: What to do about name conflicts here?
        if self._is_s3_scheme():
            json_dict["path"] = "s3://" + json_dict["path"]
        document.properties.update({"path": json_dict["path"]})

        if self._metadata_provider:
            document.properties.update(self._metadata_provider.get_metadata(json_dict["path"]))

        return [{"doc": document.serialize()}]

    def _extract_properties(self, record: dict[str, Any]) -> dict[str, Any]:
        properties = {}
        if self._properties is None:
            return record
        elif isinstance(self._properties, str):
            if self._properties in record:
                properties[self._properties] = record.get(self._properties)
        else:
            for prop in self._properties:
                if prop in record:
                    properties[prop] = record.get(prop)

        return properties

    def execute(self) -> Dataset:
        json_dataset = read_json(
            self._paths,
            include_paths=True,
            filesystem=self._filesystem,
            parallelism=self.parallelism,
            ray_remote_args=self.resource_args,
        )

        doc_extractor = self._doc_extractor if self._doc_extractor else self._to_document
        return json_dataset.flat_map(doc_extractor, **self.resource_args)

    def format(self):
        return "json"
