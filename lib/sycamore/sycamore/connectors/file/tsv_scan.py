import csv
import io
from typing import Any, Optional, Union, List, Dict

import ray.data
from pyarrow.fs import FileInfo, FileSystem
from ray.data import Dataset
from ray.data.datasource import FileExtensionFilter

from sycamore.data import Document
from sycamore.connectors.file.file_scan import FileScan
from sycamore.connectors.file.file_metadata_provider import FileMetadataProvider
from sycamore.utils.doc_id import mkdocid


class TsvScan(FileScan):
    def __init__(
        self,
        paths: Union[str, List[str]],
        filesystem: Optional[FileSystem] = None,
        parallelism: Optional[str] = None,  # Will be deprecated, use override_num_blocks
        override_num_blocks: Optional[int] = None,
        tsv_reader_options: Optional[dict[str, Any]] = None,
        document_body_field: Optional[str] = None,
        property_fields: Optional[List[str]] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        **resource_args,
    ):
        super().__init__(
            paths,
            filesystem=filesystem,
            parallelism=parallelism, # TODO: Deprecate parallelism
            override_num_blocks=override_num_blocks,
            metadata_provider=metadata_provider,
            **resource_args,
        )
        self._tsv_reader_options = tsv_reader_options or {}
        self._tsv_reader_options.setdefault('delimiter', '\t')
        self._document_body_field = document_body_field
        self._property_fields = property_fields

    def _row_to_document(self, row: dict, file_path: str) -> Document:
        doc = Document()
        doc.doc_id = mkdocid(prefix="tsv")
        doc.type = "tsv"

        if self._document_body_field and self._document_body_field in row:
            doc.text_representation = str(row[self._document_body_field])

        properties = {}
        if self._property_fields:
            for field in self._property_fields:
                if field in row:
                    properties[field] = row[field]
        else:
            for key, value in row.items():
                if self._document_body_field and key == self._document_body_field:
                    continue
                properties[key] = value

        properties["path"] = file_path

        if self._metadata_provider:
            properties.update(self._metadata_provider.get_metadata(file_path))

        doc.properties = properties
        return doc

    def format(self) -> str:
        return "tsv"

    def _process_ray_file_bytes(self, file_bytes_dict: dict[str, bytes]) -> List[Dict[str, Any]]:
        file_path = file_bytes_dict["path"]
        content_bytes = file_bytes_dict["bytes"]
        content_string = content_bytes.decode("utf-8")  # Assuming UTF-8, make configurable if needed

        documents = []
        # Use io.StringIO to treat the string as a file
        with io.StringIO(content_string) as string_file:
            reader = csv.DictReader(string_file, **self._tsv_reader_options)
            for row in reader:
                doc = self._row_to_document(row, file_path)
                documents.append({"doc": doc.serialize()})
        return documents

    def execute(self, **kwargs) -> "Dataset":
        file_paths = self._get_file_paths()

        dataset = ray.data.read_binary_files(
            paths=file_paths,
            filesystem=self._filesystem,
            override_num_blocks=self.override_num_blocks,
            # parallelism=self._parallelism, # TODO: use override_num_blocks
            ray_remote_args=self.resource_args,
            file_extensions=FileExtensionFilter("tsv"),
        )

        doc_dataset = dataset.flat_map(self._process_ray_file_bytes)
        return doc_dataset

    def process_file(self, file_info: FileInfo) -> List[Document]:
        if not file_info.is_file or not file_info.base_name.lower().endswith(".tsv"):
            return []

        documents = []
        with self._filesystem.open_input_stream(file_info.path) as f:
            content_bytes = f.read()
            content_string = content_bytes.decode("utf-8") # Assuming UTF-8

            with io.StringIO(content_string) as string_file:
                reader = csv.DictReader(string_file, **self._tsv_reader_options)
                for row in reader:
                    doc = self._row_to_document(row, file_info.path)
                    documents.append(doc)
        return documents
