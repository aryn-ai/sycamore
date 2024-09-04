from dataclasses import dataclass, field
import typing
from typing import Any, Optional
from typing_extensions import TypeGuard
from sycamore.utils.import_utils import requires_modules

from sycamore.data.document import Document
from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.connectors.common import flatten_data, check_dictionary_compatibility

if typing.TYPE_CHECKING:
    from elasticsearch import Elasticsearch


@dataclass
class ElasticsearchWriterClientParams(BaseDBWriter.ClientParams):
    url: str
    es_client_args: dict = field(default_factory=lambda: {})


@dataclass
class ElasticsearchWriterTargetParams(BaseDBWriter.TargetParams):
    index_name: str
    settings: dict[str, Any] = field(default_factory=lambda: {})
    mappings: dict[str, Any] = field(
        default_factory=lambda: {
            "properties": {
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
                "properties": {"type": "object"},
            }
        }
    )
    wait_for_completion: str = "false"

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, ElasticsearchWriterTargetParams):
            return False
        if self.index_name != other.index_name:
            return False
        if self.wait_for_completion != other.wait_for_completion:
            return False
        my_flat_settings = dict(flatten_data(self.settings))
        other_flat_settings = dict(flatten_data(other.settings))
        for k in my_flat_settings:
            other_k = k
            if k not in other_flat_settings:
                if "index." + k in other_flat_settings:
                    # You can specify index params without the "index" part and
                    # they'll come back with the "index" part
                    other_k = "index." + k
                else:
                    return False
            if my_flat_settings[k] != other_flat_settings[other_k]:
                return False
        my_flat_mappings = dict(flatten_data(self.mappings))
        other_flat_mappings = dict(flatten_data(other.mappings))
        return check_dictionary_compatibility(my_flat_mappings, other_flat_mappings, ["type"])


class ElasticsearchWriterClient(BaseDBWriter.Client):
    def __init__(self, client: "Elasticsearch"):
        self._client = client

    @classmethod
    @requires_modules(["elasticsearch", "elasticsearch.helpers"], extra="elasticsearch")
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "ElasticsearchWriterClient":
        from elasticsearch import Elasticsearch

        assert isinstance(params, ElasticsearchWriterClientParams)
        client = Elasticsearch(params.url, **params.es_client_args)
        return ElasticsearchWriterClient(client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        from elasticsearch.helpers import parallel_bulk

        assert isinstance(target_params, ElasticsearchWriterTargetParams)
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        with self._client:

            def bulk_action_generator():
                for r in records:
                    yield {
                        "_index": target_params.index_name,
                        "_id": r.doc_id,
                        "properties": r.properties,
                        "embeddings": r.embeddings,
                    }

            for success, info in parallel_bulk(
                self._client, bulk_action_generator(), refresh=target_params.wait_for_completion
            ):  # generator must be consumed
                if not success:
                    print(f"Insert Operation Unsuccessful: {info}")

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        from elasticsearch import ApiError

        assert isinstance(target_params, ElasticsearchWriterTargetParams)
        try:
            self._client.indices.create(
                index=target_params.index_name,
                mappings=target_params.mappings,
                settings=target_params.settings,
                timeout="30s",
            )
        except ApiError:
            return

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "ElasticsearchWriterTargetParams":
        assert isinstance(target_params, ElasticsearchWriterTargetParams)
        mappings_data = self._client.indices.get_mapping(index=target_params.index_name)
        mappings = mappings_data[target_params.index_name]["mappings"]
        settings_data = self._client.indices.get_settings(index=target_params.index_name)
        settings = settings_data[target_params.index_name]["settings"]
        return ElasticsearchWriterTargetParams(
            index_name=target_params.index_name,
            mappings=mappings,
            settings=settings,
            wait_for_completion=target_params.wait_for_completion,
        )


@dataclass
class ElasticsearchWriterDocumentRecord(BaseDBWriter.Record):
    doc_id: str
    properties: dict
    embeddings: Optional[list[float]]

    @classmethod
    def from_doc(
        cls, document: Document, target_params: BaseDBWriter.TargetParams
    ) -> "ElasticsearchWriterDocumentRecord":
        assert isinstance(target_params, ElasticsearchWriterTargetParams)
        doc_id = document.doc_id
        embedding = document.embedding
        if doc_id is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        properties = {
            "properties": document.properties,
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.coordinates if document.bbox else None,
            "shingles": document.shingles,
        }
        return ElasticsearchWriterDocumentRecord(doc_id=doc_id, properties=properties, embeddings=embedding)


def _narrow_list_of_doc_records(
    records: list[BaseDBWriter.Record],
) -> TypeGuard[list[ElasticsearchWriterDocumentRecord]]:
    return all(isinstance(r, ElasticsearchWriterDocumentRecord) for r in records)


class ElasticsearchDocumentWriter(BaseDBWriter):
    Client = ElasticsearchWriterClient
    Record = ElasticsearchWriterDocumentRecord
    ClientParams = ElasticsearchWriterClientParams
    TargetParams = ElasticsearchWriterTargetParams
