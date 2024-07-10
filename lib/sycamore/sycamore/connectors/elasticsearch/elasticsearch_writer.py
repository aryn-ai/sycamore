from dataclasses import dataclass, field
from typing import Any, Optional
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.connectors.base import BaseDBWriter

from elasticsearch import Elasticsearch, ApiError
from elasticsearch.helpers import parallel_bulk


@dataclass
class ElasticClientParams(BaseDBWriter.ClientParams):
    url: str
    es_client_args: dict = field(default_factory=lambda: {})


@dataclass
class ElasticTargetParams(BaseDBWriter.TargetParams):
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
        if not isinstance(other, ElasticTargetParams):
            return False
        if self.index_name != other.index_name:
            return False
        if self.wait_for_completion != other.wait_for_completion:
            return False
        if not compare_nested_dicts(self.mappings, other.mappings):
            return False
        return compare_nested_dicts(self.settings, other.settings)


class ElasticClient(BaseDBWriter.Client):
    def __init__(self, client: Elasticsearch):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "ElasticClient":
        assert isinstance(params, ElasticClientParams)
        client = Elasticsearch(params.url, **params.es_client_args)
        return ElasticClient(client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, ElasticTargetParams)
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
        assert isinstance(target_params, ElasticTargetParams)
        try:
            self._client.indices.create(
                index=target_params.index_name,
                mappings=target_params.mappings,
                settings=target_params.settings,
                timeout="30s",
            )
        except ApiError as e:
            if e.status_code == 400 and "resource_already_exists_exception" in e.message:
                return
            raise e

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "ElasticTargetParams":
        assert isinstance(target_params, ElasticTargetParams)
        mappings_data = self._client.indices.get_mapping(index=target_params.index_name)
        mappings = mappings_data[target_params.index_name]["mappings"]
        settings_data = self._client.indices.get_settings(index=target_params.index_name)
        settings = settings_data[target_params.index_name]["settings"]
        return ElasticTargetParams(
            index_name=target_params.index_name,
            mappings=mappings,
            settings=settings,
            wait_for_completion=target_params.wait_for_completion,
        )


@dataclass
class ElasticDocumentRecord(BaseDBWriter.Record):
    doc_id: str
    properties: dict
    embeddings: Optional[list[float]]

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "ElasticDocumentRecord":
        assert isinstance(target_params, ElasticTargetParams)
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
        return ElasticDocumentRecord(doc_id=doc_id, properties=properties, embeddings=embedding)


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[ElasticDocumentRecord]]:
    return all(isinstance(r, ElasticDocumentRecord) for r in records)


def compare_nested_dicts(dict1: dict[Any, Any], dict2: dict[Any, Any]):
    for key, value in dict1.items():
        if value == "object":
            continue
        if key not in dict2:
            return False
        if isinstance(value, dict):
            if not isinstance(dict2[key], dict):
                return False
            if not compare_nested_dicts(value, dict2[key]):
                return False
        elif dict1[key] != dict2[key]:
            return False
    return True


class ElasticDocumentWriter(BaseDBWriter):
    Client = ElasticClient
    Record = ElasticDocumentRecord
    ClientParams = ElasticClientParams
    TargetParams = ElasticTargetParams
