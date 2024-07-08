from dataclasses import dataclass, field
from typing import Any, Optional
from sycamore.connectors.common import drop_types, flatten_data
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.connectors.base import BaseDBWriter

from elasticsearch import Elasticsearch, ApiError
from elasticsearch.helpers import parallel_bulk


@dataclass
class ElasticClientParams(BaseDBWriter.ClientParams):
    es_client_args: dict = field(default_factory=lambda: {})
    url: Optional[str] = None


@dataclass
class ElasticTargetParams(BaseDBWriter.TargetParams):
    index_name: str
    mappings: dict[str, Any] = field(
        default_factory=lambda: {
            "properties": {
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": "true",
                    "similarity": "cosine",
                },
                "properties": {"type": "json"},
            }
        }
    )
    flatten_properties: bool = False

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, ElasticTargetParams):
            return False
        if self.index_name != other.index_name:
            return False
        if self.mappings != other.mappings:
            return False
        return True


class ElasticClient(BaseDBWriter.Client):
    def __init__(self, client: Elasticsearch):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "ElasticClient":
        assert isinstance(params, ElasticClientParams)
        client = Elasticsearch(**params.es_client_args)
        if params.url:
            client = Elasticsearch(params.url)
        return ElasticClient(client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, ElasticTargetParams)
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        with self._client:

            def insert_into_index():
                for r in records:
                    yield {
                        "_index": target_params.index_name,
                        "_id": r.doc_id,
                        "properties": r.properties,
                        "embeddings": r.embeddings,
                    }

            for success, info in parallel_bulk(self._client, insert_into_index()):  # generator must be consumed
                if not success:
                    print(f"Insert Operation Unsuccessful: {info}")

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, ElasticTargetParams)
        try:
            with self._client:
                self._client.indices.create(
                    index=target_params.index_name, mappings=target_params.mappings, timeout="30"
                )
        except ApiError as e:
            if e.status_code == 400:
                return
            raise e

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "ElasticTargetParams":
        assert isinstance(target_params, ElasticTargetParams)
        with self._client:
            data = self._client.indices.get_mapping(index=target_params.index_name)
            mapping_keys = data[target_params.index_name]["mappings"].keys()
            return ElasticTargetParams(
                index_name=target_params.index_name,
                mappings=mapping_keys,
                flatten_properties=target_params.flatten_properties,
            )


@dataclass
class ElasticDocumentRecord(BaseDBWriter.Record):
    doc_id: str
    properties: dict
    embeddings: list[float]

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "ElasticDocumentRecord":
        assert isinstance(target_params, ElasticTargetParams)
        doc_id = document.doc_id
        embedding = document.embedding
        if doc_id is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        if embedding is None:
            raise ValueError(f"Cannot write documents without an embedding. Found {document}")
        properties = {
            "properties": document.properties,
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.coordinates if document.bbox else None,
            "shingles": document.shingles,
        }
        droperties = drop_types(properties, drop_empty_lists=True, drop_empty_dicts=True)
        if target_params.flatten_properties:
            # Property names must be [a-zA-Z][_0-9a-zA-Z]{0,230}, so use __ as a separator rather than .
            droperties = dict(flatten_data(droperties, allowed_list_types=[int, str, bool, float], separator="__"))
            droperties = {k.replace("-", "_"): v for k, v in droperties.items()}  # e.g. properties__y-index is invalid
        assert isinstance(droperties, dict)
        return ElasticDocumentRecord(doc_id=doc_id, properties=droperties, embeddings=embedding)


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[ElasticDocumentRecord]]:
    return all(isinstance(r, ElasticDocumentRecord) for r in records)


class ElasticDocumentWriter(BaseDBWriter):
    Client = ElasticClient
    Record = ElasticDocumentRecord
    ClientParams = ElasticClientParams
    TargetParams = ElasticTargetParams
