from dataclasses import dataclass, asdict
import typing
from typing import Any, Optional, Union
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.utils.import_utils import requires_modules

if typing.TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass
class QdrantWriterClientParams(BaseDBWriter.ClientParams):
    location: Optional[str] = None
    url: Optional[str] = None
    port: Optional[int] = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    https: Optional[bool] = None
    api_key: Optional[str] = None
    prefix: Optional[str] = None
    timeout: Optional[int] = None
    host: Optional[str] = None
    path: Optional[str] = None


@dataclass
class QdrantWriterTargetParams(BaseDBWriter.TargetParams):
    collection_params: dict

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        return True


class QdrantWriterClient(BaseDBWriter.Client):
    @requires_modules("qdrant_client", extra="qdrant")
    def __init__(self, client: "QdrantClient"):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "QdrantWriterClient":
        assert isinstance(params, QdrantWriterClientParams)

        from qdrant_client import QdrantClient

        client = QdrantClient(**asdict(params))
        return cls(client)

    def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, QdrantWriterTargetParams)
        assert _narrow_list_of_qdrant_records(records), f"Found bad records in {records}"

        from qdrant_client.models import PointStruct

        points = [PointStruct(id=record.id, vector=record.vector, payload=record.payload) for record in records]

        self._client.upload_points(collection_name=target_params.collection_params["collection_name"], points=points)

    def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, QdrantWriterTargetParams)
        params = target_params.collection_params
        if not self._client.collection_exists(params["collection_name"]):
            self._client.create_collection(**params)

    def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> QdrantWriterTargetParams:
        assert isinstance(target_params, QdrantWriterTargetParams)

        collection_info = self._client.get_collection(target_params.collection_params["collection_name"])
        return QdrantWriterTargetParams(
            collection_params={
                "collection_name": target_params.collection_params["collection_name"],
                "vectors_config": collection_info.config.params.vectors,
                "sparse_vectors_config": collection_info.config.params.sparse_vectors,
                "shard_number": collection_info.config.params.shard_number,
                "sharding_method": collection_info.config.params.sharding_method,
                "replication_factor": collection_info.config.params.replication_factor,
                "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                "on_disk_payload": collection_info.config.params.on_disk_payload,
                "hnsw_config": collection_info.config.hnsw_config,
                "optimizers_config": collection_info.config.optimizer_config,
                "wal_config": collection_info.config.wal_config,
                "quantization_config": collection_info.config.quantization_config,
            },
        )


@dataclass
class QdrantWriterRecord(BaseDBWriter.Record):
    id: str
    vector: Union[list[float], dict[str, list[float]]]
    payload: dict[str, Any]

    @classmethod
    def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "QdrantWriterRecord":
        assert isinstance(target_params, QdrantWriterTargetParams)
        assert document.doc_id is not None, f"Document found with null id: {document}"
        vector = document.embedding or {}
        payload = {
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.to_dict() if document.bbox else None,
            "shingles": [str(s) for s in document.shingles] if document.shingles else None,
            "properties": document.properties,
        }

        return QdrantWriterRecord(document.doc_id, vector, payload)

    def to_http_vector(self) -> dict:
        if self.sparse_values:
            return asdict(self)
        else:
            return {"id": self.id, "values": self.values, "metadata": self.metadata}


def _narrow_list_of_qdrant_records(records: list[BaseDBWriter.Record]) -> TypeGuard[QdrantWriterRecord]:
    return all(isinstance(r, QdrantWriterRecord) for r in records)


class QdrantWriter(BaseDBWriter):
    Client = QdrantWriterClient
    Record = QdrantWriterRecord
    TargetParams = QdrantWriterTargetParams
    ClientParams = QdrantWriterClientParams
