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
    vector_name: Optional[str]

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        assert isinstance(other, QdrantWriterTargetParams)

        return all(
            [
                self.collection_params["collection_name"] == other.collection_params["collection_name"],
                self.collection_params["vectors_config"] == other.collection_params["vectors_config"],
                self.vector_name == other.vector_name,
            ]
        )


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

        points = [
            PointStruct(id=record.id, vector=record.vector, payload=record.payload)  # type: ignore
            for record in records
        ]

        self._client.upload_points(
            collection_name=target_params.collection_params["collection_name"], points=points, wait=True
        )

    def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, QdrantWriterTargetParams)
        try:
            self._client.create_collection(**target_params.collection_params)
        except Exception:  # Can swallow since we validate with get_existing_target_params + compatible_with
            return

    def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> QdrantWriterTargetParams:
        assert isinstance(target_params, QdrantWriterTargetParams)

        collection_info = self._client.get_collection(target_params.collection_params["collection_name"])
        vectors_config = collection_info.config.params.vectors
        assert vectors_config is not None
        return QdrantWriterTargetParams(
            collection_params={
                "collection_name": target_params.collection_params["collection_name"],
                "vectors_config": (
                    vectors_config
                    if isinstance(vectors_config, dict)
                    else vectors_config.model_dump(exclude_defaults=True)
                ),
            },
            vector_name=target_params.vector_name,
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
        if document.embedding:
            if target_params.vector_name:
                vector = {target_params.vector_name: document.embedding}
            else:
                vector = document.embedding
        else:
            vector = {}

        payload = {
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.to_dict() if document.bbox else None,
            "shingles": document.shingles or None,
            "properties": document.properties,
        }

        return QdrantWriterRecord(document.doc_id, vector, payload)


def _narrow_list_of_qdrant_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[QdrantWriterRecord]]:
    return all(isinstance(r, QdrantWriterRecord) for r in records)


class QdrantWriter(BaseDBWriter):
    Client = QdrantWriterClient
    Record = QdrantWriterRecord
    TargetParams = QdrantWriterTargetParams
    ClientParams = QdrantWriterClientParams
