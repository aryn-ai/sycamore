from sycamore.data import Document

from sycamore.connectors.common import unflatten_data
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass, asdict
from typing import Optional, Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass
class QdrantReaderClientParams(BaseDBReader.ClientParams):
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
class QdrantReaderQueryParams(BaseDBReader.QueryParams):
    query_params: Dict
    "Arguments to pass to the `QdrantClient.query_points` method."


class QdrantReaderClient(BaseDBReader.Client):
    def __init__(self, client: "QdrantClient"):
        self._client = client

    @classmethod
    @requires_modules("qdrant_client", extra="qdrant")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "QdrantReaderClient":
        assert isinstance(params, QdrantReaderClientParams)

        from qdrant_client import QdrantClient

        client = QdrantClient(**asdict(params))
        return cls(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "QdrantReaderQueryResponse":
        assert isinstance(
            query_params, QdrantReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        results = QdrantReaderQueryResponse(points=self._client.query_points(**query_params.query_params).points)

        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, QdrantReaderQueryParams)
        assert "collection_name" in query_params.query_params, "collection_name is required to check target presence"
        return self._client.collection_exists(query_params.query_params["collection_name"])


@dataclass
class QdrantReaderQueryResponse(BaseDBReader.QueryResponse):
    points: list

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        from qdrant_client.models import ScoredPoint

        assert isinstance(self, QdrantReaderQueryResponse)
        assert isinstance(query_params, QdrantReaderQueryParams)
        result = []
        for point in self.points:
            assert isinstance(point, ScoredPoint)
            doc = Document({"doc_id": point.id, "embedding": point.vector} | unflatten_data(point.payload or {}))
            result.append(doc)
        return result


class QdrantReader(BaseDBReader):
    Client = QdrantReaderClient
    Record = QdrantReaderQueryResponse
    ClientParams = QdrantReaderClientParams
    QueryParams = QdrantReaderQueryParams
