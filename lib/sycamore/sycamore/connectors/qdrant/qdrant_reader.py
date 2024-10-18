from sycamore.data import Document
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
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
            if isinstance(point.vector, dict):
                # https://api.qdrant.tech/api-reference/search/query-points#request.body.using
                vector_name = query_params.query_params.get("using")
                if not point.vector:
                    vector = None
                elif vector_name:
                    vector = point.vector.get(vector_name)
                else:
                    # Get the first vector if no vector name is provided
                    vector = list(point.vector.values())[0]
            else:
                vector = point.vector
            doc_dict = (
                {"doc_id": point.id, "embedding": vector} | unflatten_data(point.payload, "__") if point.payload else {}
            )
            doc_dict["bbox"] = bbox.values() if (bbox := doc_dict.get("bbox")) else []
            doc = Document(doc_dict)
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            result.append(doc)
        return result


class QdrantReader(BaseDBReader):
    Client = QdrantReaderClient
    Record = QdrantReaderQueryResponse
    ClientParams = QdrantReaderClientParams
    QueryParams = QdrantReaderQueryParams
