from sycamore.data import Document
from sycamore.connectors.common import unflatten_data
from sycamore.connectors.base_reader import BaseDBReader
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dataclasses import asdict
from weaviate.client import (
    AdditionalConfig,
    AuthCredentials,
    ConnectionParams,
    EmbeddedOptions,
)
from weaviate import WeaviateClient


@dataclass
class WeaviateReaderClientParams(BaseDBReader.ClientParams):
    connection_params: Optional[ConnectionParams] = None
    embedded_options: Optional[EmbeddedOptions] = None
    auth_client_secret: Optional[AuthCredentials] = None
    additional_headers: Optional[dict] = None
    additional_config: Optional[AdditionalConfig] = None
    skip_init_checks: bool = False


@dataclass
class WeaviateReaderQueryParams(BaseDBReader.QueryParams):
    collection_name: str
    query_kwargs: Optional[Dict] = None


class WeaviateReaderClient(BaseDBReader.Client):
    def __init__(self, client: WeaviateClient):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "WeaviateReaderClient":
        assert isinstance(params, WeaviateReaderClientParams)
        client = WeaviateClient(**asdict(params))
        return WeaviateReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "WeaviateReaderQueryResponse":
        assert isinstance(
            query_params, WeaviateReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        with self._client:
            query = None
            collection = []
            if query_params.query_kwargs:
                query = self._client.collections.get(query_params.collection_name).query
                for method, value in query_params.query_kwargs.items():
                    if value is not None:
                        method_name = f"{method}"
                        if hasattr(query, method_name):
                            query = getattr(query, method_name)(**value)
                        else:
                            raise ValueError(f"Error: Method '{method_name}' not found in query object.")
                query = query.objects  # type: ignore
            else:
                collection = list(
                    self._client.collections.get(query_params.collection_name).iterator(include_vector=True)
                )
            results = WeaviateReaderQueryResponse(collection=collection, query_output=query)
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, WeaviateReaderQueryParams)
        with self._client:
            return self._client.collections.exists(query_params.collection_name)


@dataclass
class WeaviateReaderQueryResponse(BaseDBReader.QueryResponse):
    collection: list
    query_output: Any

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, WeaviateReaderQueryResponse)
        result = []
        output_list = self.query_output if self.query_output else self.collection
        for object in output_list:
            doc = Document(
                object.vector | unflatten_data(dict(object.properties), "__") | {"doc_id": str(object.uuid)}
            )  # type: ignore
            result.append(doc)
        return result


class WeaviateReader(BaseDBReader):
    Client = WeaviateReaderClient
    Record = WeaviateReaderQueryResponse
    ClientParams = WeaviateReaderClientParams
    QueryParams = WeaviateReaderQueryParams
