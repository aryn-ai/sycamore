from sycamore.data import Document
from sycamore.connectors.common import unflatten_data
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass
import typing
from typing import Optional, Dict, Any


if typing.TYPE_CHECKING:
    from weaviate import WeaviateClient
    from weaviate.client import (
        AdditionalConfig,
        AuthCredentials,
        ConnectionParams,
        EmbeddedOptions,
    )


@dataclass
class WeaviateReaderClientParams(BaseDBReader.ClientParams):
    connection_params: Optional["ConnectionParams"] = None
    embedded_options: Optional["EmbeddedOptions"] = None
    auth_client_secret: Optional["AuthCredentials"] = None
    additional_headers: Optional[dict] = None
    additional_config: Optional["AdditionalConfig"] = None
    skip_init_checks: bool = False


@dataclass
class WeaviateReaderQueryParams(BaseDBReader.QueryParams):
    collection_name: str
    query_kwargs: Optional[Dict] = None


class WeaviateReaderClient(BaseDBReader.Client):
    def __init__(self, client: "WeaviateClient"):
        self._client = client

    @classmethod
    @requires_modules("weaviate", extra="weaviate")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "WeaviateReaderClient":
        from weaviate import WeaviateClient

        assert isinstance(params, WeaviateReaderClientParams)
        client = WeaviateClient(**params.__dict__)
        return WeaviateReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "WeaviateReaderQueryResponse":
        assert isinstance(
            query_params, WeaviateReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        with self._client:
            collection = None
            if query_params.query_kwargs:
                collection = self._client.collections.get(query_params.collection_name).query
                for method, value in query_params.query_kwargs.items():
                    if value is not None:
                        method_name = str(method)
                        if hasattr(collection, method_name):
                            collection = getattr(collection, method_name)(**value)
                        else:
                            raise ValueError(f"Error: Method '{method_name}' not found in query object.")
                collection = collection.objects if hasattr(collection, "objects") else list[collection]  # type: ignore
            else:
                collection = list(
                    self._client.collections.get(query_params.collection_name).iterator(include_vector=True)
                )
            results = WeaviateReaderQueryResponse(collection=collection)
            return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, WeaviateReaderQueryParams)
        with self._client:
            return self._client.collections.exists(query_params.collection_name)


@dataclass
class WeaviateReaderQueryResponse(BaseDBReader.QueryResponse):
    collection: Any

    def __init__(self, collection):
        self.collection = collection

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, WeaviateReaderQueryResponse)
        result = []
        for object in self.collection:
            doc = Document(
                (object.vector if hasattr(object, "vector") else {})
                | unflatten_data(dict(object.properties), "__")
                | {"doc_id": str(object.uuid)}
            )  # type: ignore
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            result.append(doc)
        return result


class WeaviateReader(BaseDBReader):
    Client = WeaviateReaderClient
    Record = WeaviateReaderQueryResponse
    ClientParams = WeaviateReaderClientParams
    QueryParams = WeaviateReaderQueryParams
