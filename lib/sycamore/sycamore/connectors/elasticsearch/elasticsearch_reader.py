from sycamore.data import Document
from pinecone.grpc import PineconeGRPC
from sycamore.connectors.common import unflatten_data
from sycamore.connectors.base_reader import BaseDBReader
from dataclasses import dataclass, field
from typing import Optional, Dict

from elasticsearch import Elasticsearch, ApiError


@dataclass
class ElasticsearchReaderClientParams(BaseDBReader.ClientParams):
    url: str
    es_client_args: dict = field(default_factory=lambda: {})


@dataclass
class ElasticsearchReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    size: int
    query: Dict = field(default_factory=lambda: {"match_all": {}})


class ElasticsearchReaderClient(BaseDBReader.Client):
    def __init__(self, client: Elasticsearch):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "ElasticsearchReaderClient":
        assert isinstance(params, ElasticsearchReaderClientParams)
        client = Elasticsearch(params.url, **params.es_client_args)
        return ElasticsearchReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "ElasticsearchReaderQueryResponse":
        assert isinstance(
            query_params, ElasticsearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        print(self._client.search(index=query_params.index_name, query=query_params.query, size=query_params.size))
        results = ElasticsearchReaderQueryResponse(
            list(self._client.search(index=query_params.index_name, query=query_params.query, size=query_params.size))
        )
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, ElasticsearchReaderQueryParams)
        return self._client.indices.exists(index=query_params.index_name)


@dataclass
class ElasticsearchReaderQueryResponse(BaseDBReader.QueryResponse):
    output: list

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, ElasticsearchReaderQueryResponse)
        result = []
        for data in self.output:
            doc_id = data.id.split("#")[1] if len(data.id.split("#")) > 1 else id
            if data.sparse_vector:
                term_frequency = dict(zip(data.sparse_vector.indices, data.sparse_vector.values))
                data.metadata["properties.term_frequency"] = term_frequency
            metadata = data.metadata if data.metadata else {}
            doc = Document({"doc_id": doc_id, "embedding": data.values} | unflatten_data(metadata))  # type: ignore
            result.append(doc)
        return result


class ElasticsearchReader(BaseDBReader):
    Client = ElasticsearchReaderClient
    Record = ElasticsearchReaderQueryResponse
    ClientParams = ElasticsearchReaderClientParams
    QueryParams = ElasticsearchReaderQueryParams
