from sycamore.data import Document
from sycamore.connectors.base_reader import BaseDBReader
from dataclasses import dataclass, field
from typing import Dict

from opensearchpy import OpenSearch


@dataclass
class OpenSearchReaderClientParams(BaseDBReader.ClientParams):
    os_client_args: dict = field(default_factory=lambda: {})


@dataclass
class OpenSearchReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    query: Dict = field(default_factory=lambda: {"query": {"match_all": {}}})
    kwargs: Dict = field(default_factory=lambda: {})


class OpenSearchReaderClient(BaseDBReader.Client):
    def __init__(self, client: OpenSearch):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "OpenSearchReaderClient":
        assert isinstance(params, OpenSearchReaderClientParams)
        client = OpenSearch(**params.os_client_args)
        return OpenSearchReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "OpenSearchReaderQueryResponse":
        assert isinstance(
            query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        assert "index" not in query_params.kwargs and "body" not in query_params.kwargs
        if "scroll" not in query_params.kwargs:
            query_params.kwargs["scroll"] = "1m"
        if "size" not in query_params.kwargs:
            query_params.kwargs["size"] = 200
        response = self._client.search(index=query_params.index_name, body=query_params.query, **query_params.kwargs)
        scroll_id = response["_scroll_id"]
        result = []
        try:
            while True:
                hits = response["hits"]["hits"]
                for hit in hits:
                    result += [hit]

                if not hits:
                    break
                response = self._client.scroll(scroll_id=scroll_id, scroll=query_params.kwargs["scroll"])
        finally:
            self._client.clear_scroll(scroll_id=scroll_id)
        return OpenSearchReaderQueryResponse(result)

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, OpenSearchReaderQueryParams)
        return self._client.indices.exists(index=query_params.index_name)


@dataclass
class OpenSearchReaderQueryResponse(BaseDBReader.QueryResponse):
    output: list

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, OpenSearchReaderQueryResponse)
        result = []
        for data in self.output:
            doc = Document(
                {
                    **data.get("_source", {}),
                }
            )
            result.append(doc)
        return result


class OpenSearchReader(BaseDBReader):
    Client = OpenSearchReaderClient
    Record = OpenSearchReaderQueryResponse
    ClientParams = OpenSearchReaderClientParams
    QueryParams = OpenSearchReaderQueryParams
