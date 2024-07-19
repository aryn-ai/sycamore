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
        # no_specification = ["query", "pit", "search_after", "index_name"]
        # assert all(no_specification) not in query_params.kwargs
        # if not query_params.kwargs.get("track_total_hits"):
        #     query_params.kwargs["track_total_hits"] = False
        # if not query_params.kwargs.get("sort"):
        #     query_params.kwargs["sort"] = [
        #         {"_shard_doc": "desc"},
        #     ]
        scroll = "1m"
        response = self._client.search(index=query_params.index_name, scroll=scroll, size=200, body=query_params.query)
        scroll_id = response["_scroll_id"]
        result = []
        try:
            while True:
                hits = response["hits"]["hits"]
                for hit in hits:
                    result += [hit]

                if not hits:
                    break
                response = self._client.scroll(scroll_id=scroll_id, scroll=scroll)
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
            doc_id = data["_id"]
            doc = Document(
                {"doc_id": doc_id, "embedding": data["_source"].get("embeddings"), **data["_source"].get("properties")}
            )
            result.append(doc)
        return result


class OpenSearchReader(BaseDBReader):
    Client = OpenSearchReaderClient
    Record = OpenSearchReaderQueryResponse
    ClientParams = OpenSearchReaderClientParams
    QueryParams = OpenSearchReaderQueryParams
