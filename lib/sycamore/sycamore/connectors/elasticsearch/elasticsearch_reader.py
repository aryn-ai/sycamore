from sycamore.data import Document
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass, field
import typing
from typing import Dict

if typing.TYPE_CHECKING:
    from elasticsearch import Elasticsearch


@dataclass
class ElasticsearchReaderClientParams(BaseDBReader.ClientParams):
    url: str
    es_client_args: dict = field(default_factory=lambda: {})


@dataclass
class ElasticsearchReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    query: Dict = field(default_factory=lambda: {"match_all": {}})
    keep_alive: str = "1m"
    kwargs: Dict = field(default_factory=lambda: {})


class ElasticsearchReaderClient(BaseDBReader.Client):
    def __init__(self, client: "Elasticsearch"):
        self._client = client

    @classmethod
    @requires_modules("elasticsearch", extra="elasticsearch")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "ElasticsearchReaderClient":
        from elasticsearch import Elasticsearch

        assert isinstance(params, ElasticsearchReaderClientParams)
        client = Elasticsearch(params.url, **params.es_client_args)
        return ElasticsearchReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "ElasticsearchReaderQueryResponse":
        assert isinstance(
            query_params, ElasticsearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        no_specification = ["query", "pit", "search_after", "index_name"]
        assert (
            all(no_specification) not in query_params.kwargs
        ), "Please do not specify the following parameters: " + ", ".join(no_specification)
        if not query_params.kwargs.get("track_total_hits"):
            query_params.kwargs["track_total_hits"] = False
        if not query_params.kwargs.get("sort"):
            query_params.kwargs["sort"] = [
                {"_shard_doc": "desc"},
            ]
        pit = self._client.open_point_in_time(index=query_params.index_name, keep_alive=query_params.keep_alive)["id"]
        pit_dict = {"id": pit, "keep_alive": query_params.keep_alive}
        overall_list = []
        return_object = self._client.search(pit=pit_dict, query=query_params.query, **query_params.kwargs)
        results_list = list(return_object["hits"]["hits"])
        overall_list.extend(results_list)
        while results_list:
            query_params.kwargs["search_after"] = results_list[-1]["sort"]
            pit = return_object["pit_id"]
            pit_dict["id"] = pit
            return_object = self._client.search(pit=pit_dict, query=query_params.query, **query_params.kwargs)
            results_list = list(return_object["hits"]["hits"])
            overall_list.extend(results_list)
        self._client.close_point_in_time(id=pit)
        return ElasticsearchReaderQueryResponse(overall_list)

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
            doc_id = data["_id"]
            doc = Document(
                {
                    "doc_id": doc_id,
                    "parent_id": data["_source"].get("parent_id"),
                    "embedding": data["_source"].get("embeddings"),
                    **data["_source"].get("properties"),
                }
            )
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            result.append(doc)
        return result


class ElasticsearchReader(BaseDBReader):
    Client = ElasticsearchReaderClient
    Record = ElasticsearchReaderQueryResponse
    ClientParams = ElasticsearchReaderClientParams
    QueryParams = ElasticsearchReaderQueryParams
