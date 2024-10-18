from sycamore.data import Document

from sycamore.connectors.common import unflatten_data
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PineconeReaderClientParams(BaseDBReader.ClientParams):
    api_key: str


@dataclass
class PineconeReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    namespace: str
    query: Optional[Dict]


class PineconeReaderClient(BaseDBReader.Client):
    @requires_modules("pinecone", extra="pinecone")
    def __init__(self, client_params: PineconeReaderClientParams):
        from pinecone.grpc import PineconeGRPC

        self._client = PineconeGRPC(api_key=client_params.api_key, source_tag="Aryn")

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "PineconeReaderClient":
        assert isinstance(params, PineconeReaderClientParams)
        return PineconeReaderClient(params)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "PineconeReaderQueryResponse":
        assert isinstance(
            query_params, PineconeReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        index = self._client.Index(query_params.index_name)
        if query_params.query:
            results = PineconeReaderQueryResponse(list(index.query(**query_params.query)["matches"]))
        else:
            ids = []
            for pids in index.list(namespace=query_params.namespace):
                ids.extend(pids)
            results = PineconeReaderQueryResponse(
                list(dict(index.fetch(ids=ids, namespace=query_params.namespace)["vectors"]).values())
            )
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, PineconeReaderQueryParams)
        try:
            index = self._client.Index(query_params.index_name)
            return query_params.namespace in dict(index.describe_index_stats()["namespaces"])
        except Exception:
            return False


@dataclass
class PineconeReaderQueryResponse(BaseDBReader.QueryResponse):
    output: list

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, PineconeReaderQueryResponse)
        result = []
        for data in self.output:
            if len(id := data.id.split("#")) > 1:
                parent_id = id[0]
                doc_id = id[1]
            else:
                parent_id = None
                doc_id = data.id
            if data.sparse_vector:
                term_frequency = dict(zip(data.sparse_vector.indices, data.sparse_vector.values))
                data.metadata["properties.term_frequency"] = term_frequency
            metadata = data.metadata if data.metadata else {}
            doc = Document(
                {"doc_id": doc_id, "embedding": data.values, "parent_id": parent_id} | unflatten_data(metadata)
            )
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            doc.bbox = doc.bbox.values() if doc.bbox else []
            result.append(doc)
        return result


class PineconeReader(BaseDBReader):
    Client = PineconeReaderClient
    Record = PineconeReaderQueryResponse
    ClientParams = PineconeReaderClientParams
    QueryParams = PineconeReaderQueryParams
