import logging

from sycamore.data import Document, Element
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass, field
import typing
from typing import Dict

if typing.TYPE_CHECKING:
    from opensearchpy import OpenSearch


@dataclass
class OpenSearchReaderClientParams(BaseDBReader.ClientParams):
    os_client_args: dict = field(default_factory=lambda: {})


@dataclass
class OpenSearchReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    query: Dict = field(default_factory=lambda: {"query": {"match_all": {}}})
    kwargs: Dict = field(default_factory=lambda: {})
    reconstruct_document: bool = False


class OpenSearchReaderClient(BaseDBReader.Client):
    def __init__(self, client: "OpenSearch"):
        self._client = client

    @classmethod
    @requires_modules("opensearchpy", extra="opensearch")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "OpenSearchReaderClient":
        from opensearchpy import OpenSearch

        assert isinstance(params, OpenSearchReaderClientParams)
        client = OpenSearch(**params.os_client_args)
        return OpenSearchReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "OpenSearchReaderQueryResponse":
        assert isinstance(
            query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        assert "index" not in query_params.kwargs and "body" not in query_params.kwargs
        if "scroll" not in query_params.kwargs:
            query_params.kwargs["scroll"] = "10m"
        if "size" not in query_params.query and "size" not in query_params.kwargs:
            query_params.kwargs["size"] = 200
        logging.debug(f"OpenSearch query on {query_params.index_name}: {query_params.query}")
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
        return OpenSearchReaderQueryResponse(result, self._client)

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, OpenSearchReaderQueryParams)
        return self._client.indices.exists(index=query_params.index_name)


@dataclass
class OpenSearchReaderQueryResponse(BaseDBReader.QueryResponse):
    output: list

    """
    The client used to implement document reconstruction. Can also be used for lazy loading.
    """
    client: typing.Optional["OpenSearch"] = None

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(query_params, OpenSearchReaderQueryParams)
        result: list[Document] = []
        if not query_params.reconstruct_document:
            for data in self.output:
                doc = Document(
                    {
                        **data.get("_source", {}),
                    }
                )
                doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
                doc.properties["score"] = data["_score"]
                result.append(doc)
        else:
            assert (
                self.client is not None
            ), "Document reconstruction requires an OpenSearch client in OpenSearchReaderQueryResponse"
            """
            Document reconstruction:
            1. Construct a map of all unique parent Documents (i.e. no parent_id field)
                1.1 If we find doc_ids without parent documents, we create empty parent Documents
            2. Perform a terms query to retrieve all (including non-matched) other records for that parent_id
            3. Add elements to unique parent Documents
            """
            # Get unique documents
            unique_docs: dict[str, Document] = {}
            query_result_elements_per_doc: dict[str, set[str]] = {}
            for data in self.output:
                doc = Document(
                    {
                        **data.get("_source", {}),
                    }
                )
                doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
                assert doc.doc_id, "Retrieved invalid doc with missing doc_id"
                if not doc.parent_id:
                    # Always use retrieved doc as the unique parent doc - override any empty parent doc created below
                    unique_docs[doc.doc_id] = doc
                else:
                    # Create empty parent documents if no parent document was in result set
                    unique_docs[doc.parent_id] = unique_docs.get(
                        doc.parent_id,
                        Document(
                            {
                                "doc_id": doc.parent_id,
                                "properties": {
                                    **doc.properties,
                                    DocumentPropertyTypes.SOURCE: DocumentSource.DOCUMENT_RECONSTRUCTION_PARENT,
                                },
                            }
                        ),
                    )
                    elements = query_result_elements_per_doc.get(doc.parent_id, set())
                    elements.add(doc.doc_id)
                    query_result_elements_per_doc[doc.parent_id] = elements

            # Batched retrieval of all elements belong to unique docs
            doc_ids = list(unique_docs.keys())
            all_elements_for_docs = self._get_all_elements_for_doc_ids(doc_ids, query_params.index_name)

            """
            Add elements to unique docs. If they were not part of the original result, 
            we set properties.DocumentPropertyTypes.SOURCE = DOCUMENT_RECONSTRUCTION_RETRIEVAL
            """
            for element in all_elements_for_docs:
                doc = Document(
                    {
                        **element.get("_source", {}),
                    }
                )
                assert doc.parent_id, "Got non-element record from OpenSearch reconstruction query"
                if doc.doc_id not in query_result_elements_per_doc.get(doc.parent_id, {}):
                    doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DOCUMENT_RECONSTRUCTION_RETRIEVAL
                else:
                    doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
                parent = unique_docs[doc.parent_id]
                parent.elements.append(Element(doc.data))

            result = list(unique_docs.values())

            # sort elements per doc
            for doc in result:
                doc.elements.sort(key=lambda e: e.element_index if e.element_index is not None else float("inf"))

        return result

    def _get_all_elements_for_doc_ids(self, doc_ids: list[str], index: str) -> list[typing.Any]:
        assert self.client, "_get_all_elements_for_doc_ids requires an OpenSearch client instance in this class"
        """
        Returns all records in OpenSearch belonging to a list of Document ids (element.parent_id)
        """
        batch_size = 100
        page_size = 500

        all_elements = []
        for i in range(0, len(doc_ids), batch_size):
            doc_ids_batch = doc_ids[i : i + batch_size]
            from_offset = 0
            while True:
                query = {
                    "query": {"terms": {"parent_id.keyword": doc_ids_batch}},
                    "size": page_size,
                    "from": from_offset,
                }
                response = self.client.search(index=index, body=query)
                hits = response["hits"]["hits"]
                all_elements.extend(hits)
                if len(hits) < page_size:
                    break
                from_offset += page_size
        return all_elements


class OpenSearchReader(BaseDBReader):
    Client = OpenSearchReaderClient
    Record = OpenSearchReaderQueryResponse
    ClientParams = OpenSearchReaderClientParams
    QueryParams = OpenSearchReaderQueryParams
