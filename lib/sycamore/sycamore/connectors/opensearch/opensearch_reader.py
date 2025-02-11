import json
import logging
from copy import deepcopy

import pandas as pd

from sycamore.connectors.doc_reconstruct import DocumentReconstructor
from sycamore.data import Document, Element
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules
from dataclasses import dataclass, field
from typing import Optional, Any, List, TYPE_CHECKING

from sycamore.utils.time_trace import TimeTrace, timetrace

if TYPE_CHECKING:
    from opensearchpy import OpenSearch
    from ray.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class OpenSearchReaderClientParams(BaseDBReader.ClientParams):
    os_client_args: dict = field(default_factory=lambda: {})


@dataclass
class OpenSearchReaderQueryParams(BaseDBReader.QueryParams):
    index_name: str
    query: dict
    kwargs: dict = field(default_factory=lambda: {})
    reconstruct_document: bool = False
    doc_reconstructor: Optional[DocumentReconstructor] = None


class OpenSearchReaderClient(BaseDBReader.Client):
    def __init__(self, client: "OpenSearch"):
        self._client = client

    @classmethod
    @requires_modules("opensearchpy", extra="opensearch")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "OpenSearchReaderClient":
        from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging

        assert isinstance(params, OpenSearchReaderClientParams)
        client = OpenSearchClientWithLogging(**params.os_client_args)
        return OpenSearchReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams) -> "OpenSearchReaderQueryResponse":
        assert isinstance(
            query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"

        assert "index" not in query_params.kwargs and "body" not in query_params.kwargs
        logger.debug(f"OpenSearch query on {query_params.index_name}: {query_params.query}")
        if "size" not in query_params.query and "size" not in query_params.kwargs:
            query_params.kwargs["size"] = 200
        result = []
        if query_params.reconstruct_document and "_source_includes" not in query_params.kwargs:
            query_params.kwargs["_source_includes"] = ["doc_id", "parent_id", "properties"]
        if query_params.doc_reconstructor is not None:
            query_params.kwargs["_source_includes"] = query_params.doc_reconstructor.get_required_source_fields()
        # No pagination needed for knn queries
        if "query" in query_params.query and "knn" in query_params.query["query"]:
            response = self._client.search(
                index=query_params.index_name, body=query_params.query, **query_params.kwargs
            )
            hits = response["hits"]["hits"]
            if hits:
                for hit in hits:
                    result += [hit]
        else:
            if "scroll" not in query_params.kwargs:
                query_params.kwargs["scroll"] = "10m"
            response = self._client.search(
                index=query_params.index_name, body=query_params.query, **query_params.kwargs
            )
            scroll_id = response["_scroll_id"]
            try:
                while True:
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        result += [hit]

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
    client: Optional["OpenSearch"] = None

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(query_params, OpenSearchReaderQueryParams)
        result: list[Document] = []
        if query_params.doc_reconstructor is not None:
            logger.info("Using DocID to Document reconstructor")
            unique = set()
            for data in self.output:
                doc_id = query_params.doc_reconstructor.get_doc_id(data)
                if doc_id not in unique:
                    result.append(query_params.doc_reconstructor.reconstruct(data))
                    unique.add(doc_id)
        elif not query_params.reconstruct_document:
            for data in self.output:
                doc = Document(
                    {
                        **data.get("_source", {}),
                    }
                )
                doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
                doc.properties["score"] = (
                    data["_score"] if doc.properties.get("score") is None else doc.properties["score"]
                )
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

            def get_batches(doc_ids) -> list[list[str]]:
                batches = []
                batch_doc_count = 0
                cur_batch: list[str] = []
                for i in range(len(doc_ids)):
                    query = {
                        "query": {"terms": {"parent_id.keyword": [doc_ids[i]]}},
                    }
                    doc_count = get_doc_count(self.client, query_params.index_name, query)
                    if batch_doc_count + doc_count > 10000:
                        batches.append(cur_batch)
                        cur_batch = [doc_ids[i]]
                        batch_doc_count = 0
                    else:
                        batch_doc_count += doc_count
                        cur_batch.append(doc_ids[i])

                if len(cur_batch) > 0:
                    batches.append(cur_batch)
                return batches

            batches = get_batches(doc_ids)

            all_elements_for_docs = []
            for batch in batches:
                all_elements_for_docs += self._get_all_elements_for_doc_ids(batch, query_params.index_name)

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

    def _get_all_elements_for_doc_ids(self, doc_ids: list[str], index: str) -> list[Any]:
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


def get_doc_count(os_client, index_name: str, query: Optional[dict[str, Any]] = None) -> int:
    res = os_client.search(index=index_name, body=query, size=0, track_total_hits=True)
    return res["hits"]["total"]["value"]


def get_doc_count_for_slice(os_client, slice_query: dict[str, Any]) -> int:
    res = os_client.search(body=slice_query, size=0, track_total_hits=True)
    return res["hits"]["total"]["value"]


class OpenSearchReader(BaseDBReader):
    Client = OpenSearchReaderClient
    Record = OpenSearchReaderQueryResponse
    ClientParams = OpenSearchReaderClientParams
    QueryParams = OpenSearchReaderQueryParams

    def __init__(
        self,
        client_params: OpenSearchReaderClientParams,
        query_params: BaseDBReader.QueryParams,
        use_pit: bool = True,
        **kwargs,
    ):
        assert isinstance(
            query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"

        super().__init__(client_params, query_params, **kwargs)
        self._client_params = client_params
        self._query_params = query_params
        # TODO add support for 'search_after' pagination if a sort field is provided.
        self.use_pit = use_pit
        logger.info(f"OpenSearchReader using PIT: {self.use_pit}")

    @timetrace("OpenSearchReader")
    def _to_parent_doc(self, doc: dict[str, Any]) -> List[dict[str, Any]]:
        """
        Get all parent documents from a given slice.
        """
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        client = None
        try:
            client = self.Client.from_client_params(self._client_params)

            if not client.check_target_presence(self._query_params):
                raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")

            os_client = client._client
            slice_query = json.loads(doc["doc"])

            assert (
                get_doc_count_for_slice(os_client, slice_query) < 10000
            ), "Slice query should return <= 10,000 documents"

            results = []
            size = 1000
            page = 0
            logger.info(f"Executing {slice_query} against {self._query_params.index_name}")

            query_params = {"_source_includes": "parent_id"}
            parent_ids = set()
            while True:
                res = os_client.search(
                    body=slice_query,
                    size=size,
                    from_=page * size,
                    **query_params,
                )
                hits = res["hits"]["hits"]
                if hits is None or len(hits) == 0:
                    break

                for hit in hits:
                    if (
                        "parent_id" in hit["_source"]
                        and hit["_source"]["parent_id"] is not None
                        and hit["_source"]["parent_id"] not in parent_ids
                    ):
                        # Only add a child doc whose parent_id has not been found, yet.
                        parent_ids.add(hit["_source"]["parent_id"])
                        results.append(hit)
                    elif ("parent_id" not in hit["_source"] or hit["_source"]["parent_id"] is None) and hit[
                        "_id"
                    ] not in parent_ids:
                        # Add a parent doc if it's a match.
                        parent_id = hit["_id"]
                        parent_ids.add(parent_id)
                        hit["_source"]["parent_id"] = parent_id
                        results.append(hit)

                page += 1

            logger.info(f"Read {len(results)} documents from {self._query_params.index_name}")

        except Exception as e:
            raise ValueError(f"Error reading from target: {e}, query: {slice_query}")
        finally:
            if client is not None:
                client.close()

        ret = [doc["_source"] for doc in results]
        return ret

    @timetrace("OpenSearchReader")
    def _to_doc(self, doc: dict[str, Any]) -> List[dict[str, Any]]:
        """
        Get all documents from a given slice.
        """
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        assert self._query_params.reconstruct_document is False, "Reconstruct document is not supported in this method"

        client = None
        try:
            client = self.Client.from_client_params(self._client_params)

            if not client.check_target_presence(self._query_params):
                raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")

            os_client = client._client
            slice_query = json.loads(doc["doc"])
            slice_count = get_doc_count_for_slice(os_client, slice_query)
            assert slice_count <= 10000, f"Slice count ({slice_count}) should return <= 10,000 documents"

            results = []
            size = 1000
            page = 0

            while True:
                res = os_client.search(
                    body=slice_query,
                    size=size,
                    from_=page * size,
                )
                hits = res["hits"]["hits"]
                if hits is None or len(hits) == 0:
                    break

                for hit in hits:
                    hit["_source"]["slice"] = slice_query
                    results += [hit]
                page += 1

        except Exception as e:
            raise ValueError(f"Error reading from target: {e}")
        finally:
            if client is not None:
                client.close()

        records = OpenSearchReaderQueryResponse(results, os_client)
        docs = records.to_docs(query_params=self._query_params)
        return [{"doc": doc.serialize()} for doc in docs]

    def map_reduce_parent_id(self, group: pd.DataFrame) -> pd.DataFrame:
        parent_ids = set()
        for row in group["parent_id"]:
            if row not in parent_ids:
                parent_ids.add(row)

        return pd.DataFrame([{"_source": {"doc_id": parent_id}} for parent_id in parent_ids])

    def reconstruct(self, doc: dict[str, Any]) -> dict[str, Any]:
        client = self.Client.from_client_params(self._client_params)

        if not client.check_target_presence(self._query_params):
            raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")

        os_client = client._client
        doc_id = doc["_source"]["doc_id"]
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        parent_doc = os_client.get(index=self._query_params.index_name, id=doc_id)
        records = OpenSearchReaderQueryResponse([parent_doc], os_client)
        docs = records.to_docs(query_params=self._query_params)

        return {"doc": docs[0].serialize()}

    def execute(self, **kwargs) -> "Dataset":
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        if "query" in self._query_params.query and "knn" in self._query_params.query["query"]:
            return super().execute(**kwargs)

        if self.use_pit:
            return self._execute_pit(**kwargs)
        else:
            return self._execute(**kwargs)

    def map_reduce_docs_by_slice(self, group: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        doc_ids = set()
        for row in group["doc_id"]:
            doc_ids.add(row)

        logger.info(f"No. of IDs: {len(doc_ids)}")
        client = self.Client.from_client_params(self._client_params)

        if not client.check_target_presence(self._query_params):
            raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")

        os_client = client._client
        fetched_docs = []
        for doc_id in doc_ids:
            res = os_client.get(index=self._query_params.index_name, id=doc_id)
            fetched_docs.append(res["_source"])
        return pd.DataFrame(fetched_docs)

    def _execute(self, **kwargs) -> "Dataset":
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"
        # docs = None
        import hashlib

        parallelism = self.resource_args.get("parallelism", 2)
        try:
            client = self.Client.from_client_params(self._client_params)

            if not client.check_target_presence(self._query_params):
                raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")
            query_params = deepcopy(self._query_params)
            query_params.kwargs["_source_includes"] = ["doc_id", "parent_id"]
            records = client.read_records(query_params=query_params)
            docs = records.output  #  records.to_docs(query_params=self._query_params)
            for doc in docs:
                if "parent_id" not in doc["_source"] or doc["_source"]["parent_id"] is None:
                    doc["_source"]["parent_id"] = doc["_id"]
                if not self._query_params.reconstruct_document:
                    ho = hashlib.sha256(doc["_id"].encode())
                    doc["_source"]["slice"] = str(int(ho.hexdigest(), 16) % parallelism)
        except Exception as e:
            raise ValueError(f"Error reading from target: {e}")
        finally:
            client.close()

        from ray.data import from_items

        # logger.info(f"Sample docs: {docs[:5]}")
        ds = from_items(items=[doc["_source"] for doc in docs])
        if self._query_params.reconstruct_document:
            return ds.groupby("parent_id").map_groups(self.map_reduce_parent_id).map(self.reconstruct)
        else:
            return (
                ds.groupby("slice")
                .map_groups(self.map_reduce_docs_by_slice)
                .map(lambda d: {"doc": Document(**d).serialize()})
            )

    def _execute_pit(self, **kwargs) -> "Dataset":
        """Distribute the work evenly across available workers.
        We don't want a slice with more than 10k documents as we need to use 'from' to paginate through the results."""
        assert isinstance(
            self._query_params, OpenSearchReaderQueryParams
        ), f"Wrong kind of query parameters found: {self._query_params}"

        from sycamore.utils.ray_utils import check_serializable

        check_serializable(self._client_params, self._query_params)

        client = None
        try:
            client = self.Client.from_client_params(self._client_params)

            if not client.check_target_presence(self._query_params):
                raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")

            index_name = self._query_params.index_name
            query = self._query_params.query

            os_client = client._client
            doc_count = get_doc_count(os_client, index_name, query)
            # We want the document count to be well below 10k in each slice.
            slice_size = 2500
            num_slices = max(1 + doc_count // slice_size, 2)
            # num_workers = self.resource_args.get("parallelism", 2)  # 2 is the minimum number of slices.
            # num_slices = num_workers

            res = os_client.create_pit(index=index_name, keep_alive="100m")
            pit_id = res["pit_id"]
            docs = []
            for i in range(num_slices):
                _query = {
                    "slice": {
                        "id": i,
                        "max": num_slices,
                    },
                    "pit": {
                        "id": pit_id,
                        "keep_alive": "1m",
                    },
                }
                if "query" in query:
                    _query["query"] = query["query"]

                docs.append({"doc": json.dumps(_query)})
                logger.debug(f"Added slice {i} to the query {_query}")
        except Exception as e:
            raise ValueError(f"Error reading from target: {e}")
        finally:
            if client is not None:
                client.close()

        with TimeTrace("OpenSearchReader"):
            from ray.data import from_items

            ds = from_items(items=docs)

            if self._query_params.reconstruct_document:
                # Step 1: Construct slices (pages)
                # Step 2: For each page, get all parent documents
                # Step 3: Group by parent_id
                # Step 4: Deduplicate parent documents by ID
                # Step 5: Reconstruct parent documents using 'parent_id'

                return (
                    ds.flat_map(self._to_parent_doc, **self.resource_args)
                    .groupby("parent_id")
                    .map_groups(self.map_reduce_parent_id)
                    # TODO use map_batches to improve 'get_all_elements_for_doc_ids' performance
                    #  by making fewer requests to OpenSearch
                    .map(self.reconstruct)
                )
            else:
                # Step 1: Construct slices (pages)
                # Step 2: For each page, get all documents
                return ds.flat_map(self._to_doc, **self.resource_args)
