import logging
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from opensearchpy import OpenSearch
from ray.data import Datasource, ReadTask
from ray.data.block import BlockMetadata, Block
import pyarrow as pa

from sycamore.connectors.base_reader import BaseDBReader
from sycamore.connectors.doc_reconstruct import DocumentReconstructor
from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes, DocumentSource

logger = logging.getLogger(__name__)
SEARCH_AFTER_SORT_FIELD = "_seq_no"

def get_doc_count(os_client, index_name: str, query: Optional[dict[str, Any]] = None) -> int:
    res = os_client.search(index=index_name, body=query, size=0, track_total_hits=True)
    return res["hits"]["total"]["value"]

def get_doc_count_for_slice(os_client, slice_query: dict[str, Any]) -> int:
    res = os_client.search(body=slice_query, size=0, track_total_hits=True)
    return res["hits"]["total"]["value"]


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


@dataclass
class OpenSearchReaderQueryResponse(BaseDBReader.QueryResponse):
    output: list

    """
    The client used to implement document reconstruction. Can also be used for lazy loading.
    """
    client: Optional["OpenSearch"] = None

    def to_docs(self, query_params: "BaseDBReader.QueryParams", **os_client_args) -> list[Document]:
        assert isinstance(query_params, OpenSearchReaderQueryParams)

        if self.client is None:
            from opensearchpy import OpenSearch
            self.client = OpenSearch(**os_client_args)
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


def build_slice_queries(query: dict[str, Any], pit_id: str, num_slices: int) -> list[dict[str, Any]]:
    bodies = []
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
            "sort": [{SEARCH_AFTER_SORT_FIELD: "asc"}],
        }
        if "query" in query:
            _query["query"] = query["query"]
        bodies.append(_query)
    return bodies


def search_slice(body, os_client, source_includes: Optional[list[str]] = None) -> list[dict]:
    hits = []
    page_size = 1000
    while True:
        if source_includes is not None:
            res = os_client.search(body=body, size=page_size, _source_includes=source_includes)
        else:
            res = os_client.search(body=body, size=page_size)
        _hits = res["hits"]["hits"]
        if _hits is None or len(_hits) == 0:
            break
        hits.extend(_hits)
        body["search_after"] = _hits[-1]["sort"]

    logger.debug(f"Slice hits: {len(hits)}")
    return hits


class OpenSearchDatasource(Datasource):
    """
    TODO:
    - Add support for reading a subset of fields ("source_includes")
    - Add support for reading using a Document schema
    """

    def __init__(self, client_args: dict[str, Any], index: str, query: dict[str, Any], doc_reconstruct: bool = True):
        self.client_args = client_args
        self.index = index
        self.query = query
        self.doc_reconstruct = doc_reconstruct
        client = OpenSearch(**client_args)

        from ray.data import DataContext

        context = DataContext.get_current()
        context.verbose_stats_logs = True

        shards = []
        response = client.cat.shards(index=index, format="json")
        logger.debug(response)
        total_doc_count = 0
        for item in response:
            if item["prirep"] == "p":
                logger.debug(item)
                # {
                #   'index': 'test_opensearch_read_large',
                #   'shard': '0',
                #   'prirep': 'p',
                #   'state': 'STARTED',
                #   'docs': '4096',
                #   'store': '2.5mb',
                #   'ip': '127.0.0.1',
                #   'node': 'austin-linux'
                # }
                doc_count = int(item["docs"])
                shards.append(doc_count)
                total_doc_count += doc_count

        # TODO improve this heuristic
        self.slice_count = 100 # 20 * max(2, len(shards))
        res = client.create_pit(index=index, keep_alive="10m")
        self.pit_id = res["pit_id"]
        if doc_reconstruct:
            queries = build_slice_queries(query, self.pit_id, self.slice_count)
            executor = ThreadPoolExecutor(max_workers=20)
            futures = [executor.submit(search_slice, body, client, ["doc_id", "parent_id"]) for body in queries]
            hits = []
            for future in futures:
                hits.extend(future.result())
            parent_ids = set()
            self.parent_docs = []
            for hit in hits:
                if "parent_id" not in hit["_source"] or hit["_source"]["parent_id"] is None:
                    hit["_source"]["parent_id"] = hit["_id"]
                if hit["_source"]["parent_id"] not in parent_ids:
                    self.parent_docs.append(hit)
                    parent_ids.add(hit["_source"]["parent_id"])

            logger.info(f"Parent docs: {len(self.parent_docs)}")
            self.read_fn = self._get_read_task_reconstruct
        else:
            self.read_fn = self._get_read_task_slice

    def _get_read_task_reconstruct(self, task_index, parallelism: int) -> ReadTask:

        start = task_index * len(self.parent_docs) // parallelism
        end = (task_index + 1) * len(self.parent_docs) // parallelism
        row_count = end - start
        def read_fn() -> Iterable[Block]:
            query_params = OpenSearchReaderQueryParams(index_name=self.index, query=self.query, reconstruct_document=self.doc_reconstruct)
            response = OpenSearchReaderQueryResponse(self.parent_docs[start:end])
            docs = response.to_docs(query_params, **self.client_args)
            columns = set()
            for doc in docs:
                columns.update(doc.data.keys())
                elements = doc.elements
                element_dicts = [element.data for element in elements]
                doc.elements = element_dicts
            pydict = {column: [row.get(column, None) for row in docs] for column in columns}
            yield pa.Table.from_pydict(pydict)

        md = BlockMetadata(num_rows=row_count, size_bytes=0, schema=None, input_files=None, exec_stats=None)
        return ReadTask(read_fn=read_fn, metadata=md)

    def _get_read_task_slice(self, task_index, parallelism: int):
        client = OpenSearch(**self.client_args)
        query = {
            "slice": {
                "id": task_index,
                "max": self.slice_count,
            },
            "pit": {
                "id": self.pit_id,
                "keep_alive": "1m",
            },
            "sort": [{SEARCH_AFTER_SORT_FIELD: "asc"}],
        }
        if "query" in self.query:
            query["query"] = self.query["query"]

        num_rows = get_doc_count_for_slice(client, query)
        def read_fn() -> Iterable[Block]:
            client = OpenSearch(**self.client_args)
            hits = search_slice(query, client)
            query_params = OpenSearchReaderQueryParams(index_name=self.index, query=query,
                                                       reconstruct_document=self.doc_reconstruct)
            response = OpenSearchReaderQueryResponse(hits)
            docs = response.to_docs(query_params, **self.client_args)
            columns = set()
            for doc in docs:
                columns.update(doc.data.keys())
                elements = doc.elements
                element_dicts = [element.data for element in elements]
                doc.elements = element_dicts
            pydict = {column: [row.get(column, None) for row in docs] for column in columns}
            yield pa.Table.from_pydict(pydict)

        md = BlockMetadata(num_rows=num_rows, size_bytes=0, schema=None, input_files=None, exec_stats=None)
        return ReadTask(read_fn=read_fn, metadata=md)

    def estimate_inmemory_data_size(self) -> int:
        # If we're doing a full scan, we can get the index size via /_cat/indices
        return 0

    def get_read_tasks(self, parallelism: int) -> list[ReadTask]:
        assert parallelism > 0, f"Invalid parallelism {parallelism}"

        if self.doc_reconstruct:
            if parallelism > len(self.parent_docs):
                parallelism = len(self.parent_docs)
        else:
            if parallelism > self.slice_count:
                parallelism = self.slice_count

        return [self.read_fn(i, parallelism) for i in range(parallelism)]