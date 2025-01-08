import os
import tempfile
import uuid
from typing import Optional, Dict, Any

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore import EXEC_LOCAL, ExecMode
from sycamore.connectors.doc_reconstruct import DocumentReconstructor
from sycamore.data import Document
from sycamore.tests.integration.connectors.common import compare_connector_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")
TEST_CACHE_DIR = "/tmp/test_cache_dir"


@pytest.fixture(scope="class")
def os_client():
    client = OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS)
    yield client


@pytest.fixture(scope="class")
def setup_index(os_client):
    # Recreate before
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)
    os_client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)

    yield TestOpenSearchRead.INDEX

    # Delete after
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)


@pytest.fixture(scope="class")
def setup_index_large(os_client):

    yield "test_opensearch_read_large"


def get_doc_count(os_client, index_name: str, query: Optional[Dict[str, Any]] = None) -> int:
    res = os_client.count(index=index_name)
    return res["count"]


class TestOpenSearchRead:
    INDEX_SETTINGS = {
        "body": {
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "faiss"},
                    },
                    "text": {"type": "text"},
                }
            },
        }
    }

    INDEX = "test_opensearch_read"

    OS_CLIENT_ARGS = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", OS_ADMIN_PASSWORD),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    def test_ingest_and_read(self, setup_index, os_client, exec_mode):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=exec_mode)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=TestOpenSearchRead.INDEX,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(TestOpenSearchRead.INDEX)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, TestOpenSearchRead.INDEX)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX
        )
        target_doc_id = original_docs[-1].doc_id if original_docs[-1].doc_id else ""
        query = {"query": {"term": {"_id": target_doc_id}}}
        query_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX, query=query
        )

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=TestOpenSearchRead.INDEX,
            reconstruct_document=True,
        )
        original_materialized = sorted(original_docs, key=lambda d: d.doc_id)

        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)
        query_materialized = query_docs.take_all()
        retrieved_materialized_reconstructed = sorted(retrieved_docs_reconstructed.take_all(), key=lambda d: d.doc_id)

        os_client.indices.delete(TestOpenSearchRead.INDEX)
        assert len(query_materialized) == 1  # exactly one doc should be returned
        compare_connector_docs(original_materialized, retrieved_materialized)

        assert len(retrieved_materialized_reconstructed) == 1
        doc = retrieved_materialized_reconstructed[0]
        assert len(doc.elements) == len(retrieved_materialized) - 1  # drop the document parent record

        for i in range(len(doc.elements) - 1):
            assert doc.elements[i].element_index < doc.elements[i + 1].element_index

    def _test_ingest_and_read_via_docid_reconstructor(self, setup_index, os_client, cache_dir):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        print(f"Using cache dir: {cache_dir}")

        def doc_reconstructor(index_name: str, doc_id: str) -> Document:
            import pickle

            data = pickle.load(open(f"{cache_dir}/{TestOpenSearchRead.INDEX}-{doc_id}", "rb"))
            return Document(**data)

        def doc_to_name(doc: Document, bin: bytes) -> str:
            return f"{TestOpenSearchRead.INDEX}-{doc.doc_id}"

        context = sycamore.init(exec_mode=EXEC_LOCAL)
        hidden = str(uuid.uuid4())
        # make sure we read from pickle files -- this part won't be written into opensearch.
        dicts = [
            {
                "doc_id": "1",
                "hidden": hidden,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "2",
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": "3",
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {
                "doc_id": "4",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": "5",
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {
                "doc_id": "6",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        original_docs = (
            context.read.document(docs)
            .materialize(path={"root": cache_dir, "name": doc_to_name})
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=TestOpenSearchRead.INDEX,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(TestOpenSearchRead.INDEX)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, TestOpenSearchRead.INDEX)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=TestOpenSearchRead.INDEX,
            reconstruct_document=True,
            doc_reconstructor=DocumentReconstructor(TestOpenSearchRead.INDEX, doc_reconstructor),
        ).take_all()

        assert len(retrieved_docs_reconstructed) == 6
        retrieved_sorted = sorted(retrieved_docs_reconstructed, key=lambda d: d.doc_id)
        assert retrieved_sorted[0].data["hidden"] == hidden
        assert docs == retrieved_sorted

        # Clean slate between Execution Modes
        os_client.indices.delete(TestOpenSearchRead.INDEX)
        os_client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)
        os_client.indices.refresh(TestOpenSearchRead.INDEX)

    def test_ingest_and_read_via_docid_reconstructor(self, setup_index, os_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            self._test_ingest_and_read_via_docid_reconstructor(setup_index, os_client, cache_dir)

    def test_slice_and_shards(self, setup_index_large, os_client):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        """
        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=TestOpenSearchRead.INDEX,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(TestOpenSearchRead.INDEX)

        expected_count = len(original_docs)
        """
        actual_count = get_doc_count(os_client, setup_index_large)
        # print(f"Expected {expected_count} documents, found {actual_count}")

        # refresh should have made all ingested docs immediately available for search
        # assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        pit = os_client.create_pit(index=setup_index_large, keep_alive="100m")
        num_slices = 20
        search_body = {
            "query": {
                "match_all": {},
            },
            # "size": 100,
            "pit": {"id": pit["pit_id"], "keep_alive": "100m"},
            "slice": {"id": 0, "max": num_slices},
        }

        ids = set()
        page_size = 1000
        for i in range(num_slices):
            slice_count = 0
            for j in range(10):
                search_body["slice"]["id"] = i
                res = os_client.search(body=search_body, size=page_size, from_=j * page_size)
                print(f"{j}: {res['hits']['total']['value']}")
                hits = res["hits"]["hits"]
                if hits is None:
                    print(f"None: {j}")
                else:
                    print(f"Length: {len(hits)}")
                for hit in hits:
                    doc_id = hit["_source"]["doc_id"]
                    if doc_id in ids:
                        print(f"Duplicate doc_id: {doc_id}")
                    else:
                        ids.add(doc_id)
                        slice_count += 1
            print(f"Slice {i} count: {slice_count}")

        print(len(ids))

    def doc_to_name(doc: Document, bin: bytes) -> str:
        return f"{TestOpenSearchRead.INDEX}-{doc.doc_id}"

    def test_parallel_read_reconstruct(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=True,
        ).take_all()

        print(f"Retrieved {len(retrieved_docs_reconstructed)} documents")
        expected_docs = self.get_ids(os_client, setup_index_large, parents_only=True)
        assert len(retrieved_docs_reconstructed) == len(expected_docs)
        for doc in retrieved_docs_reconstructed:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=False,
        ).take_all()

        print(f"Retrieved {len(retrieved_docs)} documents")
        expected_docs = self.get_ids(os_client, setup_index_large)
        assert len(retrieved_docs) == len(expected_docs)
        for doc in retrieved_docs:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read_reconstruct_with_pit(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=True,
            query_kwargs={"use_pit": True},
        ).take_all()

        print(f"Retrieved {len(retrieved_docs_reconstructed)} documents")
        expected_docs = self.get_ids(os_client, setup_index_large, parents_only=True)
        assert len(retrieved_docs_reconstructed) == len(expected_docs)
        for doc in retrieved_docs_reconstructed:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read_with_pit(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=False,
            query_kwargs={"use_pit": True},
        ).take_all()

        print(f"Retrieved {len(retrieved_docs)} documents")
        expected_docs = self.get_ids(os_client, setup_index_large)
        assert len(retrieved_docs) == len(expected_docs)
        for doc in retrieved_docs:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    @staticmethod
    def get_ids(
        os_client, index_name, parents_only: bool = False, query: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        if query is None:
            query = {"query": {"match_all": {}}}

        query_params = {"_source_includes": ["doc_id", "parent_id"]}
        if "scroll" not in query_params:
            query_params["scroll"] = "10m"
        response = os_client.search(index=index_name, body=query, **query_params)
        scroll_id = response["_scroll_id"]
        # no_parent_ids = 0
        parents = set()
        no_parents = set()
        all_hits = 0
        docs = {}
        try:
            while True:
                hits = response["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    all_hits += 1
                    # result.append(hit)
                    if "parent_id" in hit["_source"] and hit["_source"]["parent_id"] is not None:
                        parents.add(hit["_source"]["parent_id"])
                        # print(f"Parent id: {hit['_source']['parent_id']}")
                    else:
                        no_parents.add(hit["_id"])
                        # print(f"No parent id: {hit['_id']}")

                    if parents_only:
                        if "parent_id" not in hit["_source"] or hit["_source"]["parent_id"] is None:
                            docs[hit["_id"]] = {
                                "doc_id": hit["_source"]["doc_id"],
                                "parent_id": hit["_source"]["parent_id"],
                            }
                    else:
                        docs[hit["_id"]] = {
                            "doc_id": hit["_source"]["doc_id"],
                            "parent_id": hit["_source"].get("parent_id"),
                        }

                response = os_client.scroll(scroll_id=scroll_id, scroll=query_params["scroll"])
        finally:
            os_client.clear_scroll(scroll_id=scroll_id)
        # print(f"Parents: {len(parents)}, no parents: {len(no_parents)}, all hits: {all_hits}")
        # print(parents)
        # print("-------------------------------")
        # print(no_parents)

        return docs

    def test_bulk_load(self, setup_index_large, os_client):

        # Only run this to populate a test index.
        return

        if not os_client.indices.exists(setup_index_large):
            os_client.indices.create(setup_index_large, **TestOpenSearchRead.INDEX_SETTINGS)
        """
        Used for generating a large number of documents in OpenSearch for testing purposes.
        """
        os_client.indices.refresh(setup_index_large)
        doc_count = get_doc_count(os_client, setup_index_large)
        print(f"Current count: {doc_count}")

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        while doc_count < 20000:
            (
                context.read.binary(path, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                # .materialize(path={"root": TEST_CACHE_DIR, "name": self.doc_to_name})
                .explode()
                .write.opensearch(
                    os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                    index_name=setup_index_large,
                    index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                    execute=False,
                )
                .take_all()
            )

            os_client.indices.refresh(setup_index_large)

            # expected_count = len(original_docs)
            doc_count = get_doc_count(os_client, setup_index_large)
            print(f"Current count: {doc_count}")

        print(f"Current count: {doc_count}")

    def test_cat(self, setup_index_large, os_client):
        response = os_client.cat.shards(index=setup_index_large, format="json")
        print(response)
        doc_count = 0
        for item in response:
            if item["prirep"] == "p":
                print(item)
                doc_count += int(item["docs"])
        print(f"Total docs: {doc_count}")