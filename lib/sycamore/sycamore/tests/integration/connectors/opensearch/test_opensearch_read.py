import os
import tempfile
import uuid

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore import EXEC_LOCAL
from sycamore.connectors.doc_reconstruct import DocumentReconstructor
from sycamore.data import Document
from sycamore.tests.integration.connectors.common import compare_connector_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")


@pytest.fixture(scope="class")
def os_client():
    client = OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS)
    yield client


@pytest.fixture(scope="class")
def setup_index(os_client):
    # client = OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS)

    # Recreate before
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)
    os_client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)

    yield TestOpenSearchRead.INDEX

    # Delete after
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)


def get_doc_count(os_client, index_name: str) -> int:
    res = os_client.cat.indices(format="json", index=index_name)
    return int(res[0]["docs.count"])


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

    def test_ingest_and_read2(self, setup_index, os_client, exec_mode):
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
        print(f"Expected {expected_count} documents, found {actual_count}")

        # refresh should have made all ingested docs immediately available for search
        # assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        pit = os_client.create_pit(index=TestOpenSearchRead.INDEX, keep_alive="100m")
        search_body = {
            "query": {
                "match_all": {},
            },
            # "size": 100,
            "pit": {
                "id": pit["pit_id"],
                "keep_alive": "100m"
            },
            "slice": {"id": 0, "max": 10},
        }

        total = 0
        ids = set()
        for i in range(10):
            for j in range(10):
                search_body["slice"]["id"] = i
                res = os_client.search(body=search_body, size=10, from_=j*10)
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

        print(len(ids))
