import time

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore.connectors.common import compare_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner


@pytest.fixture(scope="class")
def setup_index():
    client = OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS)

    # Recreate before
    client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)
    client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)

    yield TestOpenSearchRead.INDEX

    # Delete after
    client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)


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
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    def test_ingest_and_read(self, setup_index):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init()
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

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX
        )
        target_doc_id = original_docs[-1].doc_id if original_docs[-1].doc_id else ""
        query = {"query": {"term": {"_id": target_doc_id}}}
        query_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX, query=query
        )
        original_materialized = sorted(original_docs, key=lambda d: d.doc_id)

        # hack to allow opensearch time to index the data. Without this it's possible we try to query the index before
        # all the records are available
        time.sleep(1)
        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)
        query_materialized = query_docs.take_all()
        with OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS) as os_client:
            os_client.indices.delete(TestOpenSearchRead.INDEX)
        assert len(original_materialized) == len(retrieved_materialized)
        assert len(query_materialized) == 1  # exactly one doc should be returned
        for original, retrieved in zip(original_materialized, retrieved_materialized):
            assert compare_docs(original, retrieved)
