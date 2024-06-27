import time

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore.connectors.opensearch.opensearch_writer import DEFAULT_OPENSEARCH_RECORD_PROPERTIES
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner


@pytest.fixture(scope="class")
def setup_index():
    index_settings = {
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
    client = OpenSearch(**TestOpenSearchScan.OS_CLIENT_ARGS)

    # Recreate before
    client.indices.delete(TestOpenSearchScan.INDEX, ignore_unavailable=True)
    client.indices.create(TestOpenSearchScan.INDEX, **index_settings)

    yield TestOpenSearchScan.INDEX

    # Delete after
    client.indices.delete(TestOpenSearchScan.INDEX, ignore_unavailable=True)


def filter_doc(obj, include):
    return {k: v for k, v in obj.__dict__.items() if k in include}


def compare_docs(doc1, doc2):
    filtered_doc1 = filter_doc(doc1, DEFAULT_OPENSEARCH_RECORD_PROPERTIES.keys())
    filtered_doc2 = filter_doc(doc2, DEFAULT_OPENSEARCH_RECORD_PROPERTIES.keys())
    return filtered_doc1 == filtered_doc2


class TestOpenSearchScan:

    INDEX = "test_opensearch_scan"

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
                os_client_args=TestOpenSearchScan.OS_CLIENT_ARGS, index_name=TestOpenSearchScan.INDEX, execute=False
            )
        )

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchScan.OS_CLIENT_ARGS, index_name=TestOpenSearchScan.INDEX
        )

        original_materialized = sorted(original_docs.take_all(), key=lambda d: d.doc_id)

        # hack to allow opensearch time to index the data. Without this it's possible we try to query the index before
        # all the records are available
        time.sleep(1)
        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)

        assert len(original_materialized) == len(retrieved_materialized)

        for original, retrieved in zip(original_materialized, retrieved_materialized):
            assert compare_docs(original, retrieved)
