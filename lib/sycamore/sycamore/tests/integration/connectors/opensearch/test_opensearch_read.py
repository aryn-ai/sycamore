import os
import tempfile
import time
from tempfile import tempdir
from uuid import uuid4

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore.data import Document
from sycamore.tests.integration.connectors.common import compare_connector_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.utils.cache import cache_from_path

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")


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
        "http_auth": ("admin", OS_ADMIN_PASSWORD),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    def test_ingest_and_read(self, setup_index, exec_mode):
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

        # hack to allow opensearch time to index the data. Without this it's possible we try to query the index before
        # all the records are available
        time.sleep(1)
        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)
        query_materialized = query_docs.take_all()
        retrieved_materialized_reconstructed = sorted(retrieved_docs_reconstructed.take_all(), key=lambda d: d.doc_id)

        with OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS) as os_client:
            os_client.indices.delete(TestOpenSearchRead.INDEX)
        assert len(query_materialized) == 1  # exactly one doc should be returned
        compare_connector_docs(original_materialized, retrieved_materialized)

        assert len(retrieved_materialized_reconstructed) == 1
        doc = retrieved_materialized_reconstructed[0]
        assert len(doc.elements) == len(retrieved_materialized) - 1  # drop the document parent record

        for i in range(len(doc.elements) - 1):
            assert doc.elements[i].element_index < doc.elements[i + 1].element_index

    def _test_ingest_and_read_via_docid_reconstructor(self, setup_index, exec_mode, cache_dir):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        print(f"Using cache dir: {cache_dir}")

        # doc_cache = cache_from_path(cache_dir)
        def doc_reconstructor(index_name: str, doc_id: str) -> Document:
            prefix = f"doc-{doc_id}"
            found = None
            files = os.listdir(cache_dir)
            for f in files:
                if f.startswith(prefix):
                    found = f
                    break
            assert found, "Doc not found in cache"
            import pickle

            data = pickle.load(open(f"{cache_dir}/{found}", "rb"))
            return Document(**data)

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=exec_mode)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .materialize(cache_dir)
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=TestOpenSearchRead.INDEX,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        with OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS) as os_client:
            os_client.indices.refresh(TestOpenSearchRead.INDEX)

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
            doc_reconstructor=doc_reconstructor,
        )
        original_materialized = sorted(original_docs, key=lambda d: d.doc_id)

        # hack to allow opensearch time to index the data. Without this it's possible we try to query the index before
        # all the records are available
        time.sleep(1)
        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)
        query_materialized = query_docs.take_all()
        retrieved_materialized_reconstructed = sorted(retrieved_docs_reconstructed.take_all(), key=lambda d: d.doc_id)

        # with OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS) as os_client:
        #    os_client.indices.delete(TestOpenSearchRead.INDEX)
        assert len(query_materialized) == 1  # exactly one doc should be returned
        compare_connector_docs(original_materialized, retrieved_materialized)

        assert len(retrieved_materialized_reconstructed) == 1
        doc = retrieved_materialized_reconstructed[0]
        assert len(doc.elements) == len(retrieved_materialized) - 1  # drop the document parent record

        for i in range(len(doc.elements) - 1):
            assert doc.elements[i].element_index < doc.elements[i + 1].element_index

        # Clean slate between Execution Modes
        with OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS) as os_client:
            os_client.indices.delete(TestOpenSearchRead.INDEX)
            os_client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)
            os_client.indices.refresh(TestOpenSearchRead.INDEX)

    def test_ingest_and_read_via_docid_reconstructor(self, setup_index, exec_mode):
        with tempfile.TemporaryDirectory() as cache_dir:
            self._test_ingest_and_read_via_docid_reconstructor(setup_index, exec_mode, cache_dir)
