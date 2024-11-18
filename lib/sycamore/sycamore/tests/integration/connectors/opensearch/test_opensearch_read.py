import os
import time

import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore.connectors.opensearch.opensearch_reader import OpenSearchReaderClient, \
    OpenSearchReaderClientParams
from sycamore.tests.integration.connectors.common import compare_connector_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner

os_admin_password = os.getenv("OS_ADMIN_PASSWORD", "admin")

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
        "http_auth": ("admin", os_admin_password),
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

        kwargs = {'use_refs': True}

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX, **kwargs
        )
        target_doc_id = original_docs[-1].doc_id if original_docs[-1].doc_id else ""
        query = {"query": {"term": {"_id": target_doc_id}}}
        query_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX, query=query, **kwargs
        )

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=TestOpenSearchRead.INDEX,
            reconstruct_document=True,
            **kwargs
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

    def test_ingest_and_count(self, setup_index, exec_mode):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        client = OpenSearch(**TestOpenSearchRead.OS_CLIENT_ARGS)

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

        client.indices.refresh(index=TestOpenSearchRead.INDEX)

        use_refs = False
        kwargs = {'use_refs': use_refs}

        query = {"query": {"match_all": {}}}
        ds1 = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=TestOpenSearchRead.INDEX, query=query, **kwargs
        ).take_all()

        print(f"ExecMode: {exec_mode}, count: {len(ds1)}")

        ds2 = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=TestOpenSearchRead.INDEX,
            query=query,
            reconstruct_document=True,
            **kwargs
        ).take_all()  # count()

        print(f"ExecMode: {exec_mode}, count2: {len(ds2)}")

        assert len(ds2) == 1
        assert len(ds1) == 580

