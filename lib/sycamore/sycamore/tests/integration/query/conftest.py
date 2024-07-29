import pytest
from opensearchpy import OpenSearch

import sycamore
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import UnstructuredPdfPartitioner

QUERY_INTEGRATION_TEST_INDEX_NAME = "sycamore_query_ntsb_integration_tests"

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

OS_CONFIG = {
    "search_pipeline": "hybrid_pipeline",
}


@pytest.fixture(scope="package")
def query_integration_test_index():
    index_settings = {
        "body": {
            "settings": {
                "index.knn": True,
                "number_of_shards": 5,
                "number_of_replicas": 1,
            },
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
    paths = str(TEST_DIR / "resources/data/pdfs/ntsb-report.pdf")

    context = sycamore.init()
    ds = (
        context.read.binary(paths, binary_format="pdf")
        .limit(1)
        .partition(partitioner=UnstructuredPdfPartitioner())
        .explode()
        .embed(
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    )
    ds.write.opensearch(
        os_client_args=OS_CLIENT_ARGS,
        index_name=QUERY_INTEGRATION_TEST_INDEX_NAME,
        index_settings=index_settings,
    )
    osc = OpenSearch(**OS_CLIENT_ARGS)
    osc.indices.refresh(QUERY_INTEGRATION_TEST_INDEX_NAME)
    yield QUERY_INTEGRATION_TEST_INDEX_NAME
    osc.indices.delete(QUERY_INTEGRATION_TEST_INDEX_NAME)
