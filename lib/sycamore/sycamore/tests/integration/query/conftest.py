import os
import pytest

import sycamore
from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging
from sycamore.data import Document
from sycamore.functions import HuggingFaceTokenizer
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner

QUERY_INTEGRATION_TEST_INDEX_NAME = "sycamore_query_ntsb_integration_tests"
OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")

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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = HuggingFaceTokenizer(model_name)

    context = sycamore.init()
    ds = (
        context.read.binary(paths, binary_format="pdf")
        .limit(1)
        .partition(partitioner=UnstructuredPdfPartitioner())
        .merge(GreedyTextElementMerger(tokenizer=tokenizer, max_tokens=1000))
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
    osc = OpenSearchClientWithLogging(**OS_CLIENT_ARGS)
    osc.indices.refresh(QUERY_INTEGRATION_TEST_INDEX_NAME)
    yield QUERY_INTEGRATION_TEST_INDEX_NAME
    osc.indices.delete(QUERY_INTEGRATION_TEST_INDEX_NAME)


@pytest.fixture(scope="package")
def query_integration_test_index2():
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
    paths = [
        str(TEST_DIR / "resources/data/pdfs/ntsb0.pdf"),
        str(TEST_DIR / "resources/data/pdfs/ntsb1.pdf"),
        str(TEST_DIR / "resources/data/pdfs/ntsb3.pdf"),
    ]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = HuggingFaceTokenizer(model_name)

    context = sycamore.init()

    def set_page_numbers(d: Document) -> Document:
        d.properties["page_numbers"] = d.elements[0].properties["page_numbers"]
        return d

    ds = (
        context.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .merge(GreedyTextElementMerger(tokenizer=tokenizer, max_tokens=1000))
        .map(set_page_numbers)
        .spread_properties(["path"])
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
    osc = OpenSearchClientWithLogging(**OS_CLIENT_ARGS)
    osc.indices.refresh(QUERY_INTEGRATION_TEST_INDEX_NAME)
    yield QUERY_INTEGRATION_TEST_INDEX_NAME
    osc.indices.delete(QUERY_INTEGRATION_TEST_INDEX_NAME)
