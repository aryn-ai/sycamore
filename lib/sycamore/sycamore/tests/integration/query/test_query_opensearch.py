import pytest
import sycamore
from sycamore.data.document import OpenSearchQuery
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.query import OpenSearchQueryExecutor
from opensearchpy import OpenSearch


@pytest.fixture(scope="class")
def setup_index():
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
    paths = str(TEST_DIR / "resources/data/pdfs/")

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
        os_client_args=TestQueryOpenSearch.OS_CLIENT_ARGS,
        index_name=TestQueryOpenSearch.INDEX,
        index_settings=index_settings,
    )
    osc = OpenSearch(**TestQueryOpenSearch.OS_CLIENT_ARGS)
    osc.indices.refresh(TestQueryOpenSearch.INDEX)
    yield TestQueryOpenSearch.INDEX
    osc.indices.delete(TestQueryOpenSearch.INDEX)


class TestQueryOpenSearch:
    INDEX = "toyindex"

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

    def test_single_query(self, setup_index):
        query_executor = OpenSearchQueryExecutor(self.OS_CLIENT_ARGS)
        query = OpenSearchQuery()
        query.query = {"query": {"match_all": {}}, "size": 1}
        query.index = self.INDEX
        result = query_executor.query(query)
        assert len(result.hits) > 0

    def test_query_docset(self, setup_index):
        query_executor = OpenSearchQueryExecutor(self.OS_CLIENT_ARGS)

        query1 = OpenSearchQuery()
        query1.query = {"query": {"match_all": {}}, "size": 1}
        query1.index = self.INDEX

        query2 = OpenSearchQuery()
        query2.query = {"query": {"match_all": {}}, "size": 2}
        query2.index = self.INDEX

        queries = [query1, query2]

        context = sycamore.init()
        result = context.read.document(queries).query(query_executor=query_executor)

        assert result.count() == 2

        query_results = result.take(2)

        assert len(query_results[0]["hits"]) == 1
        assert len(query_results[1]["hits"]) == 2
