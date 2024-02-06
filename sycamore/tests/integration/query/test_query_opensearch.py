import sycamore
from sycamore.data.document import OpenSearchQuery
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.query import OpenSearchQueryExecutor


class TestQueryOpenSearch:
    INDEX = "toyindex"

    @classmethod
    def setup_class(cls):
        os_client_args = {
            "hosts": [{"host": "localhost", "port": 9200}],
            "http_compress": True,
            "http_auth": ("admin", "admin"),
            "use_ssl": True,
            "verify_certs": False,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 120,
        }

        index_settings = {
            "body": {
                "settings": {
                    "index.knn": True,
                    "number_of_shards": 5,
                    "number_of_replicas": 1,
                },
                "mappings": {
                    "properties": {
                        "embeddings": {
                            "type": "knn_vector",
                            "dimension": 384,
                            "method": {"name": "hnsw", "engine": "nmslib"},
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
                embedder=SentenceTransformerEmbedder(
                    batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        )

        ds.write.opensearch(
            os_client_args=os_client_args,
            index_name=cls.INDEX,
            index_settings=index_settings,
        )

    def test_single_query(self):
        query_executor = OpenSearchQueryExecutor("http://localhost:9200")
        query = OpenSearchQuery()
        query.query = {"query": {"match_all": {}}, "size": 100}
        query.url_params = f"/{self.INDEX}/_search"
        result = query_executor.query(query)
        assert result.query["params"]["q"] == query.query
        assert result.query["url"] == "http://localhost:9200" + query.url_params
        assert len(result.hits) > 0

    def test_query_docset(self):
        query_executor = OpenSearchQueryExecutor("http://localhost:9200")

        query1 = OpenSearchQuery()
        query1.query = {"query": {"match_all": {}}}
        query1.url_params = f"/{self.INDEX}/_search?size=1"

        query2 = OpenSearchQuery()
        query2.query = {"query": {"match_all": {}}}
        query2.url_params = f"/{self.INDEX}/_search?size=2"

        queries = [query1, query2]

        context = sycamore.init()
        result = context.read.document(queries).query(query_executor=query_executor)

        assert result.count() == 2

        query_results = result.take(2)
        print(query_results[0])
        assert len(query_results[0]["hits"]) == 1
        assert len(query_results[1]["hits"]) == 2
