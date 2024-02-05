import sycamore
from sycamore.data.document import OpenSearchQuery
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from transforms.query import OpenSearchQueryExecutor

INDEX = "toyindex"


def test_simple_query():
    setup()
    query_executor = OpenSearchQueryExecutor("http://localhost:9200")
    query = OpenSearchQuery()
    query.query = {"query": {"match_all": {}}, "size": 100}
    query.url_params = f"/{INDEX}/_search"
    result = query_executor.query(query)
    assert result.query["params"]["q"] == query.query
    assert result.query["url"] == "http://localhost:9200" + query.url_params
    assert len(result.hits) > 0


def setup():
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
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
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    )

    ds.write.opensearch(
        os_client_args=os_client_args,
        index_name=INDEX,
        index_settings=index_settings,
    )
