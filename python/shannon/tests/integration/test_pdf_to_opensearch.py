import shannon
from shannon.tests.config import TEST_DIR


def test_pdf_to_opensearch():
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ('admin', 'admin'),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120
    }

    index_settings = {
        "body": {
            "settings": {
                "index.knn": True,
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib"
                        }
                    },
                    "text": {
                        "type": "text"
                    }
                }
            }
        }
    }

    paths = str(TEST_DIR / "resources/data/pdfs/")
    context = shannon.init()
    ds = context.read.binary(paths, binary_format="pdf") \
        .partition_pdf("bytes", max_partition=256) \
        .sentence_transformer_embed(
        col_name="bytes",
        batch_size=100,
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    ds.write.opensearch(os_client_args=os_client_args, index_name="toyindex", index_settings=index_settings)
