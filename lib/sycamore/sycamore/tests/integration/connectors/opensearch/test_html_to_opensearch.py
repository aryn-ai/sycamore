import json
import os
import tempfile

from opensearchpy import OpenSearch
import sycamore
from sycamore.connectors.file.file_scan import JsonManifestMetadataProvider
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import HtmlPartitioner

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")

def test_html_to_opensearch(exec_mode):
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", OS_ADMIN_PASSWORD),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

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
                    "text_representation": {"type": "text"},
                }
            },
        }
    }

    base_path = str(TEST_DIR / "resources/data/htmls/")

    remote_url = "https://en.wikipedia.org/wiki/Binary_search_algorithm"
    indexed_at = "2023-10-04"
    manifest = {
        base_path + "/wikipedia_binary_search.html": {"remote_url": remote_url, "indexed_at": indexed_at},
        "other file.html": {"remote_url": "value", "indexed_at": "date"},
        "non-dict element": {"key1": "value1", "key2": ["listItem1", "listItem2"]},
        "list property": ["listItem1", "listItem2"],
    }
    tmp_manifest = tempfile.NamedTemporaryFile(mode="w+")
    try:
        json.dump(manifest, tmp_manifest)
        tmp_manifest.flush()
        manifest_path = tmp_manifest.name

        context = sycamore.init(exec_mode=exec_mode)
        ds = (
            context.read.binary(
                base_path, binary_format="html", metadata_provider=JsonManifestMetadataProvider(manifest_path)
            )
            .partition(partitioner=HtmlPartitioner())
            .explode()
            .sketch()
            .embed(SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"))
        )
        # assert metadata properties are propagated to child elements
        for doc in ds.take(5):
            assert doc.properties["remote_url"] == remote_url
            assert doc.properties["indexed_at"] == indexed_at

        ds.write.opensearch(os_client_args=os_client_args, index_name="toyindex", index_settings=index_settings)
    finally:
        tmp_manifest.close()
        OpenSearch(**os_client_args).indices.delete("toyindex")
