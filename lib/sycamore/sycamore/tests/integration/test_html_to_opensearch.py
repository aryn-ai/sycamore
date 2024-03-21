import json
import tempfile

import sycamore
from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import HtmlPartitioner


def test_html_to_opensearch():
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
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "nmslib"},
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

        context = sycamore.init()
        ds = (
            context.read.binary(
                base_path, binary_format="html", metadata_provider=JsonManifestMetadataProvider(manifest_path)
            )
            .partition(partitioner=HtmlPartitioner())
            .explode()
            .embed(SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"))
        )
        # assert metadata properties are propagated to child elements
        for doc in ds.take(5):
            assert doc.properties["remote_url"] == remote_url
            assert doc.properties["indexed_at"] == indexed_at

        ds.write.opensearch(os_client_args=os_client_args, index_name="toyindex", index_settings=index_settings)
    finally:
        tmp_manifest.close()
