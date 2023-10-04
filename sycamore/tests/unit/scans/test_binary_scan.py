import json
import tempfile

from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.scans import BinaryScan
from sycamore.tests.config import TEST_DIR


class TestBinaryScan:
    def test_partition(self):
        paths = str(TEST_DIR / "resources/data/pdfs/")
        scan = BinaryScan(paths, binary_format="pdf")
        ds = scan.execute()
        assert ds.schema().names == [
            "doc_id",
            "type",
            "text_representation",
            "binary_representation",
            "elements",
            "embedding",
            "parent_id",
            "bbox",
            "properties",
        ]

    def test_json_manifest(self):
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

            scan = BinaryScan(
                base_path, binary_format="html", metadata_provider=JsonManifestMetadataProvider(manifest_path)
            )
            ds = scan.execute()
            doc = ds.take(1)[0]
            assert doc["properties"]["remote_url"] == remote_url
            assert doc["properties"]["indexed_at"] == indexed_at
        finally:
            tmp_manifest.close()
