import json
import tempfile
from typing import Any

from sycamore.data import Document
from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.scans import BinaryScan, JsonScan
from sycamore.tests.config import TEST_DIR


class TestFileScan:
    def test_binary_scan(self):
        paths = str(TEST_DIR / "resources/data/pdfs/")
        scan = BinaryScan(paths, binary_format="pdf")
        ds = scan.execute()
        assert ds.schema().names == ["doc"]

    def test_json_scan(self):
        paths = str(TEST_DIR / "resources/data/json/")
        scan = JsonScan(paths, properties="props")
        ds = scan.execute()
        raw_doc = ds.take(1)[0]

        doc = Document.from_row(raw_doc)

        assert set(doc.properties.keys()) == set(["props", "path"])
        assert doc.properties["props"] == "propValue"

    def test_json_scan_all_props(self):
        paths = str(TEST_DIR / "resources/data/json/example.json")
        scan = JsonScan(paths)
        ds = scan.execute()
        raw_doc = ds.take(1)[0]

        doc = Document.from_row(raw_doc)

        assert set(doc.properties.keys()) == set(["web-app", "props", "path"])
        assert doc.properties["props"] == "propValue"
        assert isinstance(doc.properties["web-app"], dict)

    def test_json_scan_body_field(self):
        paths = str(TEST_DIR / "resources/data/json/example.json")
        scan = JsonScan(paths, document_body_field="props")
        ds = scan.execute()
        raw_doc = ds.take(1)[0]

        doc = Document.from_row(raw_doc)

        assert doc.text_representation == "propValue"

    def test_nested_json_scan(self):
        def _to_document(json_dict: dict[str, Any]) -> list[dict[str, Any]]:
            rows = json_dict["rows"]
            result = []
            for row in rows:
                document = Document(row)
                result += [{"doc": document.serialize()}]
            return result

        paths = str(TEST_DIR / "resources/data/nested_json/")
        scan = JsonScan(paths, doc_extractor=_to_document)
        ds = scan.execute()
        raw_doc = ds.take(1)[0]

        doc = Document.from_row(raw_doc)

        assert "row" in doc.data
        assert "question" in doc.data["row"]

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
            doc = Document.from_row(ds.take(1)[0])
            assert doc.properties["remote_url"] == remote_url
            assert doc.properties["indexed_at"] == indexed_at
        finally:
            tmp_manifest.close()
