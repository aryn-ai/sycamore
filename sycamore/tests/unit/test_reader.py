import json
from pathlib import Path

import sycamore
from sycamore.docset import DocSet
from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.tests.config import TEST_DIR


class TestDocSetReader:
    def test_pdf(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/pdf", binary_format="pdf")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "pdf"

    def test_json(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/json", binary_format="json")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "json"

    def test_html_binary(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/html", binary_format="html")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "html"

    def test_manifest(self, tmp_path: Path):
        base_path = str(TEST_DIR / "resources/data/htmls/")
        remote_url = "https://en.wikipedia.org/wiki/Binary_search_algorithm"
        indexed_at = "2023-10-04"
        manifest = {base_path + "/wikipedia_binary_search.html": {"remote_url": remote_url, "indexed_at": indexed_at}}
        manifest_loc = str(f"{tmp_path}/manifest.json")

        with open(manifest_loc, "w") as file:
            json.dump(manifest, file)

        context = sycamore.init()
        docset = context.read.manifest(JsonManifestMetadataProvider(manifest_loc), binary_format="html")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "html"  # type: ignore
