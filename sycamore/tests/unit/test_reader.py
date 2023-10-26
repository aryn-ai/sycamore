import json
import tempfile

from sycamore.docset import DocSet
import sycamore
from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.tests.config import TEST_DIR


class TestDocSetReader:
    def test_pdf(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/pdf", binary_format="pdf")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "pdf"

    def test_html_binary(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/html", binary_format="html")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "html"

    def test_manifest(self):
        base_path = str(TEST_DIR / "resources/data/htmls/")
        remote_url = "https://en.wikipedia.org/wiki/Binary_search_algorithm"
        indexed_at = "2023-10-04"
        manifest = {base_path + "/wikipedia_binary_search.html": {"remote_url": remote_url, "indexed_at": indexed_at}}
        tmp_manifest = tempfile.NamedTemporaryFile(mode="w+")
        try:
            json.dump(manifest, tmp_manifest)
            tmp_manifest.flush()
            manifest_path = tmp_manifest.name

            context = sycamore.init()
            docset = context.read.manifest(JsonManifestMetadataProvider(manifest_path), binary_format="html")
            assert isinstance(docset, DocSet)
            assert docset.plan.format() == "html"

        finally:
            tmp_manifest.close()
