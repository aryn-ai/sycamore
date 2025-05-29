import json
from pathlib import Path
import tempfile
import unittest
import pytest

import sycamore
from sycamore.docset import DocSet
from sycamore.connectors.file.file_scan import JsonManifestMetadataProvider
from sycamore.tests.config import TEST_DIR
from sycamore.tests.unit.test_materialize import make_docs, NumCalls, mock_mrr_reset_fn, noop_fn, ids
from sycamore.context import ExecMode
from sycamore.materialize import MaterializeReadReliability


class TestDocSetReader:
    def test_pdf(self):
        context = sycamore.init()
        docset = context.read.binary("s3://bucket/prefix/pdf", binary_format="pdf")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "pdf"

    def test_json(self):
        context = sycamore.init()
        docset = context.read.json("s3://bucket/prefix/json")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "json"

    def test_json_doc(self):
        context = sycamore.init()
        docset = context.read.json_document("s3://bucket/prefix/json", binary_format="json")
        assert isinstance(docset, DocSet)
        assert docset.plan.format() == "jsonl"

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

    def test_opensearch_input_validation(self):
        context = sycamore.init()
        with pytest.raises(ValueError):
            filter = {"property1": ["1", "2"], "property2": ["3", "4"]}
            context.read.opensearch(os_client_args={}, index_name="test", result_filter=filter)

        with pytest.raises(ValueError):
            filter = {"property1": "1"}
            context.read.opensearch(os_client_args={}, index_name="test", result_filter=filter)

        with pytest.raises(ValueError):
            filter = {"property1": ["1", 2]}
            context.read.opensearch(os_client_args={}, index_name="test", result_filter=filter)

        with pytest.raises(ValueError):
            filter = {"property1": ["1", "2"]}
            query = {"query": {"knn": {"embedding": {}, "filter": {}}}}
            context.read.opensearch(os_client_args={}, index_name="test", query=query, result_filter=filter)


class TestFileReadReliability(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.LOCAL

    def test_read_with_wrong_nodes(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as tmpdir1:

            docs = make_docs(10)
            docs.pop()

            for doc in docs:
                path = Path(tmpdir) / f"{doc.doc_id}.{doc.properties.get('extension', 'json')}"
                path.write_bytes(b"test content")
            ctx = sycamore.init(exec_mode=self.exec_mode)
            mrr = MaterializeReadReliability(max_batch=3)

            ctx.rewrite_rules.append(mrr)

            ## Reliability does not work if first node is not binaryScan, Materialize
            with pytest.raises(AssertionError):
                ctx.read.json(tmpdir, binary_format="json").map(noop_fn).materialize(path=tmpdir1).execute()

    def test_binary_file_read_reliability_list_of_paths(self):
        ctx = sycamore.init(exec_mode=self.exec_mode)
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
            tempfile.TemporaryDirectory() as tmpdir3,
            tempfile.TemporaryDirectory() as tmpdir4,
        ):
            counter = NumCalls()
            docs = make_docs(10)
            docs.pop()
            mrr = MaterializeReadReliability(max_batch=3)
            mrr = mock_mrr_reset_fn(mrr, counter)

            ctx.rewrite_rules.append(mrr)
            paths = []
            for doc in docs:
                path = Path(tmpdir1) / f"{doc.doc_id}.{doc.properties.get('extension', 'pdf')}"
                path.write_bytes(b"test content")
                paths.append(str(path))
            (
                ctx.read.binary(paths, binary_format="pdf")
                .materialize(
                    path={"root": tmpdir2},
                )
                .execute()
            )

            e1 = ctx.read.materialize(path=tmpdir2).take_all()
            # Verify batching works (4 + 1 (mrr.reset at the end))
            assert counter.x == 5

            # Check with directory as well

            counter.x = 0

            (ctx.read.binary(tmpdir1, binary_format="pdf").materialize(path={"root": tmpdir3}).execute())
            e2 = ctx.read.materialize(path=tmpdir3).take_all()

            # Verify batching works (4 + 1 (mrr.reset at the end))
            assert counter.x == 5

            assert ids(e1) == ids(e2)

            # Use same context for second pipelines
            (
                ctx.read.materialize(path={"root": tmpdir3})
                .map(noop_fn)
                .map(noop_fn)
                .materialize(
                    path={"root": tmpdir4},
                )
                .execute()
            )
            e3 = ctx.read.materialize(path=tmpdir4).take_all()
            assert ids(e3) == ids(e2)

    def test_binary_file_read_reliability_list_of_paths_retries_successful(self):
        ctx = sycamore.init(exec_mode=self.exec_mode)

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
            tempfile.TemporaryDirectory() as tmpdir3,
        ):

            failure_counter = NumCalls()
            retry_counter = NumCalls()
            docs = make_docs(10)
            docs.pop()
            mrr = MaterializeReadReliability(max_batch=3)
            mrr = mock_mrr_reset_fn(mrr, retry_counter)

            ctx.rewrite_rules.append(mrr)
            paths = []
            for doc in docs:
                path = Path(tmpdir1) / f"{doc.doc_id}.{doc.properties.get('extension', 'pdf')}"
                path.write_bytes(b"test content")
                paths.append(str(path))

            def failing_map(doc):
                failure_counter.x += 1
                if failure_counter.x % 4 == 0:  # Fail batch with every 4th document
                    raise ValueError("Simulated failure")
                return doc

            ds = (
                ctx.read.binary(paths, binary_format="pdf")
                .map(failing_map)
                .materialize(
                    path=tmpdir2,
                )
            )
            ds.execute()
            assert retry_counter.x == 8  # 4 success +3 extra retries for 3 failures + 1 for mrr.reset()

            # Check with directory as well

            retry_counter.x = 0
            failure_counter.x = 0

            ds = ctx.read.binary(tmpdir1, binary_format="pdf").map(failing_map).materialize(path=tmpdir3)
            ds.execute()
            assert retry_counter.x == 8  # 4 success +3 extra retries for 3 failures + 1 for mrr.reset()

    def test_binary_file_read_reliability_list_of_paths_retries_failure(self):
        ctx = sycamore.init(exec_mode=self.exec_mode)

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
            tempfile.TemporaryDirectory() as tmpdir3,
        ):

            failure_counter = NumCalls()
            retry_counter = NumCalls()
            docs = make_docs(10)
            docs.pop()
            mrr = MaterializeReadReliability(max_batch=3)
            mrr = mock_mrr_reset_fn(mrr, retry_counter)

            ctx.rewrite_rules.append(mrr)
            paths = []
            for doc in docs:
                path = Path(tmpdir1) / f"{doc.doc_id}.{doc.properties.get('extension', 'pdf')}"
                path.write_bytes(b"test content")
                paths.append(str(path))
                # Create a function that fails for specific documents

            def failing_map(doc):
                failure_counter.x += 1
                if failure_counter.x >= 9:  # Perpetual fail after 9th document
                    raise ValueError("Simulated failure")
                return doc

            ds = (
                ctx.read.binary(paths, binary_format="pdf")
                .map(failing_map)
                .materialize(
                    path={"root": tmpdir2},
                )
            )
            ds.execute()

            assert retry_counter.x == 23  # 2 successful, 21 unsuccessful

            # Check with directory as well

            failure_counter.x = 0
            retry_counter.x = 0

            ds = (
                ctx.read.binary(tmpdir1, binary_format="pdf")
                .map(failing_map)
                .materialize(
                    path={"root": tmpdir3},
                )
            )
            ds.execute()

            assert retry_counter.x == 23  # 2 successful, 21 unsuccessful
