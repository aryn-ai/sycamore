import os
import tempfile
import time
import uuid
from typing import Optional, Dict, Any

import pytest

import sycamore
from sycamore import EXEC_LOCAL, ExecMode
from sycamore.connectors.doc_reconstruct import DocumentReconstructor
from sycamore.data import Document
from sycamore.data.document import DocumentPropertyTypes
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.tests.integration.connectors.common import compare_connector_docs
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.partition import ArynPartitioner

from sycamore.transforms.extract_entity import OpenAIEntityExtractor

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")
TEST_CACHE_DIR = "/tmp/test_cache_dir"
ARYN_API_KEY = os.getenv("ARYN_API_KEY")


@pytest.fixture(scope="class")
def os_client():
    from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging

    client = OpenSearchClientWithLogging(**TestOpenSearchRead.OS_CLIENT_ARGS)
    yield client


@pytest.fixture(scope="class")
def setup_index(os_client):
    # Recreate before
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)
    os_client.indices.create(TestOpenSearchRead.INDEX, **TestOpenSearchRead.INDEX_SETTINGS)

    yield TestOpenSearchRead.INDEX

    # Delete after
    os_client.indices.delete(TestOpenSearchRead.INDEX, ignore_unavailable=True)


@pytest.fixture(scope="class")
def setup_index_large(os_client):
    index_name = "test_opensearch_read_2"
    os_client.indices.delete(index_name, ignore_unavailable=True)
    os_client.indices.create(index_name, **TestOpenSearchRead.INDEX_SETTINGS)

    path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
    context = sycamore.init(exec_mode=ExecMode.RAY)

    (
        context.read.binary(path, binary_format="pdf")
        .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
        .explode()
        .write.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=index_name,
            index_settings=TestOpenSearchRead.INDEX_SETTINGS,
            execute=False,
        )
        .take_all()
    )

    os_client.indices.refresh(index_name)

    yield index_name

    # Delete after
    os_client.indices.delete(index_name, ignore_unavailable=True)


def get_doc_count(os_client, index_name: str, query: Optional[Dict[str, Any]] = None) -> int:
    res = os_client.count(index=index_name)
    return res["count"]


class TestOpenSearchRead:
    INDEX_SETTINGS = {
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
                }
            },
        }
    }

    INDEX = "test_opensearch_read"

    OS_CLIENT_ARGS = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", OS_ADMIN_PASSWORD),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    def test_ingest_and_read(self, setup_index, os_client, exec_mode):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=exec_mode)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=setup_index
        )
        target_doc_id = original_docs[-1].doc_id if original_docs[-1].doc_id else ""
        query = {"query": {"term": {"_id": target_doc_id}}}
        query_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS, index_name=setup_index, query=query
        )

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            reconstruct_document=True,
        )
        original_materialized = sorted(original_docs, key=lambda d: d.doc_id)

        retrieved_materialized = sorted(retrieved_docs.take_all(), key=lambda d: d.doc_id)
        query_materialized = query_docs.take_all()
        retrieved_materialized_reconstructed = sorted(retrieved_docs_reconstructed.take_all(), key=lambda d: d.doc_id)

        assert len(query_materialized) == 1  # exactly one doc should be returned
        compare_connector_docs(original_materialized, retrieved_materialized)

        assert len(retrieved_materialized_reconstructed) == 1
        doc = retrieved_materialized_reconstructed[0]
        assert len(doc.elements) == len(retrieved_materialized) - 1  # drop the document parent record

        for i in range(len(doc.elements) - 1):
            assert doc.elements[i].element_index < doc.elements[i + 1].element_index

        os_client.indices.delete(setup_index, ignore_unavailable=True)

    def test_doc_reconstruct(self, setup_index, os_client):
        with tempfile.TemporaryDirectory() as materialized_dir:
            self._test_doc_reconstruct(setup_index, os_client, materialized_dir)

    def _test_doc_reconstruct(self, setup_index, os_client, materialized_dir):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        print(f"Using materialized dir: {materialized_dir}")

        def doc_reconstructor(doc_id: str) -> Document:
            import pickle

            data = pickle.load(open(f"{materialized_dir}/{setup_index}-{doc_id}", "rb"))
            return Document(**data)

        def doc_to_name(doc: Document, bin: bytes) -> str:
            return f"{setup_index}-{doc.doc_id}"

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        llm = OpenAI(OpenAIModels.GPT_4O_MINI)
        extractor = OpenAIEntityExtractor("title", llm=llm)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
            .extract_entity(extractor)
            .materialize(path={"root": materialized_dir, "name": doc_to_name})
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            reconstruct_document=True,
        ).take_all()

        assert len(retrieved_docs_reconstructed) == 1
        retrieved_sorted = sorted(retrieved_docs_reconstructed, key=lambda d: d.doc_id)

        def remove_reconstruct_doc_property(doc: Document):
            for element in doc.data["elements"]:
                element["properties"].pop(DocumentPropertyTypes.SOURCE)

        for doc in retrieved_sorted:
            remove_reconstruct_doc_property(doc)

        from_materialized = [doc_reconstructor(doc.doc_id) for doc in retrieved_sorted]
        compare_connector_docs(from_materialized, retrieved_sorted)

        # Clean up
        os_client.indices.delete(setup_index, ignore_unavailable=True)

    def test_write_with_reliability(self, setup_index, os_client, exec_mode):
        """
        Validates that when materialized pickle outputs are deleted, the index is rewritten
        with the correct (reduced) number of chunks.
        """
        with tempfile.TemporaryDirectory() as tmpdir1:
            path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
            context = sycamore.init(exec_mode=exec_mode)

            # 2 docs for ray execution
            (
                context.read.binary([path, path], binary_format="pdf")
                .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
                .explode()
                .materialize(path=tmpdir1)
                .execute()
            )

            (
                context.read.materialize(tmpdir1).write.opensearch(
                    os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                    index_name=setup_index,
                    index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                    reliability_rewriter=True,
                )
            )
            os_client.indices.refresh(setup_index)
            count = get_doc_count(os_client, setup_index)

            # Delete 1 pickle file to make sure reliability rewriter works
            pickle_files = [f for f in os.listdir(tmpdir1) if f.endswith(".pickle")]
            assert pickle_files, "No pickle files found in materialized directory"
            os.remove(os.path.join(tmpdir1, pickle_files[0]))

            # Delete and recreate the index - should have fewer chunks
            (
                context.read.materialize(tmpdir1).write.opensearch(
                    os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                    index_name=setup_index,
                    index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                    reliability_rewriter=True,
                )
            )
            os_client.indices.refresh(setup_index)
            re_count = get_doc_count(os_client, setup_index)

        # Verify document count is reduced
        assert count - 1 == re_count, f"Expected {count} documents, found {re_count}"
        os_client.indices.delete(setup_index)

    def _test_ingest_and_read_via_docid_reconstructor(self, setup_index, os_client, cache_dir):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        print(f"Using cache dir: {cache_dir}")

        def doc_reconstructor(index_name: str, doc_id: str) -> Document:
            import pickle

            data = pickle.load(open(f"{cache_dir}/{setup_index}-{doc_id}", "rb"))
            return Document(**data)

        def doc_to_name(doc: Document, bin: bytes) -> str:
            return f"{setup_index}-{doc.doc_id}"

        context = sycamore.init(exec_mode=EXEC_LOCAL)
        hidden = str(uuid.uuid4())
        # make sure we read from pickle files -- this part won't be written into opensearch.
        dicts = [
            {
                "doc_id": "1",
                "hidden": hidden,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "2",
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": "3",
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {
                "doc_id": "4",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": "5",
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {
                "doc_id": "6",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        original_docs = (
            context.read.document(docs)
            .materialize(path={"root": cache_dir, "name": doc_to_name})
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            doc_reconstructor=DocumentReconstructor(index_name=setup_index, reconstruct_fn=doc_reconstructor),
        ).take_all()

        assert len(retrieved_docs_reconstructed) == 6
        retrieved_sorted = sorted(retrieved_docs_reconstructed, key=lambda d: d.doc_id)
        assert retrieved_sorted[0].data["hidden"] == hidden
        assert docs == retrieved_sorted

        # Clean slate between Execution Modes
        os_client.indices.delete(setup_index)
        os_client.indices.create(setup_index, **TestOpenSearchRead.INDEX_SETTINGS)
        os_client.indices.refresh(setup_index)

    def test_ingest_and_read_via_docid_reconstructor(self, setup_index, os_client):
        with tempfile.TemporaryDirectory() as cache_dir:
            self._test_ingest_and_read_via_docid_reconstructor(setup_index, os_client, cache_dir)

    def test_slice_and_shards(self, setup_index_large, os_client):
        """
        Validates data is readable from OpenSearch, and that we can rebuild processed Sycamore documents.
        """

        """
        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=TestOpenSearchRead.INDEX,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(TestOpenSearchRead.INDEX)

        expected_count = len(original_docs)
        """
        # actual_count = get_doc_count(os_client, setup_index_large)
        # print(f"Expected {expected_count} documents, found {actual_count}")

        # refresh should have made all ingested docs immediately available for search
        # assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        pit = os_client.create_pit(index=setup_index_large, keep_alive="100m")
        num_slices = 20
        search_body = {
            "query": {
                "match_all": {},
            },
            # "size": 100,
            "pit": {"id": pit["pit_id"], "keep_alive": "100m"},
            "slice": {"id": 0, "max": num_slices},
        }

        ids = set()
        page_size = 1000
        for i in range(num_slices):
            slice_count = 0
            for j in range(10):
                search_body["slice"]["id"] = i
                res = os_client.search(body=search_body, size=page_size, from_=j * page_size)
                print(f"{j}: {res['hits']['total']['value']}")
                hits = res["hits"]["hits"]
                if hits is None:
                    print(f"None: {j}")
                else:
                    print(f"Length: {len(hits)}")
                for hit in hits:
                    doc_id = hit["_source"]["doc_id"]
                    if doc_id in ids:
                        print(f"Duplicate doc_id: {doc_id}")
                    else:
                        ids.add(doc_id)
                        slice_count += 1
            print(f"Slice {i} count: {slice_count}")

        print(len(ids))

    @staticmethod
    def doc_to_name(doc: Document, bin: bytes) -> str:
        return f"{TestOpenSearchRead.INDEX}-{doc.doc_id}"

    def test_parallel_read_reconstruct(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        t0 = time.time()
        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=True,
            use_pit=False,
        ).take_all()
        t1 = time.time()
        print(f"Retrieved {len(retrieved_docs_reconstructed)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index_large, parents_only=True)
        assert len(retrieved_docs_reconstructed) == len(expected_docs)
        for doc in retrieved_docs_reconstructed:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)

        t0 = time.time()
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=False,
            use_pit=False,
        ).take_all()
        t1 = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index_large)
        assert len(retrieved_docs) == len(expected_docs)
        for doc in retrieved_docs:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read_reconstruct_with_pit(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        t0 = time.time()
        retrieved_docs_reconstructed = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=True,
        ).take_all()
        t1 = time.time()
        print(f"Retrieved {len(retrieved_docs_reconstructed)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index_large, parents_only=True)
        assert len(retrieved_docs_reconstructed) == len(expected_docs)
        for doc in retrieved_docs_reconstructed:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_read_with_pit(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)

        t0 = time.time()
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            reconstruct_document=False,
            concurrency=2,
        ).take_all()
        t1 = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index_large)
        assert len(retrieved_docs) == len(expected_docs)
        for doc in retrieved_docs:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_query_with_pit(self, setup_index_large, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)

        query = {"query": {"match": {"text_representation": "ray"}}}

        t0 = time.time()
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index_large,
            query=query,
            reconstruct_document=False,
            concurrency=2,
        ).take_all()
        t1 = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index_large, False, query)
        assert len(retrieved_docs) == len(expected_docs)
        for doc in retrieved_docs:
            assert doc.doc_id in expected_docs
            if doc.parent_id is not None:
                assert doc.parent_id == expected_docs[doc.doc_id]["parent_id"]

    def test_parallel_query_on_property_with_pit(self, setup_index, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        key = "property1"
        hidden = str(uuid.uuid4())
        query = {"query": {"match": {f"properties.{key}": hidden}}}
        # make sure we read from pickle files -- this part won't be written into opensearch.
        dicts = [
            {
                "doc_id": "1",
                "properties": {key: hidden},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "2",
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": "3",
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {
                "doc_id": "4",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": "5",
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {
                "doc_id": "6",
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        original_docs = (
            context.read.document(docs)
            # .materialize(path={"root": cache_dir, "name": doc_to_name})
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        t0 = time.time()
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
        ).take_all()
        t1 = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index, True, query)
        assert len(retrieved_docs) == len(expected_docs)
        assert "1" == retrieved_docs[0].doc_id
        assert hidden == retrieved_docs[0].properties[key]

        os_client.indices.delete(setup_index, ignore_unavailable=True)

    def test_parallel_query_on_extracted_property_with_pit(self, setup_index, os_client):

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        llm = OpenAI(OpenAIModels.GPT_4O_MINI)
        extractor = OpenAIEntityExtractor("title", llm=llm)
        original_docs = (
            context.read.binary(path, binary_format="pdf")
            .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
            .extract_entity(extractor)
            # .materialize(path={"root": materialized_dir, "name": doc_to_name})
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        # refresh should have made all ingested docs immediately available for search
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        query = {"query": {"match": {"properties.title": "ray"}}}

        t0 = time.time()
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
        ).take_all()
        t1 = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {t1 - t0} seconds")
        expected_docs = self.get_ids(os_client, setup_index, True, query)
        assert len(retrieved_docs) == len(expected_docs)

        os_client.indices.delete(setup_index, ignore_unavailable=True)

    def test_result_filter_on_property(self, setup_index, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        query = {"query": {"match_all": {}}}
        dicts = [
            {
                "doc_id": "p1",
                "properties": {"tags": ["1", "2", "3"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p2",
                "properties": {"tags": ["1", "2", "5"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p3",
                "properties": {"tags": ["1", "6", "7"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        original_docs = (
            context.read.document(docs)
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        filter = {"properties.tags": ["1"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p1", "p2", "p3"}
        assert expected == {d.doc_id for d in retrieved_docs}

        filter = {"properties.tags": ["2"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p1", "p2"}
        assert expected == {d.doc_id for d in retrieved_docs}

        filter = {"properties.tags": ["7"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p3"}
        assert expected == {d.doc_id for d in retrieved_docs}

        filter = {"properties.tags": ["8"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        assert 0 == len(retrieved_docs)

        os_client.indices.delete(setup_index, ignore_unavailable=True)

    def test_compound_query_on_property(self, setup_index, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        dicts = [
            {
                "doc_id": "p1",
                "properties": {"tags": ["1", "2", "3"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p2",
                "properties": {"tags": ["1", "2", "5"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p3",
                "properties": {"tags": ["1", "6", "7"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        original_docs = (
            context.read.document(docs)
            .explode()
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        compound_query = {
            "bool": {
                "must": [
                    {
                        "term": {
                            "properties.tags.keyword": {
                                "value": "1",
                            }
                        },
                    },
                    {
                        "term": {
                            "properties.tags.keyword": {
                                "value": "7",
                            }
                        },
                    },
                ],
            }
        }
        filter = {"properties.tags": ["7"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query={"query": compound_query},
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p3"}
        assert expected == {d.doc_id for d in retrieved_docs}

        compound_query_with_filter = {
            "bool": {
                "must": [
                    {
                        "term": {
                            "properties.tags.keyword": {
                                "value": "1",
                            }
                        },
                    },
                    {
                        "term": {
                            "properties.tags.keyword": {
                                "value": "7",
                            }
                        },
                    },
                ],
                "filter": [{"terms": filter}],
            }
        }
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query={"query": compound_query_with_filter},
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p3"}
        assert expected == {d.doc_id for d in retrieved_docs}

    def test_result_filter_on_property_knn(self, setup_index, os_client):
        context = sycamore.init(exec_mode=ExecMode.RAY)
        query = {"query": {"match_all": {}}}
        dicts = [
            {
                "doc_id": "p1",
                "properties": {"tags": ["1", "2", "3"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p2",
                "properties": {"tags": ["1", "2", "5"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": "p3",
                "properties": {"tags": ["1", "6", "7"]},
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = SentenceTransformerEmbedder(batch_size=10, model_name=model_name)

        original_docs = (
            context.read.document(docs)
            .spread_properties(["tags"])
            .explode()
            .embed(embedder)
            .write.opensearch(
                os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                index_name=setup_index,
                index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                execute=False,
            )
            .take_all()
        )

        os_client.indices.refresh(setup_index)

        expected_count = len(original_docs)
        actual_count = get_doc_count(os_client, setup_index)
        assert actual_count == expected_count, f"Expected {expected_count} documents, found {actual_count}"

        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=query,
            reconstruct_document=True,
        ).take_all()

        vector = retrieved_docs[0].elements[0].embedding

        knn_query = {
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": 100,
                    }
                }
            }
        }

        # Without the filter, the vector above can be used to retrieve all documents
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=knn_query,
            reconstruct_document=True,
        ).take_all()

        assert len(retrieved_docs) == 3

        filter = {"properties.tags": ["2"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=knn_query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p1", "p2"}
        assert expected == {d.doc_id for d in retrieved_docs}

        filter = {"properties.tags": ["7"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=knn_query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        expected = {"p3"}
        assert expected == {d.doc_id for d in retrieved_docs}

        filter = {"properties.tags": ["8"]}
        retrieved_docs = context.read.opensearch(
            os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
            index_name=setup_index,
            query=knn_query,
            reconstruct_document=True,
            result_filter=filter,
        ).take_all()

        assert 0 == len(retrieved_docs)

        os_client.indices.delete(setup_index, ignore_unavailable=True)

    @staticmethod
    def get_ids(
        os_client, index_name, parents_only: bool = False, query: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        if query is None:
            query = {"query": {"match_all": {}}}

        query_params = {"_source_includes": ["doc_id", "parent_id"], "scroll": "10m"}
        response = os_client.search(index=index_name, body=query, **query_params)
        scroll_id = response["_scroll_id"]
        # no_parent_ids = 0
        parents = set()
        no_parents = set()
        all_hits = 0
        docs = {}
        try:
            while True:
                hits = response["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    all_hits += 1
                    # result.append(hit)
                    if "parent_id" in hit["_source"] and hit["_source"]["parent_id"] is not None:
                        parents.add(hit["_source"]["parent_id"])
                        # print(f"Parent id: {hit['_source']['parent_id']}")
                    else:
                        no_parents.add(hit["_id"])
                        # print(f"No parent id: {hit['_id']}")

                    if parents_only:
                        if "parent_id" not in hit["_source"] or hit["_source"]["parent_id"] is None:
                            docs[hit["_id"]] = {
                                "doc_id": hit["_source"]["doc_id"],
                                "parent_id": hit["_source"]["parent_id"],
                            }
                    else:
                        docs[hit["_id"]] = {
                            "doc_id": hit["_source"]["doc_id"],
                            "parent_id": hit["_source"].get("parent_id"),
                        }

                response = os_client.scroll(scroll_id=scroll_id, scroll=query_params["scroll"])
        finally:
            os_client.clear_scroll(scroll_id=scroll_id)
        # print(f"Parents: {len(parents)}, no parents: {len(no_parents)}, all hits: {all_hits}")
        # print(parents)
        # print("-------------------------------")
        # print(no_parents)

        return docs

    def _test_pagination(self, setup_index_large, os_client):
        res = os_client.create_pit(index=setup_index_large, keep_alive="10m")
        pit_id = res["pit_id"]
        bodies = []
        query = {"query": {"match_all": {}}}
        num_slices = 5
        for i in range(num_slices):
            _query = {
                "slice": {
                    "id": i,
                    "max": num_slices,
                },
                "pit": {
                    "id": pit_id,
                    "keep_alive": "1m",
                },
                "sort": [{"_seq_no": "asc"}],
            }
            if "query" in query:
                _query["query"] = query["query"]
            bodies.append(_query)

        def search_slice(body, os_client) -> list[dict]:
            hits = []
            page = 0
            page_size = 10
            while True:
                # res = os_client.search(body=body, size=page_size, from_=page * page_size)
                res = os_client.search(body=body, size=page_size)
                _hits = res["hits"]["hits"]
                # if len(res["hits"]["hits"]) < page_size:
                if _hits is None or len(_hits) == 0:
                    break
                hits.extend(_hits)
                page += 1
                body["search_after"] = _hits[-1]["sort"]

            print(f"Slice hits: {len(hits)}")
            return hits

        all_hits = []
        for body in bodies:
            all_hits.extend(search_slice(body, os_client))

        print(f"Total hits: {len(all_hits)}")
        expected_docs = self.get_ids(os_client, setup_index_large)
        assert len(all_hits) == len(expected_docs)

    def _test_bulk_load(self, setup_index_large, os_client):

        # Only run this to populate a test index.
        return

        if not os_client.indices.exists(setup_index_large):
            os_client.indices.create(setup_index_large, **TestOpenSearchRead.INDEX_SETTINGS)
        """
        Used for generating a large number of documents in OpenSearch for testing purposes.
        """
        os_client.indices.refresh(setup_index_large)
        doc_count = get_doc_count(os_client, setup_index_large)
        print(f"Current count: {doc_count}")

        path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
        context = sycamore.init(exec_mode=ExecMode.RAY)
        while doc_count < 20000:
            (
                context.read.binary(path, binary_format="pdf")
                .partition(ArynPartitioner(aryn_api_key=ARYN_API_KEY))
                # .materialize(path={"root": TEST_CACHE_DIR, "name": self.doc_to_name})
                .explode()
                .write.opensearch(
                    os_client_args=TestOpenSearchRead.OS_CLIENT_ARGS,
                    index_name=setup_index_large,
                    index_settings=TestOpenSearchRead.INDEX_SETTINGS,
                    execute=False,
                )
                .take_all()
            )

            os_client.indices.refresh(setup_index_large)

            # expected_count = len(original_docs)
            doc_count = get_doc_count(os_client, setup_index_large)
            print(f"Current count: {doc_count}")

        print(f"Current count: {doc_count}")

    def _test_cat(self, setup_index_large, os_client):
        response = os_client.cat.shards(index=setup_index_large, format="json")
        print(response)
        doc_count = 0
        for item in response:
            if item["prirep"] == "p":
                print(item)
                doc_count += int(item["docs"])
        print(f"Total docs: {doc_count}")
