from unittest.mock import Mock
import random

from opensearchpy import OpenSearch, RequestError, ConnectionError
import pytest

from sycamore.connectors.opensearch.opensearch_reader import OpenSearchReaderQueryResponse, OpenSearchReaderQueryParams

from sycamore import Context
from sycamore.connectors.opensearch import (
    OpenSearchWriterClient,
    OpenSearchWriterClientParams,
    OpenSearchWriterRecord,
    OpenSearchWriterTargetParams,
)
from sycamore.connectors.common import HostAndPort
from sycamore.connectors.opensearch.utils import get_knn_query
from sycamore.data.document import Document, DocumentPropertyTypes, DocumentSource
from sycamore.transforms import Embedder

MATCH_ALL_QUERY = {"query": {"match_all": {}}}


class TestOpenSearchTargetParams:

    def test_compat_equal_params(self):
        p1 = OpenSearchWriterTargetParams(index_name="test")
        p2 = OpenSearchWriterTargetParams(index_name="test")
        assert p1.compatible_with(p2)
        assert p2.compatible_with(p1)

    def test_compat_smaller_params(self):
        p1 = OpenSearchWriterTargetParams(
            index_name="test", settings={"key": "value"}, mappings={"otherkey": "othervalue"}
        )
        p2 = OpenSearchWriterTargetParams(
            index_name="test",
            settings={"key": "value", "setting": "2"},
            mappings={"otherkey": "othervalue", "fourthkey": "fourthvalue"},
        )
        assert p1.compatible_with(p2)
        assert not p2.compatible_with(p1)

    def test_compat_diff_index_names(self):
        p1 = OpenSearchWriterTargetParams(index_name="test")
        p2 = OpenSearchWriterTargetParams(index_name="nottest")
        assert not p1.compatible_with(p2)
        assert not p2.compatible_with(p1)

    def test_compat_nested_params(self):
        p1 = OpenSearchWriterTargetParams(
            index_name="test",
            settings={"key": {"nestedkey": "nestedvalue", "othernestedkey": "othernestedvalue"}},
            mappings={"otherkey": "othervalue"},
        )
        p2 = OpenSearchWriterTargetParams(
            index_name="test",
            settings={"key.nestedkey": "nestedvalue", "key": {"othernestedkey": "othernestedvalue"}},
            mappings={"otherkey": "othervalue"},
        )
        assert p1.compatible_with(p2)
        assert p2.compatible_with(p1)

    def test_compat_index_autonesting(self):
        p1 = OpenSearchWriterTargetParams(
            index_name="test", settings={"index": {"key": "value"}, "index.otherkey": "othervalue"}
        )
        p2 = OpenSearchWriterTargetParams(index_name="test", settings={"key": "value", "otherkey": "othervalue"})
        assert not p1.compatible_with(p2)
        assert p2.compatible_with(p1)


class TestOpenSearchClient:
    def test_create_target_request_error(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        client.indices = mocker.Mock()
        client.indices.create = mocker.Mock()
        client.indices.create.side_effect = RequestError(400, "Some Reason", {})

        params = OpenSearchWriterTargetParams(index_name="found")
        osc_testing = OpenSearchWriterClient(client)

        # Should not fail
        osc_testing.create_target_idempotent(params)

    def test_create_target_fails_for_some_other_reason(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        client.indices = mocker.Mock()
        client.indices.create = mocker.Mock()
        client.indices.create.side_effect = ConnectionError(400, "Not connected", {})

        params = OpenSearchWriterTargetParams(index_name="fail")
        osc_testing = OpenSearchWriterClient(client)

        with pytest.raises(ConnectionError) as einfo:
            osc_testing.create_target_idempotent(params)
        assert einfo.value.error == "Not connected"

    def test_get_target_awkward_field_types_n_stuff(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        client.indices = mocker.Mock()
        client.indices.get = mocker.Mock()
        client.indices.get.return_value = {
            "test": {
                "mappings": {
                    "bool_key": "true",
                    "other_bool_key": "false",
                    "int_key": "3",
                    "float_key": "3.14",
                    "string_key": "string",
                },
                "settings": {
                    "nested_dict": {
                        "nested_array": [
                            "true",
                            "false",
                            "3",
                            "3.14",
                            "valid_python_types_are_ok_too",
                            True,
                            False,
                            3,
                            3.14,
                        ]
                    }
                },
            }
        }
        p1 = OpenSearchWriterTargetParams(index_name="test")
        osc_testing = OpenSearchWriterClient(client)
        p2 = osc_testing.get_existing_target_params(p1)
        mappings = p2.mappings
        assert mappings["bool_key"] is True
        assert mappings["other_bool_key"] is False
        assert mappings["int_key"] == 3
        assert mappings["float_key"] == 3.14
        assert mappings["string_key"] == "string"

        settings = p2.settings
        assert "nested_dict" in settings
        assert "nested_array" in settings["nested_dict"]
        assert settings["nested_dict"]["nested_array"] == [
            True,
            False,
            3,
            3.14,
            "valid_python_types_are_ok_too",
            True,
            False,
            3,
            3.14,
        ]

    def test_create_client_from_params(self, mocker):
        ping = mocker.patch.object(OpenSearch, "ping")
        ping.return_value = True
        client_params = OpenSearchWriterClientParams(hosts=[HostAndPort(host="localhost", port=9200)])
        OpenSearchWriterClient.from_client_params(client_params)

    def test_write_many_documents(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        parallel_blk = mocker.patch("opensearchpy.helpers.parallel_bulk")
        parallel_blk.return_value = []
        records = [
            OpenSearchWriterRecord(_source={"field": 1}, _index="test", _id="1"),
            OpenSearchWriterRecord(_source={"field": 2}, _index="test", _id="2"),
        ]
        target_params = OpenSearchWriterTargetParams(index_name="test")
        osc_testing = OpenSearchWriterClient(client)
        osc_testing.write_many_records(records, target_params)


class TestOpenSearchReaderQueryResponse:
    def test_to_docs(self):
        records = [
            {
                "text_representation": "this is an element",
                "parent_id": "doc_1",
            },
            {"text_representation": "this is a parent doc", "parent_id": None, "doc_id": "doc_1"},
        ]
        hits = [{"_source": record, "_score": random.random()} for record in records]
        query_response = OpenSearchReaderQueryResponse(hits)
        query_params = OpenSearchReaderQueryParams(index_name="some index", query=MATCH_ALL_QUERY)
        docs = query_response.to_docs(query_params)

        assert len(docs) == 2

        for i in range(len(docs)):
            assert docs[i].parent_id == records[i]["parent_id"]
            assert docs[i].text_representation == records[i]["text_representation"]
            assert "score" in docs[i].properties

    def test_to_docs_reconstruct_require_client(self):
        query_response = OpenSearchReaderQueryResponse([])
        query_params = OpenSearchReaderQueryParams(
            index_name="some index", query=MATCH_ALL_QUERY, reconstruct_document=True
        )
        with pytest.raises(AssertionError):
            query_response.to_docs(query_params)

    def test_to_docs_reconstruct(self, mocker):
        records = [
            {
                "text_representation": "this is an element belonging to parent doc 1",
                "parent_id": "doc_1",
                "doc_id": "element_1",
            },
            {
                "text_representation": "this is an element belonging to parent doc 1",
                "parent_id": "doc_1",
                "doc_id": "element_2",
            },
            {
                "text_representation": "this is an element belonging to parent doc 2",
                "parent_id": "doc_2",
                "doc_id": "element_1",
            },
            {
                "text_representation": "the parent doc of this element was not part of the result set",
                "parent_id": "doc_4",
                "doc_id": "element_1",
            },
            {"text_representation": "this is a parent doc 1", "parent_id": None, "doc_id": "doc_1"},
            {"text_representation": "this is a parent doc 2", "parent_id": None, "doc_id": "doc_2"},
            {"text_representation": "this is a parent doc 3", "parent_id": None, "doc_id": "doc_3"},
        ]
        client = mocker.Mock(spec=OpenSearch)

        # no elements match
        hits = [{"_source": record} for record in records]
        return_val = {"hits": {"hits": [hit for hit in hits if hit["_source"].get("parent_id")]}}
        return_val["hits"]["hits"] += [
            {
                "_source": {
                    "text_representation": "this is an element belonging to parent doc 2 retrieved via reconstruction",
                    "parent_id": "doc_2",
                    "doc_id": "element_2",
                }
            },
            {
                "_source": {
                    "text_representation": "this is an element belonging to parent doc 2 retrieved via reconstruction",
                    "parent_id": "doc_2",
                    "doc_id": "element_3",
                }
            },
        ]
        client.search.return_value = return_val
        query_response = OpenSearchReaderQueryResponse(hits, client=client)
        query_params = OpenSearchReaderQueryParams(
            index_name="some index", query=MATCH_ALL_QUERY, reconstruct_document=True
        )
        docs = query_response.to_docs(query_params)

        assert len(docs) == 4

        # since docs are unordered
        doc_1 = [doc for doc in docs if doc.doc_id == "doc_1"][0]
        doc_2 = [doc for doc in docs if doc.doc_id == "doc_2"][0]
        doc_3 = [doc for doc in docs if doc.doc_id == "doc_3"][0]
        doc_4 = [doc for doc in docs if doc.doc_id == "doc_4"][0]

        assert len(doc_1.elements) == 2
        assert len(doc_2.elements) == 3
        assert len(doc_3.elements) == 0
        assert len(doc_4.elements) == 1

        assert doc_1.elements[0].text_representation == "this is an element belonging to parent doc 1"
        assert doc_1.elements[1].text_representation == "this is an element belonging to parent doc 1"
        assert doc_1.elements[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY
        assert doc_1.elements[1].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY

        assert doc_2.elements[0].text_representation == "this is an element belonging to parent doc 2"
        assert (
            doc_2.elements[1].text_representation
            == "this is an element belonging to parent doc 2 retrieved via reconstruction"
        )
        assert (
            doc_2.elements[2].text_representation
            == "this is an element belonging to parent doc 2 retrieved via reconstruction"
        )
        assert doc_2.elements[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY
        assert (
            doc_2.elements[1].properties[DocumentPropertyTypes.SOURCE]
            == DocumentSource.DOCUMENT_RECONSTRUCTION_RETRIEVAL
        )
        assert (
            doc_2.elements[2].properties[DocumentPropertyTypes.SOURCE]
            == DocumentSource.DOCUMENT_RECONSTRUCTION_RETRIEVAL
        )
        assert doc_4.elements[0].text_representation == "the parent doc of this element was not part of the result set"
        assert doc_4.properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DOCUMENT_RECONSTRUCTION_PARENT
        assert doc_4.elements[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY

    def test_to_docs_reconstruct_no_additional_elements(self, mocker):
        records = [
            {
                "text_representation": "this is an element belonging to parent doc 1",
                "parent_id": "doc_1",
                "doc_id": "element_1",
            },
            {
                "text_representation": "this is an element belonging to parent doc 1",
                "parent_id": "doc_1",
                "doc_id": "element_2",
            },
            {
                "text_representation": "this is an element belonging to parent doc 2",
                "parent_id": "doc_2",
                "doc_id": "element_1",
            },
            {"text_representation": "this is a parent doc 1", "parent_id": None, "doc_id": "doc_1"},
            {"text_representation": "this is a parent doc 2", "parent_id": None, "doc_id": "doc_2"},
            {"text_representation": "this is a parent doc 3", "parent_id": None, "doc_id": "doc_3"},
        ]
        client = mocker.Mock(spec=OpenSearch)

        # no elements match
        hits = [{"_source": record} for record in records]
        client.search.return_value = {"hits": {"hits": [hit for hit in hits if hit["_source"].get("parent_id")]}}
        query_response = OpenSearchReaderQueryResponse(hits, client=client)
        query_params = OpenSearchReaderQueryParams(
            index_name="some index", query=MATCH_ALL_QUERY, reconstruct_document=True
        )
        docs = query_response.to_docs(query_params)

        assert len(docs) == 3

        # since docs are unordered
        doc_1 = [doc for doc in docs if doc.doc_id == "doc_1"][0]
        doc_2 = [doc for doc in docs if doc.doc_id == "doc_2"][0]
        doc_3 = [doc for doc in docs if doc.doc_id == "doc_3"][0]

        assert len(doc_1.elements) == 2
        assert len(doc_2.elements) == 1
        assert len(doc_3.elements) == 0

        assert doc_1.elements[0].text_representation == "this is an element belonging to parent doc 1"
        assert doc_1.elements[1].text_representation == "this is an element belonging to parent doc 1"

        assert doc_2.elements[0].text_representation == "this is an element belonging to parent doc 2"


class TestOpenSearchRecord:
    def test_from_document_only_text(self):
        tp = OpenSearchWriterTargetParams(index_name="test")
        document = Document({"text_representation": "text", "doc_id": "id"})
        record = OpenSearchWriterRecord.from_doc(document, tp)
        assert record._source == {
            "doc_id": "id",
            "type": None,
            "text_representation": "text",
            "elements": [],
            "embedding": None,
            "parent_id": None,
            "properties": {},
            "bbox": None,
            "shingles": None,
        }
        assert record._id == document.doc_id
        assert record._index == tp.index_name

    def test_from_document_all_fields(self):
        tp = OpenSearchWriterTargetParams(index_name="test")
        data = {
            "text_representation": "text_representation",
            "type": "text",
            "embedding": [0.2] * 384,
            "properties": {"some": "field"},
            "elements": [],
            "parent_id": None,
            "bbox": (1, 2, 3, 4),
            "shingles": [1, 2, 3, 4],
            "doc_id": "id",
        }
        document = Document(data)
        record = OpenSearchWriterRecord.from_doc(document, tp)
        assert record._source == data
        assert record._id == document.doc_id
        assert record._index == tp.index_name

    def test_from_document_too_many_fields(self):
        tp = OpenSearchWriterTargetParams(index_name="test")
        data = {
            "text_representation": "text_representation",
            "type": "text",
            "embedding": [0.2] * 384,
            "properties": {"some": "field"},
            "bbox": (1, 2, 3, 4),
            "shingles": [1, 2, 3, 4],
            "doc_id": "id",
            "another_field": "something",
        }
        document = Document(data)
        record = OpenSearchWriterRecord.from_doc(document, tp)
        assert record._source == {
            "text_representation": "text_representation",
            "type": "text",
            "embedding": [0.2] * 384,
            "properties": {"some": "field"},
            "elements": [],
            "parent_id": None,
            "bbox": (1, 2, 3, 4),
            "shingles": [1, 2, 3, 4],
            "doc_id": "id",
        }
        assert record._id == document.doc_id
        assert record._index == tp.index_name


class TestOpenSearchUtils:

    def test_get_knn_query(self):
        embedder = Mock(spec=Embedder)
        embedding = [0.1, 0.2]
        embedder.generate_text_embedding.return_value = embedding
        context = Context(
            params={
                "opensearch": {
                    "os_client_args": {
                        "hosts": [{"host": "localhost", "port": 9200}],
                        "http_compress": True,
                        "http_auth": ("admin", "admin"),
                        "use_ssl": True,
                        "verify_certs": False,
                        "ssl_assert_hostname": False,
                        "ssl_show_warn": False,
                        "timeout": 120,
                    },
                    "index_name": "test_index",
                },
                "default": {"text_embedder": embedder},
            }
        )
        expected_query = {"query": {"knn": {"embedding": {"vector": embedding, "k": 1000}}}}
        assert get_knn_query(query_phrase="test", k=1000, context=context) == expected_query
        embedder.generate_text_embedding.assert_called_with("test")

        assert get_knn_query(query_phrase="test", k=1000, text_embedder=embedder) == expected_query
        embedder.generate_text_embedding.assert_called_with("test")

        # default
        expected_query = {"query": {"knn": {"embedding": {"vector": embedding, "k": 500}}}}
        assert get_knn_query(query_phrase="test", context=context) == expected_query
        embedder.generate_text_embedding.assert_called_with("test")

        # min_score
        expected_query = {"query": {"knn": {"embedding": {"vector": embedding, "min_score": 0.5}}}}
        assert get_knn_query(query_phrase="test", min_score=0.5, context=context) == expected_query
        embedder.generate_text_embedding.assert_called_with("test")

    def test_get_knn_query_validation(self):
        with pytest.raises(ValueError, match="Only one of `k` or `min_score` should be populated"):
            get_knn_query(Mock(spec=Embedder), query_phrase="test", k=10, min_score=0.5)
