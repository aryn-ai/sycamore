from opensearchpy import OpenSearch, RequestError
import pytest
from sycamore.connectors.opensearch import (
    OpenSearchWriterClient,
    OpenSearchWriterClientParams,
    OpenSearchWriterRecord,
    OpenSearchWriterTargetParams,
)
from sycamore.connectors.common import HostAndPort
from sycamore.data.document import Document


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
    def test_create_target_already_exists(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        client.indices = mocker.Mock()
        client.indices.create = mocker.Mock()
        client.indices.create.side_effect = RequestError(400, "resource_already_exists_exception", {})

        params = OpenSearchWriterTargetParams(index_name="found")
        osc_testing = OpenSearchWriterClient(client)

        # Should not fail
        osc_testing.create_target_idempotent(params)

    def test_create_target_fails_for_some_other_reason(self, mocker):
        client = mocker.Mock(spec=OpenSearch)
        client.indices = mocker.Mock()
        client.indices.create = mocker.Mock()
        client.indices.create.side_effect = RequestError(400, "could_not_create_index_for_some_other_reason", {})

        params = OpenSearchWriterTargetParams(index_name="fail")
        osc_testing = OpenSearchWriterClient(client)

        with pytest.raises(RequestError) as einfo:
            osc_testing.create_target_idempotent(params)
        assert einfo.value.error == "could_not_create_index_for_some_other_reason"

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
