import pytest

from sycamore.connectors.weaviate.weaviate_writer import (
    CollectionConfigCreate,
    WeaviateClient,
    WeaviateClientParams,
    WeaviateCrossReferenceClient,
    WeaviateCrossReferenceRecord,
    WeaviateWriterDocumentRecord,
    WeaviateWriterTargetParams,
)
from sycamore.data.document import Document
import weaviate
from weaviate.classes.config import Property, ReferenceProperty
from weaviate.client import ConnectionParams
from weaviate.collections.classes.config import Configure, DataType
from weaviate.exceptions import WeaviateInvalidInputError


@pytest.fixture(scope="module")
def embedded_client():
    port = 8078
    grpc_port = 50059
    client = weaviate.WeaviateClient(
        embedded_options=weaviate.embedded.EmbeddedOptions(version="1.24.0", port=port, grpc_port=grpc_port)
    )
    yield client
    with client:
        client.collections.delete_all()


def collection_params_a(collection_name: str):
    return {
        "name": collection_name,
        "description": "A collection to demo data-prep with sycamore",
        "properties": [
            Property(
                name="properties",
                data_type=DataType.OBJECT,
                nested_properties=[
                    Property(
                        name="links",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(name="text", data_type=DataType.TEXT),
                            Property(name="url", data_type=DataType.TEXT),
                            Property(name="start_index", data_type=DataType.NUMBER),
                        ],
                    ),
                ],
            ),
            Property(name="bbox", data_type=DataType.NUMBER_ARRAY),
            Property(name="shingles", data_type=DataType.INT_ARRAY),
        ],
        "vectorizer_config": [Configure.NamedVectors.none(name="embedding")],
        "references": [ReferenceProperty(name="parent", target_collection=collection_name)],
    }


class TestWeaviateTargetParams:
    def test_target_params_compat_with_self(self):
        cn = "TestNumber1"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        wtp_b = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        assert wtp_a.compatible_with(wtp_b)
        assert wtp_b.compatible_with(wtp_a)

    def test_target_params_incompat_with_diff_flattens(self):
        cn = "TestNumber2"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        wtp_b = WeaviateWriterTargetParams(
            name=cn, collection_config=CollectionConfigCreate(**cp), flatten_properties=True
        )
        assert not wtp_a.compatible_with(wtp_b)
        assert not wtp_b.compatible_with(wtp_a)

    def test_target_params_compat_through_weaviate_object(self, embedded_client):
        cn = "TestNumber3"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        wcl = WeaviateClient(embedded_client)
        wcl.create_target_idempotent(wtp_a)
        wtp_b = wcl.get_existing_target_params(wtp_a)
        assert wtp_a.compatible_with(wtp_b)


class TestWeaviateClient:
    @staticmethod
    def mock_client(mocker):
        client = mocker.Mock(spec=weaviate.WeaviateClient)
        client.__enter__ = mocker.Mock()
        client.__enter__.return_value = client
        client.__exit__ = mocker.Mock()
        client.collections = mocker.Mock()
        return client

    @staticmethod
    def mock_batch(mocker, mock_client):
        mock_client.collections.get = mocker.Mock()
        collection = mocker.Mock(spec=weaviate.collections.Collection)
        mock_client.collections.get.return_value = collection
        collection.batch = mocker.Mock()
        collection.batch.dynamic = mocker.Mock()
        mock_batch = mocker.Mock()
        mock_batch.__enter__ = mocker.Mock()
        mock_batch.__enter__.return_value = mock_batch
        mock_batch.__exit__ = mocker.Mock()
        collection.batch.dynamic.return_value = mock_batch
        return mock_batch

    def test_create_target_normal(self, mocker):
        cn = "TestNumber4"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        wcl = TestWeaviateClient.mock_client(mocker)
        wcl.collections.create = mocker.Mock()
        wcl.collections.create.return_value = True
        client = WeaviateClient(wcl)
        client.create_target_idempotent(wtp_a)
        wcl.collections.create.assert_called_once()

    def test_create_target_from_target(self, mocker, embedded_client):
        cn = "TestNumber5"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        real_client = WeaviateClient(embedded_client)
        real_client.create_target_idempotent(wtp_a)
        wtp_b = real_client.get_existing_target_params(wtp_a)

        fake_inner_client = TestWeaviateClient.mock_client(mocker)
        fake_inner_client.collections.create_from_config = mocker.Mock()
        fake_inner_client.collections.create_from_config.return_value = True
        fake_client = WeaviateClient(fake_inner_client)
        fake_client.create_target_idempotent(wtp_b)
        fake_inner_client.collections.create_from_config.assert_called_once()

    def test_write_many_documents(self, mocker):
        docs = [
            WeaviateWriterDocumentRecord(uuid="1", properties={"field": "value"}, vector={"embedding": [0.2] * 4}),
            WeaviateWriterDocumentRecord(uuid="2", properties={"field": "othervalue"}, vector={"embedding": [0.1] * 4}),
            WeaviateWriterDocumentRecord(uuid="3", properties={"field": "no_vector"}),
        ]
        cn = "TestNumber6"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))

        wcl = TestWeaviateClient.mock_client(mocker)
        wbatch = TestWeaviateClient.mock_batch(mocker, wcl)
        wbatch.add_object = mocker.Mock()
        client = WeaviateClient(wcl)

        client.write_many_records(docs, wtp_a)
        assert wbatch.add_object.call_count == 3


class TestWeaviateDocumentRecord:
    def test_from_doc(self):
        doc = Document(
            {
                "doc_id": "id",
                "properties": {"field": "value", "nested": {"object": "value"}},
                "type": "text",
                "text_representation": "my first document",
            }
        )
        cn = "TestNumber7"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        record = WeaviateWriterDocumentRecord.from_doc(doc, wtp_a)
        assert record.uuid == "id"
        assert record.properties == {
            "properties": {"field": "value", "nested": {"object": "value"}},
            "type": "text",
            "text_representation": "my first document",
        }

    def test_from_doc_flattened(self):
        doc = Document(
            {
                "doc_id": "id",
                "properties": {"field": "value", "nested": {"object": "value"}},
                "type": "text",
                "text_representation": "my first document",
            }
        )
        cn = "TestNumber8"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(
            name=cn, collection_config=CollectionConfigCreate(**cp), flatten_properties=True
        )
        record = WeaviateWriterDocumentRecord.from_doc(doc, wtp_a)
        assert record.uuid == "id"
        assert record.properties == {
            "properties__field": "value",
            "properties__nested__object": "value",
            "type": "text",
            "text_representation": "my first document",
        }

    def test_from_doc_with_embedding(self):
        doc = Document({"doc_id": "id", "text_representation": "helloworld", "embedding": [0.4] * 19})
        cn = "TestNumber9"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        record = WeaviateWriterDocumentRecord.from_doc(doc, wtp_a)
        assert record.uuid == "id"
        assert record.properties == {"text_representation": "helloworld"}
        assert record.vector == {"embedding": [0.4] * 19}

    def test_from_doc_with_list_types(self):
        doc = Document(
            {
                "doc_id": "id",
                "text_representation": "my second document",
                "bbox": (0.1, 1.2, 2.3, 3.4),
                "shingles": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
            }
        )
        cn = "TestNumber10"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        record = WeaviateWriterDocumentRecord.from_doc(doc, wtp_a)
        assert record.uuid == "id"
        assert record.properties == {
            "text_representation": "my second document",
            "bbox": (0.1, 1.2, 2.3, 3.4),
            "shingles": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        }


class TestWeaviateCrossReferenceClient:
    @staticmethod
    def mock_collection_config(mocker, wcl):
        wcl.collections.get = mocker.Mock()
        mc = mocker.Mock(spec=weaviate.collections.Collection)
        wcl.collections.get.return_value = mc
        mc.config = mocker.Mock()
        return mc

    def test_inheritance_is_not_stoopid(self):
        wcrc = WeaviateCrossReferenceClient.from_client_params(
            WeaviateClientParams(
                connection_params=ConnectionParams.from_params(
                    http_host="localhost",
                    http_port=8080,
                    http_secure=False,
                    grpc_host="localhost",
                    grpc_port=50051,
                    grpc_secure=False,
                )
            )
        )
        assert isinstance(wcrc, WeaviateCrossReferenceClient)

    def test_create_target_idempotent_success(self, mocker):
        cn = "TestNumber11"
        cp = collection_params_a(cn)
        wtp = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))

        wcl = TestWeaviateClient.mock_client(mocker)
        mc = TestWeaviateCrossReferenceClient.mock_collection_config(mocker, wcl)
        mc.config.add_reference = mocker.Mock()

        wcrc = WeaviateCrossReferenceClient(wcl)
        wcrc.create_target_idempotent(wtp)
        mc.config.add_reference.assert_called_once()

    def test_create_target_idempotent_fails_expectedly(self, mocker):
        cn = "TestNumber12"
        cp = collection_params_a(cn)
        wtp = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))

        wcl = TestWeaviateClient.mock_client(mocker)
        mc = TestWeaviateCrossReferenceClient.mock_collection_config(mocker, wcl)
        mc.config.add_reference = mocker.Mock()
        mc.config.add_reference.side_effect = WeaviateInvalidInputError("parent prop already exists ya dummy")

        wcrc = WeaviateCrossReferenceClient(wcl)
        wcrc.create_target_idempotent(wtp)
        mc.config.add_reference.assert_called_once()

    def test_create_target_idempotent_fails_unexpectedly(self, mocker):
        cn = "TestNumber13"
        cp = collection_params_a(cn)
        wtp = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))

        wcl = TestWeaviateClient.mock_client(mocker)
        mc = TestWeaviateCrossReferenceClient.mock_collection_config(mocker, wcl)
        mc.config.add_reference = mocker.Mock()
        mc.config.add_reference.side_effect = ValueError("some other kind of error")

        # Not sure why, seems like the mock context manager swallows errors
        wcrc = WeaviateCrossReferenceClient(wcl)
        wcrc.create_target_idempotent(wtp)
        wcl.__exit__.assert_called_once()
        assert wcl.__exit__.call_args.args[0] == ValueError
        assert str(wcl.__exit__.call_args.args[1]) == "some other kind of error"

    def test_write_many_cross_references(self, mocker):
        docs = [
            WeaviateCrossReferenceRecord(from_uuid="1", from_property="parent", to="eozariel-lord-of-jumbo-shrimp"),
            WeaviateCrossReferenceRecord(from_uuid="2", from_property="parent", to="eozariel-lord-of-jumbo-shrimp"),
            WeaviateCrossReferenceRecord(from_uuid="3", from_property="parent", to="eozariel-lord-of-jumbo-shrimp"),
            WeaviateCrossReferenceRecord(from_uuid="eozariel-lord-of-jumbo-shrimp", from_property="parent", to=None),
        ]
        cn = "TestNumber14"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))

        wcl = TestWeaviateClient.mock_client(mocker)
        wbatch = TestWeaviateClient.mock_batch(mocker, wcl)
        wbatch.add_reference = mocker.Mock()
        client = WeaviateCrossReferenceClient(wcl)

        client.write_many_records(docs, wtp_a)
        assert wbatch.add_reference.call_count == 3


class TestWeaviateCrossReferenceRecord:
    def test_from_docs(self):
        docs = [
            Document({"doc_id": "1", "parent_id": "eozariel-lord-of-jumbo-shrimp"}),
            Document({"doc_id": "eozariel-lord-of-jumbo-shrimp"}),
        ]
        cn = "TestNumber15"
        cp = collection_params_a(cn)
        wtp_a = WeaviateWriterTargetParams(name=cn, collection_config=CollectionConfigCreate(**cp))
        cr_records = [WeaviateCrossReferenceRecord.from_doc(d, wtp_a) for d in docs]
        assert len(cr_records) == 2
        assert cr_records[0] == WeaviateCrossReferenceRecord(
            from_uuid="1", from_property="parent", to="eozariel-lord-of-jumbo-shrimp"
        )
        assert cr_records[1] == WeaviateCrossReferenceRecord(
            from_uuid="eozariel-lord-of-jumbo-shrimp", from_property="parent", to=None
        )
