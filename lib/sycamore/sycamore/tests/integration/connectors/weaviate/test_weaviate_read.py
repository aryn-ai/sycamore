import pytest

from sycamore.tests.integration.connectors.common import compare_connector_docs
import weaviate
from weaviate.classes.config import Property, ReferenceProperty
from weaviate.classes.query import Filter
from weaviate.connect.base import ConnectionParams
from weaviate.collections.classes.config import Configure, DataType

from sycamore.data.docid import docid_to_uuid
import time


@pytest.fixture()
def wv_client_args():
    port = 8078
    grpc_port = 50059
    weaviate_client_args = {
        "embedded_options": weaviate.embedded.EmbeddedOptions(version="1.24.0", port=port, grpc_port=grpc_port),
    }
    with weaviate.WeaviateClient(**weaviate_client_args) as client:
        timeout = 30
        deadline = time.time() + timeout
        while not client.is_live():
            time.sleep(1)
            if time.time() > deadline:
                raise RuntimeError(f"Weaviate failed to start in {timeout} seconds")
        yield {
            "connection_params": ConnectionParams.from_params(
                http_host="localhost",
                http_port=port,
                http_secure=False,
                grpc_host="localhost",
                grpc_port=grpc_port,
                grpc_secure=False,
            ),
        }
        client.collections.delete("TestCollection")


def test_weaviate_read(wv_client_args, shared_ctx, embedded_transformer_paper):
    collection = "TestCollection"
    collection_config_params = {
        "name": collection,
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
        "references": [ReferenceProperty(name="parent", target_collection=collection)],
    }

    docs = embedded_transformer_paper.take_all()
    shared_ctx.read.document(docs).write.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params
    )
    out_docs = shared_ctx.read.weaviate(wv_client_args=wv_client_args, collection_name=collection).take_all()
    target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
    target_doc_uuid = docid_to_uuid(target_doc_id)
    fetch_object_dict = {"filters": Filter.by_id().equal(target_doc_uuid)}
    query_docs = shared_ctx.read.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, fetch_objects=fetch_object_dict
    ).take_all()
    assert len(query_docs) == 1  # exactly one doc should be returned
    compare_connector_docs(docs, out_docs)
