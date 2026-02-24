import pytest

import weaviate
from weaviate.classes.config import Property, ReferenceProperty
from weaviate.connect.base import ConnectionParams
from weaviate.collections.classes.config import Configure, DataType
import time


@pytest.fixture()
def wv_client_args():
    port = 8078
    grpc_port = 50059
    weaviate_client_args = {
        "embedded_options": weaviate.embedded.EmbeddedOptions(version="1.24.0", port=port, grpc_port=grpc_port),
    }
    with weaviate.WeaviateClient(**weaviate_client_args) as client:
        time.time()
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


def test_to_weaviate(wv_client_args, embedded_transformer_paper):
    collection = "DemoCollection"

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
    embedded_transformer_paper.write.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params
    )
