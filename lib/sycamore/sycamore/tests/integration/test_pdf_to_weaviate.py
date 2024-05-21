import pytest

import weaviate
from weaviate.classes.config import Property, ReferenceProperty
from weaviate.client import ConnectionParams
from weaviate.collections.classes.config import Configure, DataType

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import SycamorePartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
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


def test_to_weaviate(wv_client_args):
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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/")

    OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
    tokenizer = HuggingFaceTokenizer(model_name)

    ctx = sycamore.init()

    ds = (
        ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=SycamorePartitioner())
        .regex_replace(COALESCE_WHITESPACE)
        .mark_bbox_preset(tokenizer=tokenizer)
        .merge(merger=MarkedMerger())
        .spread_properties(["path"])
        .split_elements(tokenizer=tokenizer, max_tokens=512)
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
        .sketch(window=17)
    )
    ds.write.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params
    )
