import pytest

from sycamore.tests.integration.scans.test_opensearch_scan import compare_docs
import weaviate
from weaviate.classes.config import Property, ReferenceProperty
from weaviate.client import ConnectionParams
from weaviate.collections.classes.config import Configure, DataType

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
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
        client.collections.delete("TestCollection")


def test_weaviate_scan(wv_client_args):

    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = HuggingFaceTokenizer(model_name)
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

    ctx = sycamore.init()

    docs = (
        ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .regex_replace(COALESCE_WHITESPACE)
        .mark_bbox_preset(tokenizer=tokenizer)
        .merge(merger=MarkedMerger())
        .spread_properties(["path"])
        .split_elements(tokenizer=tokenizer, max_tokens=512)
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
        .sketch(window=17)
        .take_all()
    )
    ctx.read.document(docs).write.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params
    )

    out_docs = ctx.read.weaviate(wv_client_args=wv_client_args, collection_name=collection).take_all()
    assert len(out_docs) == len(docs)
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )
