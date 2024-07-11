from pinecone import ServerlessSpec

import os
from sycamore.connectors.common import generate_random_string
import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from pinecone.grpc import PineconeGRPC


def test_to_pinecone():
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "test-index-write"
    namespace = f"{generate_random_string().lower()}"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
    api_key = os.environ.get("PINECONE_API_KEY", "")
    assert (
        api_key is not None and len(api_key) != 0
    ), "Missing api key: either provide it as an argument or set the PINECONE_API_KEY env variable."
    pc = PineconeGRPC(api_key=api_key)
    tokenizer = HuggingFaceTokenizer(model_name)

    ctx = sycamore.init()

    ds = (
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
    )
    ds.write.pinecone(index_name=index_name, namespace=namespace, dimensions=384, index_spec=spec)
    pc.Index(index_name).delete(namespace=namespace, delete_all=True)
