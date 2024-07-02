from sycamore.connectors.writer_utils.common import compare_docs
from pinecone import ServerlessSpec

import os
import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from pinecone.grpc import PineconeGRPC


def test_pinecone_scan():

    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "test-index"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    namespace = "test-namespace"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
    api_key = os.environ.get("PINECONE_API_KEY", "")
    assert (
        api_key is not None
    ), "Missing api key: either provide it as an argument or set the PINECONE_API_KEY env variable."

    pc = PineconeGRPC(api_key=api_key)

    tokenizer = HuggingFaceTokenizer(model_name)

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
    ctx.read.document(docs).write.pinecone(
        index_name=index_name, dimensions=384, namespace=namespace, index_spec=spec, log=True
    )
    out_docs = ctx.read.pinecone(index_name=index_name, api_key=api_key, namespace=namespace).take_all()
    pc.delete_index(index_name)
    assert len(docs) == (len(out_docs) + 1)  # parent doc is removed while writing
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )
