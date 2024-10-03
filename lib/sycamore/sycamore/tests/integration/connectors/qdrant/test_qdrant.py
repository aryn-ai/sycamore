import uuid
import sycamore
from sycamore.connectors.common import compare_docs
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from qdrant_client import models


def test_qdrant():
    collection_name = uuid.uuid4().hex
    qdrant_url = "http://localhost:6333"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")

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
    ctx.read.document(docs).write.qdrant(
        {
            "url": qdrant_url,
        },
        {"collection_name": collection_name, "vectors_config": {"size": 384, "distance": "Cosine"}},
    )
    out_docs = ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        {"collection_name": collection_name, "limit": 100},
    ).take_all()

    assert len(out_docs) == len(docs)
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )


def test_qdrant_named_vector():
    collection_name = uuid.uuid4().hex
    qdrant_url = "http://localhost:6333"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
    vector_name = "test_vector"

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
    ctx.read.document(docs).write.qdrant(
        {
            "url": qdrant_url,
        },
        {
            "collection_name": collection_name,
            "vectors_config": {vector_name: models.VectorParams(size=384, distance=models.Distance.COSINE)},
        },
        vector_name,
    )
    out_docs = ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        {"collection_name": collection_name, "limit": 100},
    ).take_all()

    assert len(out_docs) == len(docs)
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )
