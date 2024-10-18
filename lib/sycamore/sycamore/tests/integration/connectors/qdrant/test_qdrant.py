import uuid
import sycamore
from sycamore.tests.integration.connectors.common import compare_connector_docs
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
        {"collection_name": collection_name, "limit": 100, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)


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
        # Not specifying the vector name
        # Should be handled by getting the first available vector
        {"collection_name": collection_name, "limit": 100, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)

    out_docs = ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        # Specify the vector name
        {"collection_name": collection_name, "limit": 100, "using": vector_name, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)
