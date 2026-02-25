import uuid
from sycamore.tests.integration.connectors.common import compare_connector_docs
from qdrant_client import models


def test_qdrant(shared_ctx, embedded_transformer_paper):
    collection_name = uuid.uuid4().hex
    qdrant_url = "http://localhost:6333"

    docs = embedded_transformer_paper.take_all()
    shared_ctx.read.document(docs).write.qdrant(
        {
            "url": qdrant_url,
        },
        {"collection_name": collection_name, "vectors_config": {"size": 384, "distance": "Cosine"}},
    )
    out_docs = shared_ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        {"collection_name": collection_name, "limit": 100, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)


def test_qdrant_named_vector(shared_ctx, embedded_transformer_paper):
    collection_name = uuid.uuid4().hex
    qdrant_url = "http://localhost:6333"
    vector_name = "test_vector"

    docs = embedded_transformer_paper.take_all()
    shared_ctx.read.document(docs).write.qdrant(
        {
            "url": qdrant_url,
        },
        {
            "collection_name": collection_name,
            "vectors_config": {vector_name: models.VectorParams(size=384, distance=models.Distance.COSINE)},
        },
        vector_name,
    )
    out_docs = shared_ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        # Not specifying the vector name
        # Should be handled by getting the first available vector
        {"collection_name": collection_name, "limit": 100, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)

    out_docs = shared_ctx.read.qdrant(
        {
            "url": qdrant_url,
        },
        # Specify the vector name
        {"collection_name": collection_name, "limit": 100, "using": vector_name, "with_vectors": True},
    ).take_all()

    compare_connector_docs(docs, out_docs)
