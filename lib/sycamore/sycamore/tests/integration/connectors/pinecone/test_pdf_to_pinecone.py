from pinecone import ServerlessSpec

import os
from sycamore.connectors.common import generate_random_string
from pinecone.grpc import PineconeGRPC


def test_to_pinecone(embedded_transformer_paper):
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "test-index-write"
    namespace = f"{generate_random_string().lower()}"
    api_key = os.environ.get("PINECONE_API_KEY", "")
    assert (
        api_key is not None and len(api_key) != 0
    ), "Missing api key: either provide it as an argument or set the PINECONE_API_KEY env variable."
    pc = PineconeGRPC(api_key=api_key, source_tag="Aryn")

    embedded_transformer_paper.write.pinecone(
        index_name=index_name, namespace=namespace, dimensions=384, index_spec=spec
    )
    pc.Index(index_name).delete(namespace=namespace, delete_all=True)
