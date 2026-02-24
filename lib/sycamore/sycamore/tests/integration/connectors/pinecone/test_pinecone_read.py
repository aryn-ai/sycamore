from sycamore.tests.integration.connectors.common import compare_connector_docs
from pinecone import ServerlessSpec
from sycamore.connectors.common import generate_random_string

import os
from pinecone.grpc import PineconeGRPC
from pinecone import PineconeException
import time


def test_pinecone_read(shared_ctx, embedded_transformer_paper):

    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "test-index-read"
    namespace = f"{generate_random_string().lower()}"
    api_key = os.environ.get("PINECONE_API_KEY", "")
    assert (
        api_key is not None
    ), "Missing api key: either provide it as an argument or set the PINECONE_API_KEY env variable."

    pc = PineconeGRPC(api_key=api_key, source_tag="Aryn")

    docs = embedded_transformer_paper.take_all()

    shared_ctx.read.document(docs).write.pinecone(
        index_name=index_name, dimensions=384, namespace=namespace, index_spec=spec
    )
    target_doc_id = docs[-1].doc_id if docs[-1].doc_id and docs[0].doc_id else ""
    if len(target_doc_id) > 0:
        target_doc_id = f"{docs[-1].parent_id}#{target_doc_id}" if docs[-1].parent_id else target_doc_id
    wait_for_write_completion(client=pc, index_name=index_name, namespace=namespace, doc_id=target_doc_id)
    out_docs = shared_ctx.read.pinecone(index_name=index_name, api_key=api_key, namespace=namespace).take_all()
    query_params = {"namespace": namespace, "id": target_doc_id, "top_k": 1, "include_values": True}
    query_docs = shared_ctx.read.pinecone(
        index_name=index_name, api_key=api_key, query=query_params, namespace=namespace
    ).take_all()
    pc.Index(index_name).delete(namespace=namespace, delete_all=True)
    assert len(query_docs) == 1  # exactly one doc should be returned
    compare_connector_docs(docs, out_docs, parent_offset=1)


def wait_for_write_completion(client: PineconeGRPC, index_name: str, namespace: str, doc_id: str):
    """
    Takes the name of the last document to wait for and blocks until it's available and ready.
    """
    timeout = 30
    deadline = time.time() + timeout
    index = client.Index(index_name)
    while time.time() < deadline:
        try:
            desc = dict(index.fetch(ids=[doc_id], namespace=namespace).vectors).items()
            if len(desc) > 0:
                return
        except PineconeException:
            # NotFoundException means the last document has not been entered yet.
            pass
        time.sleep(1)
    raise TimeoutError(f"Document {doc_id} not found after {timeout} seconds.")
