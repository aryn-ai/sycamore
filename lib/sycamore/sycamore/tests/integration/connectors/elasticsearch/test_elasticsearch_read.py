from sycamore.tests.integration.connectors.common import compare_connector_docs
from elasticsearch import Elasticsearch


def test_to_elasticsearch(shared_ctx, embedded_transformer_paper):
    url = "http://localhost:9201"
    index_name = "test_index-read"
    wait_for_completion = "wait_for"

    docs = embedded_transformer_paper.take_all()
    shared_ctx.read.document(docs).write.elasticsearch(
        url=url, index_name=index_name, wait_for_completion=wait_for_completion
    )
    target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
    out_docs = shared_ctx.read.elasticsearch(url=url, index_name=index_name).take_all()
    query_params = {"term": {"_id": target_doc_id}}
    query_docs = shared_ctx.read.elasticsearch(url=url, index_name=index_name, query=query_params).take_all()
    with Elasticsearch(url) as es_client:
        es_client.indices.delete(index=index_name)
    assert len(query_docs) == 1  # exactly one doc should be returned
    compare_connector_docs(docs, out_docs)
