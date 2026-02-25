from elasticsearch import Elasticsearch


def test_to_elasticsearch(embedded_transformer_paper):
    url = "http://localhost:9201"
    index_name = "test_index-write"
    wait_for_completion = "wait_for"

    ds = embedded_transformer_paper
    count = ds.count()
    ds.write.elasticsearch(url=url, index_name=index_name, wait_for_completion=wait_for_completion)
    with Elasticsearch(url) as es_client:
        es_count = int(es_client.cat.count(index=index_name, format="json")[0]["count"])
        es_client.indices.delete(index=index_name)
    assert count == es_count
