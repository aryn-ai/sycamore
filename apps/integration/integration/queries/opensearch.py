from opensearchpy import OpenSearch


class OpenSearchHelper:
    def __init__(self, opensearch: OpenSearch):
        self._osc = opensearch

    def get_embedding_model(self):
        response = self._osc.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_search",
            body={
                "query": {
                    "bool": {
                        "must_not": [{"exists": {"field": "chunk_number"}}],
                        "must": [{"term": {"algorithm": "TEXT_EMBEDDING"}}],
                    }
                }
            },
        )
        return response["hits"]["hits"][0]["_id"]

    def get_reranking_model(self):
        response = self._osc.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_search",
            body={
                "query": {
                    "bool": {
                        "must_not": [{"exists": {"field": "chunk_number"}}],
                        "must": [{"term": {"algorithm": "TEXT_SIMILARITY"}}],
                    }
                }
            },
        )
        return response["hits"]["hits"][0]["_id"]

    def get_remote_model(self):
        response = self._osc.transport.perform_request(
            method="POST",
            url="/_plugins/_ml/models/_search",
            body={
                "query": {
                    "bool": {
                        "must_not": [{"exists": {"field": "chunk_number"}}],
                        "must": [{"term": {"algorithm": "REMOTE"}}],
                    }
                }
            },
        )
        return response["hits"]["hits"][0]["_id"]

    def get_index_mappings(self, index_name):
        response = self._osc.indices.get_mapping(index=index_name)
        return response

    def get_memory_id(self):
        response = self._osc.transport.perform_request(
            method="GET", url="/_plugins/_ml/memory", params={"max_results": 1}
        )
        if len(response["memories"]) > 0:
            return response["memories"][0]["memory_id"]
        else:
            create_mem_response = self._osc.transport.perform_request(
                method="POST", url="/_plugins/_ml/memory", body={"name": "INTEGRATION TESTS"}
            )
            return create_mem_response["memory_id"]
