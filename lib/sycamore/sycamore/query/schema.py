from opensearchpy.client.indices import IndicesClient
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.data import OpenSearchQuery


class OpenSearchSchema:
    def __init__(self, client: IndicesClient, index: str, query_executor: OpenSearchQueryExecutor) -> None:
        super().__init__()
        self._client = client
        self._index = index
        self._query_executor = query_executor

    def get_schema(self) -> dict[str, str]:
        schema = self._client.get_field_mapping(fields=["*"], index=self._index)
        query = OpenSearchQuery()
        query["index"] = self._index
        query["query"] = {"query": {"match_all": {}}, "size": 10}

        random_sample = self._query_executor.query(query)["result"]["hits"]["hits"]
        result = {}

        for k, v in schema[self._index]["mappings"].items():
            if k.startswith("properties.entity") and ".keyword" not in k:
                key = k
                # key = k[18:]
                result[key] = f"({type(random_sample[0]['_source']['properties']['entity'][k[18:]])}) e.g. "

                for i in range(len(random_sample)):
                    result[key] += "(" + str(random_sample[i]["_source"]["properties"]["entity"][k[18:]]) + ")"

                    if i != 9:
                        result[key] += ", "

        result["text_representation"] = "(<class 'str'>) Can be assumed to have all other details"

        return result
