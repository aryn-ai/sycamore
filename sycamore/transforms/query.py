from abc import abstractmethod, ABC
from typing import Any

import requests
from ray.data import Dataset

from sycamore.data import OpenSearchQueryResult, Element, OpenSearchQuery
from sycamore.plan_nodes import Node, NonCPUUser, NonGPUUser, Transform
from sycamore.utils.generate_ray_func import generate_map_function


class QueryExecutor(ABC):
    @abstractmethod
    def query(self, query: Any) -> Any:
        pass

    def __call__(self, query: Any) -> Any:
        return self.query(query)


class OpenSearchQueryExecutor(QueryExecutor):
    def __init__(self, opensearch_endpoint) -> None:
        super().__init__()
        self._opensearch_endpoint = opensearch_endpoint

    def query(self, query: OpenSearchQuery) -> OpenSearchQueryResult:
        params = {
            "q": query["query"],
        }
        url = self._opensearch_endpoint + (query["url_params"] if "url_params" in query else "")
        response = requests.get(url, params=params)
        result = OpenSearchQueryResult()
        result.query = {"url": url, "params": params}
        result.result = response.json()
        if response.status_code == 200:
            content = response.json()
            result.hits = [Element(hit["_source"]) for hit in content["hits"]["hits"]]
            if "ext" in content and "retrieval_augmented_generation" in content["ext"]:
                result.generated_answer = content["ext"]["retrieval_augmented_generation"]["answer"]
        else:
            print(f"Error: {response.status_code}")
        return result


class Query(NonCPUUser, NonGPUUser, Transform):
    """
    Given a DocSet of queries, executes them and generates a DocSet of query results.
    """

    def __init__(self, child: Node, query_executor: QueryExecutor, **kwargs):
        super().__init__(child, **kwargs)
        self._query_executor = query_executor

    def execute(self) -> Dataset:
        input_ds = self.child().execute()
        output_ds = input_ds.map(generate_map_function(self._query_executor.query))
        return output_ds
