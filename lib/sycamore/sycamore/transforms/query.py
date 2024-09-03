from abc import abstractmethod, ABC
from typing import Any
from sycamore.utils.import_utils import requires_modules

from sycamore.data import OpenSearchQueryResult, Element, OpenSearchQuery
from sycamore.plan_nodes import Node, NonCPUUser, NonGPUUser
from sycamore.transforms.map import Map
import logging

logger = logging.getLogger("ray")


class QueryExecutor(ABC):
    @abstractmethod
    def query(self, query: Any) -> Any:
        pass

    def __call__(self, query: Any) -> Any:
        return self.query(query)


class OpenSearchQueryExecutor(QueryExecutor):
    def __init__(self, os_client_args: dict) -> None:
        super().__init__()
        self._os_client_args = os_client_args

    @requires_modules("opensearchpy", extra="opensearch")
    def query(self, query: OpenSearchQuery) -> OpenSearchQueryResult:
        from opensearchpy import OpenSearch

        logger.debug("Executing OS query: " + str(query))
        client = OpenSearch(**self._os_client_args)

        os_result = client.transport.perform_request(
            "POST",
            url=f"/{query['index']}/_search",
            params=query.get("params", None),
            headers=query.get("headers", None),
            body=query["query"],
        )
        result = OpenSearchQueryResult(query)
        result.result = os_result
        result.hits = [Element(hit["_source"]) for hit in os_result["hits"]["hits"]]
        if "ext" in os_result and "retrieval_augmented_generation" in os_result["ext"]:
            result.generated_answer = os_result["ext"]["retrieval_augmented_generation"]["answer"]
        return result


class Query(NonCPUUser, NonGPUUser, Map):
    """
    Given a DocSet of queries, executes them and generates a DocSet of query results.
    """

    def __init__(self, child: Node, query_executor: QueryExecutor, **kwargs):
        super().__init__(child, f=query_executor.query, **kwargs)
