import logging
from typing import Optional, Any

from opensearchpy import OpenSearch

from sycamore import Context
from sycamore.context import context_params
from sycamore.transforms import Embedder


logger = logging.getLogger("opensearch")


class OpenSearchClientWithLogging(OpenSearch):
    def __init__(self, *args, **kwargs) -> None:
        try:
            hosts = " ".join([f"{d['host']}:{d['port']}" for d in kwargs["hosts"]])
        except (KeyError, TypeError, ValueError):
            hosts = None
        try:
            user = kwargs["http_auth"][0]
        except (KeyError, TypeError):
            user = None
        logger.info(f"Connecting to OpenSearch as {user} to {hosts}")
        super().__init__(*args, **kwargs)

    def search(self, **kwargs) -> Any:
        """Helper method to execute OpenSearch search queries, and silent errors."""
        response = super().search(**kwargs)
        shards = response.get("_shards", {})
        if shards.get("total") != shards.get("successful"):
            logger.error(f"OpenSearch query skipped shards: {response}")
        return response

    def parallel_bulk(self, record_gen, **kwargs):
        from opensearchpy.helpers import parallel_bulk

        return parallel_bulk(self, record_gen, **kwargs)


@context_params("opensearch")
def get_knn_query(
    text_embedder: Embedder,
    query_phrase: str,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
    context: Optional[Context] = None,
):
    """
    Given a query string and an Embedder, create a simple OpenSearch Knn query.
    Supports either 'k' to retrieve k-ANNs, or min_score to return all records within a given distance score.
    Uses a default k value of 500.
    This is only the base query, if you need to add filters or other specifics you can extend the object.
    """

    if k is None and min_score is None:
        k = 500
    elif k is not None and min_score is not None:
        raise ValueError("Only one of `k` or `min_score` should be populated")

    embedding = text_embedder.generate_text_embedding(query_phrase)
    query = {"query": {"knn": {"embedding": {"vector": embedding}}}}
    if k is not None:
        query["query"]["knn"]["embedding"]["k"] = k  # type: ignore
    else:
        query["query"]["knn"]["embedding"]["min_score"] = min_score  # type: ignore
    return query
