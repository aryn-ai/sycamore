from typing import Optional

from sycamore import Context
from sycamore.context import context_params
from sycamore.transforms import Embedder


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
