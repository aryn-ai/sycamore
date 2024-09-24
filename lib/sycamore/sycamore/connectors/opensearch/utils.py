from typing import Optional

from sycamore import Context
from sycamore.context import context_params
from sycamore.transforms import Embedder


@context_params("opensearch")
def get_knn_query(text_embedder: Embedder, query_phrase: str, k: int = 500, context: Optional[Context] = None):
    """
    Given a query string and an Embedder, create a simple OpenSearch Knn query. Uses a default k value of 500.
    This is only the base query, if you need to add filters or other specifics you can extend the object.
    """
    embedding = text_embedder.generate_text_embedding(query_phrase)
    query = {
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": k,
                }
            }
        }
    }
    return query
