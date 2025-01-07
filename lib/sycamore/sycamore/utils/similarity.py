from typing import Optional
from sycamore.transforms.similarity import SimilarityScorer


def make_element_sorter_fn(field: str, similarity_query: Optional[str], similarity_scorer: Optional[SimilarityScorer]):
    if similarity_query is None:
        return lambda d: None

    assert similarity_scorer is not None, "Similarity sorting requires a scorer"

    def f(doc):
        score_property_name = f"{field}_similarity_score"
        doc = similarity_scorer.generate_similarity_scores(
            doc_batch=[doc], query=similarity_query, score_property_name=score_property_name
        )[0]
        doc.elements.sort(key=lambda e: e.properties.get(score_property_name, float("-inf")), reverse=True)

    return f
