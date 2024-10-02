import sycamore

from sycamore.data import Document
from sycamore.transforms.similarity import HuggingFaceTransformersSimilarityScorer


def test_rerank_docset():
    similarity_scorer = HuggingFaceTransformersSimilarityScorer()
    score_property_name = "similarity_score"
    dicts = [
        {
            "doc_id": 1,
            "elements": [
                {"text_representation": "here is an animal with 4 legs and whiskers"},
            ],
        },
        {
            "doc_id": 2,
            "elements": [
                {"id": 7, "text_representation": "this is a cat"},
                {"id": 1, "text_representation": "this is a dog"},
            ],
        },
        {
            "doc_id": 3,
            "elements": [
                {"text_representation": "this is a dog"},
            ],
        },
        {"doc_id": 4, "elements": [{"text_representation": "the number of pages in this document are 253"}]},
        {  # handle element with not text
            "doc_id": 5,
            "elements": [
                {"id": 1},
            ],
        },
    ]
    docs = [Document(item) for item in dicts]

    context = sycamore.init()
    doc_set = context.read.document(docs).rerank(
        similarity_scorer=similarity_scorer, query="is this a cat?", score_property_name=score_property_name
    )
    result = doc_set.take()

    assert len(result) == len(docs)
    assert [doc.doc_id for doc in result] == [2, 1, 3, 5, 4]

    for doc in result:
        if doc.doc_id == 5:
            continue
        assert float(doc.properties.get(score_property_name))


def test_rerank_docset_exploded():
    similarity_scorer = HuggingFaceTransformersSimilarityScorer(ignore_doc_structure=True)
    score_property_name = "similarity_score"
    dicts = [
        {"doc_id": 1, "text_representation": "here is an animal with 4 legs and whiskers"},
        {"doc_id": 2, "text_representation": "this is a cat"},
        {"doc_id": 3, "text_representation": "this is a dog"},
        {
            "doc_id": 4,
            "elements": [
                {"text_representation": "this doc doesn't have a text representation but instead has an element"}
            ],
        },
        {"doc_id": 5, "text_representation": "the number of pages in this document are 253"},
    ]
    docs = [Document(item) for item in dicts]

    context = sycamore.init()
    doc_set = context.read.document(docs).rerank(
        similarity_scorer=similarity_scorer, query="is this a cat?", score_property_name=score_property_name
    )
    result = doc_set.take()

    assert len(result) == len(docs)
    assert [doc.doc_id for doc in result] == [2, 1, 3, 5, 4]

    for doc in result:
        if doc.doc_id == 4:
            continue
        assert float(doc.properties.get(score_property_name))
