import sycamore

from sycamore.data import Document
from sycamore.transforms.similarity import HuggingFaceTransformersSimilarityScorer

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


class TestRerank:

    def test_rerank_docset(self, exec_mode):

        similarity_scorer = HuggingFaceTransformersSimilarityScorer(RERANKER_MODEL, batch_size=5)
        score_property_name = "similarity_score"
        dicts = [
            {
                "doc_id": 1,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": 2,
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": 3,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {  # handle element with not text
                "doc_id": 4,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": 5,
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {  # drop because of limit
                "doc_id": 6,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        context = sycamore.init(exec_mode=exec_mode)
        doc_set = context.read.document(docs).rerank(
            similarity_scorer=similarity_scorer,
            query="is this a cat?",
            score_property_name=score_property_name,
            limit=5,
        )

        result = doc_set.take()

        assert len(result) == len(docs) - 1
        assert [doc.doc_id for doc in result] == [2, 1, 3, 5, 4]

        for doc in result:
            if doc.doc_id == 4:
                continue
            assert float(doc.properties.get(score_property_name))

    def test_rerank_docset_exploded(self, exec_mode):

        similarity_scorer = HuggingFaceTransformersSimilarityScorer(
            RERANKER_MODEL, ignore_doc_structure=True, batch_size=5
        )

        score_property_name = "similarity_score"
        dicts = [
            {"doc_id": 1, "text_representation": "here is an animal that meows"},
            {"doc_id": 2, "text_representation": "this is a cat"},
            {"doc_id": 3, "text_representation": "here is an animal that moos"},
            {
                "doc_id": 4,
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "this doc doesn't have a text representation but instead has an element",
                    }
                ],
            },
            {"doc_id": 5, "text_representation": "the number of pages in this document are 253"},
        ]
        docs = [Document(item) for item in dicts]

        context = sycamore.init(exec_mode=exec_mode)
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
