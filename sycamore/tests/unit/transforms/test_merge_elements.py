import ray.data

from sycamore.data import Document
from sycamore.transforms.merge_elements import GreedyTextElementMerger, Merge
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestMergeElements:
    passage1 = """Recurrent neural networks, long short-term memory [12]
                                            and gated recurrent [7] neural networks in particular,
                                            have been ﬁrmly established as state of the art approaches
                                            in sequence modeling and transduction problems such as
                                            language modeling and machine translation [29, 2, 5].
                                            Numerous efforts have since continued to push the boundaries
                                            of recurrent language models and encoder-decoder architectures
                                            [31, 21, 13]."""
    passage2 = """The intuition behind the LSTM architecture is to create an additional module in a neural network
    that learns when to remember and when to forget pertinent information.[15] In other words, the network
    effectively learns which information might be needed later on in a sequence and when that information is no
    longer needed. For instance, in the context of natural language processing, the network can learn grammatical
    dependencies.[17]"""
    dict0 = {
        "doc_id": "doc_id",
        "type": "pdf",
        "text_representation": "text",
        "binary_representation": None,
        "parent_id": None,
        "properties": {"path": "/docs/foo.txt", "title": "bar"},
        "elements": {
            "array": [
                {
                    "type": "UncategorizedText",
                    "text_representation": "text1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "UncategorizedText",
                    "text_representation": "text2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Title",
                    "text_representation": "title1",
                    "bbox": [0.17, 0.40, 0.82, 0.47],
                    "binary_representation": b"title1",
                    "properties": {"doc_title": "title"},
                },
                {
                    "type": "NarrativeText",
                    "text_representation": passage1,
                    "bbox": [0.27, 0.50, 0.92, 0.67],
                    "properties": {"doc_title": "other title", "prop2": "prop 2 value"},
                },
                {
                    "type": "NarrativeText",
                    "text_representation": passage2,
                    "bbox": [0.27, 0.50, 0.92, 0.67],
                    "properties": {"doc_title": "other title", "prop2": "prop 2 value"},
                },
                {
                    "type": "Title",
                    "text_representation": "title2",
                    "bbox": [0.17, 0.40, 0.82, 0.47],
                    "binary_representation": b"title2",
                    "properties": {"doc_title": "title"},
                },
                {},
            ]
        },
    }

    def test_merge_elements(self):
        doc = Document(self.dict0)
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedyTextElementMerger(tokenizer, 120)
        new_doc = merger.merge_elements(doc)
        assert len(new_doc.elements) == 2

        e = new_doc.elements[0]
        assert e.type == "Section"
        assert e.text_representation == ("text1\ntext2\ntitle1\n" + self.passage1)
        assert e.bbox.coordinates == (0.17, 0.40, 0.92, 0.67)
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 1,
            "doc_title": "title",
            "prop2": "prop 2 value",
        }

        e = new_doc.elements[1]
        assert e.type == "Section"
        assert e.bbox.coordinates == (0.17, 0.40, 0.92, 0.67)
        assert e.binary_representation == b"title2"
        assert e.text_representation == (self.passage2 + "\ntitle2")
        assert e.properties == {"doc_title": "other title", "prop2": "prop 2 value"}

    def test_merge_elements_via_execute(self):
        plan = FakeNode(self.dict0)
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedyTextElementMerger(tokenizer, 120)
        merge = Merge(plan, merger)
        merge.execute()
