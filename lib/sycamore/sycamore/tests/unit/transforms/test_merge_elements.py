import pytest
import ray.data

import sycamore
from sycamore.data import Document
from sycamore.transforms.merge_elements import (
    GreedyTextElementMerger,
    Merge,
    GreedySectionMerger,
    HeaderAugmenterMerger,
)
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self, **kwargs) -> ray.data.Dataset:
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
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
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
                    "text_representation": "title3",
                    "bbox": [0.17, 0.40, 0.82, 0.47],
                    "binary_representation": b"title3",
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
            ],
        }
    )

    def test_merge_elements(self):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedyTextElementMerger(tokenizer, 120)

        new_doc = merger.merge_elements(self.doc)
        assert len(new_doc.elements) == 2

        e = new_doc.elements[0]
        assert e.type == "Section"
        assert e.text_representation == ("text1\ntext2\ntitle3\n" + self.passage1)
        assert e.bbox.coordinates == (0.17, 0.40, 0.92, 0.67)
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 1,
            "page_numbers": [1, 2],
            "doc_title": "title",
            "prop2": "prop 2 value",
        }

        e = new_doc.elements[1]
        assert e.type == "Section"
        assert e.bbox.coordinates == (0.17, 0.40, 0.92, 0.67)
        assert e.binary_representation == b"title2"
        assert e.text_representation == (self.passage2 + "\ntitle2")
        assert e.properties == {"doc_title": "other title", "prop2": "prop 2 value"}

    def test_merge_elements_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedyTextElementMerger(tokenizer, 120)
        merge = Merge(node, merger)
        output_dataset = merge.execute()
        output_dataset.show()

    def test_docset_greedy(self):
        ray.shutdown()

        context = sycamore.init()
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        context.read.document([self.doc]).merge(GreedyTextElementMerger(tokenizer, 120)).show()

        # Verify that GreedyTextElementMerger can't be an argument for map.
        # We may want to change this in the future.
        with pytest.raises(ValueError):
            sycamore.init().read.document([self.doc]).map(GreedyTextElementMerger(tokenizer, 120))


class TestGreedySectionMerger:

    # 'text' + 'text' = 'text'
    # 'image' + 'text' = 'image+text'
    # 'image+text' + 'text' = 'image+text'
    # 'Section-header' + 'table' = 'Section-header+table'

    # doc = "text1 text2 text3 || text4 Image text5 text6 || Section-header table "
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
                {
                    "type": "Text",
                    "text_representation": "text1 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text2 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text3 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text4 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Image",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text5 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text6 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Section-header",
                    "text_representation": "Section-header1 on page 3",
                    "properties": {"page_number": 3},
                },
                {
                    "type": "table",
                    "text_representation": "table1 on page 3",
                    "properties": {"page_number": 3},
                },
                {},
            ],
        }
    )
    doc1 = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
                {
                    "type": "Text",
                    "text_representation": "text1 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text2 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text3 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text4 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Image",
                    "properties": {
                        "filetype": "text/plain",
                        "page_number": 2,
                        "summary": {"isgraph": False, "summary": "image1 on page 2 before text5 from summary"},
                    },
                },
                {
                    "type": "Text",
                    "text_representation": "text5 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text6 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Section-header",
                    "text_representation": "Section-header1 on page 3",
                    "properties": {"page_number": 3},
                },
                {
                    "type": "table",
                    "text_representation": "table1 on page 3",
                    "properties": {"page_number": 3},
                },
                {},
            ],
        }
    )

    def test_merge_elements(self):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedySectionMerger(tokenizer, 1200, merge_across_pages=False)

        new_doc = merger.merge_elements(self.doc)
        assert len(new_doc.elements) == 5

        # doc = "text1 text2 text3 || text4 Image text5 text6 || Section-header table"
        # new_doc = "text1+text2+text3 || text4 Image+text5+text6 || Section-header+table"
        # text1+text2+text3
        e = new_doc.elements[0]
        assert e.type == "Text"
        assert e.text_representation == ("text1 on page 1\ntext2 on page 1\ntext3 on page 1")
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 1,
            "page_numbers": [1],
        }

        e = new_doc.elements[1]
        assert e.type == "Text"
        assert e.text_representation == ("text4 on page 2")
        assert e.properties == {"filetype": "text/plain", "page_number": 2}

        e = new_doc.elements[2]
        assert e.type == "Image+Text"
        assert e.text_representation == ("text5 on page 2\ntext6 on page 2")
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 2,
            "image_format": None,
            "image_mode": None,
            "image_size": None,
            "page_numbers": [2],
        }

        e = new_doc.elements[3]
        assert e.type == "Section-header+table"
        # TODO: figure out the table representation and enhance the test with the to_html of table1
        assert e.text_representation == ("Section-header1 on page 3")
        assert e.properties == {"page_number": 3, "columns": None, "page_numbers": [3], "rows": None, "title": None}

    def test_merge_elements_image_summarize(self):

        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedySectionMerger(tokenizer, 1200, merge_across_pages=False)

        new_doc = merger.merge_elements(self.doc1)
        assert len(new_doc.elements) == 5

        # doc = "text1 text2 text3 || text4 Image text5 text6 || Section-header table"
        # new_doc = "text1+text2+text3 || text4 Image+text5+text6 || Section-header+table"
        # text1+text2+text3
        e = new_doc.elements[0]
        assert e.type == "Text"
        assert e.text_representation == ("text1 on page 1\ntext2 on page 1\ntext3 on page 1")
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 1,
            "page_numbers": [1],
        }

        e = new_doc.elements[1]
        assert e.type == "Text"
        assert e.text_representation == ("text4 on page 2")
        assert e.properties == {"filetype": "text/plain", "page_number": 2}

        e = new_doc.elements[2]
        assert e.type == "Image+Text"
        assert e.text_representation == ("image1 on page 2 before text5 from summary\ntext5 on page 2\ntext6 on page 2")
        assert e.properties["page_number"] == 2

        e = new_doc.elements[3]
        assert e.type == "Section-header+table"
        # TODO: figure out the table representation and enhance the test with the to_html of table1
        assert e.text_representation == ("Section-header1 on page 3")
        assert e.properties == {"page_number": 3, "columns": None, "page_numbers": [3], "rows": None, "title": None}

    def test_merge_elements_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedySectionMerger(tokenizer, 120, merge_across_pages=False)
        merge = Merge(node, merger)
        output_dataset = merge.execute()
        output_dataset.show()

    def test_docset_greedy(self):
        ray.shutdown()

        context = sycamore.init()
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        context.read.document([self.doc]).merge(GreedySectionMerger(tokenizer, 120, merge_across_pages=False)).show()

        # Verify that GreedyTextElementMerger can't be an argument for map.
        # We may want to change this in the future.
        with pytest.raises(ValueError):
            sycamore.init().read.document([self.doc]).map(GreedySectionMerger(tokenizer, 120, merge_across_pages=False))


class TestHeaderAugmenterMerger:

    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
                {
                    "type": "Section-header",
                    "text_representation": "section1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Section-header",
                    "text_representation": "section1.1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Text",
                    "text_representation": "text1 on page 1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "Table",
                    "text_representation": "table1 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Title",
                    "text_representation": "title1 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Section-header",
                    "text_representation": "section2 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text2 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text3 on page 2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Text",
                    "text_representation": "text4 on page 3",
                    "properties": {"filetype": "text/plain", "page_number": 3},
                },
                {},
            ],
        }
    )

    def test_merge_elements(self):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = HeaderAugmenterMerger(tokenizer, 1200, merge_across_pages=True)

        new_doc = merger.merge_elements(self.doc)
        assert len(new_doc.elements) == 4
        e = new_doc.elements[0]
        assert e.type == "Text"
        assert e.text_representation == ("section1\nsection1.1\ntext1 on page 1")
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 1,
        }
        assert e["_header"] == "section1\nsection1.1"

        e = new_doc.elements[1]
        assert e.type == "table"
        assert e.text_representation == ("section1\nsection1.1\ntable1 on page 2")
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 2,
            "title": None,
            "columns": None,
            "rows": None,
        }
        assert e["_header"] == "section1\nsection1.1"

        e = new_doc.elements[2]
        assert e.type == "Text"
        assert e.text_representation == (
            "title1 on page 2\nsection2 on page 2\ntext2 on page 2\ntext3 on page 2\ntext4 on page 3"
        )
        assert e.properties == {
            "filetype": "text/plain",
            "page_number": 2,
            "page_numbers": [2, 3],
        }
        assert e["_header"] == "title1 on page 2\nsection2 on page 2"

    def test_merge_elements_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        merger = HeaderAugmenterMerger(tokenizer, 120, merge_across_pages=True)
        merge = Merge(node, merger)
        output_dataset = merge.execute()
        output_dataset.show()

    def test_docset_greedy(self):
        ray.shutdown()

        context = sycamore.init()
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        context.read.document([self.doc]).merge(HeaderAugmenterMerger(tokenizer, 120, merge_across_pages=True)).show()

        # Verify that GreedyTextElementMerger can't be an argument for map.
        # We may want to change this in the future.
        with pytest.raises(ValueError):
            sycamore.init().read.document([self.doc]).map(
                HeaderAugmenterMerger(tokenizer, 120, merge_across_pages=True)
            )
