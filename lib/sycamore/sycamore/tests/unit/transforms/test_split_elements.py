import ray.data

from sycamore.data import Document, TableElement, Table, TableCell
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.merge_elements import HeaderAugmenterMerger
from sycamore.transforms.split_elements import SplitElements
from sycamore.functions.tokenizer import HuggingFaceTokenizer, CharacterTokenizer
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self, **kwargs) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestSplitElements:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "foobar",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
                {
                    "type": "UncategorizedText",
                    "text_representation": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour; twentyfive, twentysix. twentyseven twentyeight, twentynine; thirty thirtyone thirtytwo thirtythree thirtyfour thirtyfive thirtysix thirtyseven thirtyeight thirtynine forty fortyone fortytwo fortythree fortyfour fortyfive fortysix fortyseven fortyeight fortynine",  # noqa: E501
                },
            ],
        }
    )

    def test_split_elements(self):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        doc = SplitElements(None, tokenizer, 15).run(self.doc)
        elems = doc.elements
        self.validateElems(elems)

    def test_via_execute(self, mocker):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        node = mocker.Mock(spec=Node)
        se = SplitElements(node, tokenizer, 15)
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        ds = se.execute()
        doc = Document.from_row(ds.take(limit=1)[0])
        elems = doc.elements
        self.validateElems(elems)

    def validateElems(self, elems):
        assert len(elems) == 10
        assert elems[0].text_representation == "One two three four five six seven eight nine ten "
        assert elems[1].text_representation == "eleven twelve thirteen fourteen fifteen sixteen "
        assert elems[2].text_representation == "seventeen eighteen nineteen twenty twentyone twentytwo "
        assert elems[3].text_representation == "twentythree twentyfour; twentyfive, twentysix."
        assert elems[4].text_representation == " twentyseven twentyeight, twentynine;"
        assert elems[5].text_representation == " thirty thirtyone thirtytwo thirtythree "
        assert elems[6].text_representation == "thirtyfour thirtyfive thirtysix thirtyseven "
        assert elems[7].text_representation == "thirtyeight thirtynine forty fortyone fortytwo fortythree "
        assert elems[8].text_representation == "fortyfour fortyfive fortysix "
        assert elems[9].text_representation == "fortyseven fortyeight fortynine"

    def test_unsplittable_large_table_headers(self):
        element = TableElement()
        element.table = Table(cells=[TableCell(content="one\ntwo\nthree", rows=[0], cols=[0])])
        element.table.column_headers = [" ".join(["header "] * 100)]
        result = SplitElements.split_one(element, CharacterTokenizer(), 10)

        # one two three -> one two | three
        assert len(result) == 2

    def test_unsplittable_large_table_headers2(self):
        element = TableElement()
        element.table = Table(cells=[TableCell(content="one\ntwo\nthree", rows=[0], cols=[0])])
        element.table.column_headers = [" ".join(["header "] * 100)]

        doc = Document({"elements": [element]})
        result = SplitElements.split_doc(doc, tokenizer=CharacterTokenizer(), max=10, max_depth=None)

        # one two three -> one two | three
        assert len(result.elements) == 2

    def test_unsplittable_table_with_header(self):
        table_content = "one two three four five"
        max_chunks_size = 10
        assert len(table_content) > max_chunks_size, "Split precondition not met"

        element = TableElement()
        element.table = Table(cells=[TableCell(content=table_content, rows=[0], cols=[0])])
        element.data["_header"] = "foo"  # Prepending 'foo\n' to the table content can cause an infinite loop.
        result = SplitElements.split_one(element, CharacterTokenizer(), max_chunks_size)
        assert len(result) < 21, "Max depth exceeded"

    def test_unsplittable_table(self):
        import json
        from sycamore.functions.tokenizer import OpenAITokenizer

        large_table_json = TEST_DIR / "resources" / "data" / "json" / "large_table.json"
        with open(large_table_json, "r") as f:
            res = json.load(f)

        tokenizer = OpenAITokenizer(model_name="text-embedding-3-small")
        doc = Document({"elements": res["elements"]})

        orig_element_count = len(doc["elements"])
        orig_table_element_count = 0
        for elem in doc["elements"]:
            if isinstance(elem, TableElement):
                orig_table_element_count += 1

        merger = HeaderAugmenterMerger(tokenizer=tokenizer, max_tokens=512, merge_across_pages=True)
        merger.merge_elements(doc)

        merged_element_count = len(doc["elements"])
        merged_table_element_count = 0
        for elem in doc["elements"]:
            if isinstance(elem, TableElement):
                merged_table_element_count += 1

        assert merged_element_count < orig_element_count
        assert merged_table_element_count <= orig_table_element_count

        SplitElements.split_doc(doc, tokenizer=tokenizer, max=512, max_depth=None, add_binary=False)
        split_element_count = len(doc["elements"])
        split_table_element_count = 0
        for elem in doc["elements"]:
            if isinstance(elem, TableElement):
                split_table_element_count += 1

        assert split_element_count > merged_element_count
        assert split_table_element_count > merged_table_element_count

    def test_max_depth_hit_and_raised(self):
        element = TableElement()
        element.table = Table(cells=[TableCell(content="one\ntwo\nthree", rows=[0], cols=[0])])
        element.table.column_headers = [" ".join(["header "] * 100)]

        doc = Document({"elements": [element]})
        max_tokens = 10
        from sycamore.functions.tokenizer import Tokenizer
        from typing import Union
        from functools import cache

        class DummyTokenizer(Tokenizer):
            @cache
            def tokenize(self, text: str, as_ints: bool = False) -> Union[list[int], list[str]]:
                return ["x" for _ in range(max_tokens + 1)]

        result = SplitElements.split_doc(doc, tokenizer=DummyTokenizer(), max=max_tokens, raise_on_max_depth=True)

        # Exception raised, no splitting happened
        assert len(result.elements) == 1
