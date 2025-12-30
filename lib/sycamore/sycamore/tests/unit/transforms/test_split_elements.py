import ray.data

from sycamore.data import Document, TableElement, Table, TableCell
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
        result = SplitElements.split_one(element, CharacterTokenizer(), max=10)

        # Without early stopping, we get 1,013,259 elements.
        assert len(result) == 1

    def test_unsplittable_table_with_header(self):
        table_content = "one two three four five"
        max_chunks_size = 10
        assert len(table_content) > max_chunks_size, "Split precondition not met"

        element = TableElement()
        element.table = Table(cells=[TableCell(content=table_content, rows=[0], cols=[0])])
        element.data["_header"] = "foo"  # Prepending 'foo\n' to the table content can cause an infinite loop.
        result = SplitElements.split_one(element, CharacterTokenizer(), max=max_chunks_size)
        assert len(result) < 21, "Max depth exceeded"
