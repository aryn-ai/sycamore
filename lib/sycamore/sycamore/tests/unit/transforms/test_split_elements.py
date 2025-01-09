import ray.data

from sycamore.data import Document, TableElement, Table
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

    bigtable = """
    <table>
    <tr><th rowspan="2">headerA</th><th colspan="2">headerB</th><th>headerC</th></tr>
    <tr><th>headerD</th><th>headerE</th><th>headerF</th></tr>
    <tr><td>data1a</td><td>data2a</td><td>data3a</td><td>data4a</td></tr>
    <tr><td>data1b</td><td>data2b</td><td>data3b</td><td>data4b</td></tr>
    <tr><td>data1c</td><td>data2c</td><td>data3c</td><td>data4c</td></tr>
    <tr><td>data1d</td><td>data2d</td><td>data3d</td><td>data4d</td></tr>
    </table>
    """

    tabledoc = Document(
        {
            "doc_id": "id",
            "type": "pdf",
            "text_representation": "lkqwrg",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/filename.yolo", "title": "lkqwrg"},
            "elements": [TableElement(table=Table.from_html(bigtable))],
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

    def test_split_table(self):
        tk = CharacterTokenizer()
        doc = SplitElements(None, tk, 35).run(self.tabledoc)
        answers = {
            '<table><tr><th rowspan="2">headerA</th></tr><tr><td>data1a</td></tr><tr><td>data1b</td></tr><tr><td>data1c</td></tr><tr><td>data1d</td></tr></table>',
            "<table><tr><th>headerB</th></tr><tr><th>headerD</th></tr><tr><td>data2a</td></tr><tr><td>data2b</td></tr></table>",
            "<table><tr><th>headerB</th></tr><tr><th>headerD</th></tr><tr><td>data2c</td></tr><tr><td>data2d</td></tr></table>",
            "<table><tr><th>headerB</th></tr><tr><th>headerE</th></tr><tr><td>data3a</td></tr><tr><td>data3b</td></tr></table>",
            "<table><tr><th>headerB</th></tr><tr><th>headerE</th></tr><tr><td>data3c</td></tr><tr><td>data3d</td></tr></table>",
            "<table><tr><th>headerC</th></tr><tr><th>headerF</th></tr><tr><td>data4a</td></tr><tr><td>data4b</td></tr></table>",
            "<table><tr><th>headerC</th></tr><tr><th>headerF</th></tr><tr><td>data4c</td></tr><tr><td>data4d</td></tr></table>",
        }
        assert {e.table.to_html() for e in doc.elements} == answers
