import ray.data

from sycamore.data import Document
from sycamore.transforms.merge_elements import Merge
from sycamore.transforms.split_elements import SplitElements
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestSplitElements:
    doc = Document({
        "doc_id": "doc_id",
        "type": "pdf",
        "text_representation": "foobar",
        "binary_representation": None,
        "parent_id": None,
        "properties": {"path": "/docs/foo.txt", "title": "bar"},
        "elements": [
            {
                "type": "UncategorizedText",
                "text_representation": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour; twentyfive, twentysix. twentyseven twentyeight, twentynine; thirty thirtyone thirtytwo thirtythree thirtyfour thirtyfive thirtysix thirtyseven thirtyeight thirtynine forty fortyone fortytwo fortythree fortyfour fortyfive fortysix fortyseven fortyeight fortynine"
            },
        ]
    })

    def test_split_elements(self):
        tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")
        sp = SplitElements(None, None, 0)
        spc = sp.Callable(tokenizer, 15)
        doc = spc.run(self.doc)
        elems = doc.elements
        assert len(elems) == 9
        assert elems[0].text_representation == "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        assert elems[1].text_representation == "sixteen seventeen eighteen nineteen twenty "
        assert elems[2].text_representation == "twentyone twentytwo twentythree twentyfour;"
        assert elems[3].text_representation == " twentyfive, twentysix."
        assert elems[4].text_representation == " twentyseven twentyeight, twentynine;"
        assert elems[5].text_representation == " thirty thirtyone thirtytwo thirtythree thirtyfour "
        assert elems[6].text_representation == "thirtyfive thirtysix thirtyseven thirtyeight thirtynine "
        assert elems[7].text_representation == "forty fortyone fortytwo fortythree fortyfour "
        assert elems[8].text_representation == "fortyfive fortysix fortyseven fortyeight fortynine"

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
        assert len(elems) == 9
        assert elems[0].text_representation == "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        assert elems[1].text_representation == "sixteen seventeen eighteen nineteen twenty "
        assert elems[2].text_representation == "twentyone twentytwo twentythree twentyfour;"
        assert elems[3].text_representation == " twentyfive, twentysix."
        assert elems[4].text_representation == " twentyseven twentyeight, twentynine;"
        assert elems[5].text_representation == " thirty thirtyone thirtytwo thirtythree thirtyfour "
        assert elems[6].text_representation == "thirtyfive thirtysix thirtyseven thirtyeight thirtynine "
        assert elems[7].text_representation == "forty fortyone fortytwo fortythree fortyfour "
        assert elems[8].text_representation == "fortyfive fortysix fortyseven fortyeight fortynine"
