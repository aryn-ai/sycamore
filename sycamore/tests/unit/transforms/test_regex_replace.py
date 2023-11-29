import ray.data

from sycamore.data import Document, Element
from sycamore.transforms.regex_replace import RegexReplace
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestRegexReplace:
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
                    "text_representation": " \t foo   \t\t\n\t \t\n\n \t   bar\t  ",  # noqa: E501
                },
            ],
        }
    )

    def test_regex_replace(self):
        rr = RegexReplace(None, [])
        obj = rr.Callable([(r"\s+", " "), (r"^ ", ""), (r" $", "")])
        doc = obj.run(self.doc)
        elems = doc.elements
        self.validateElems(elems)

    def test_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        rr = RegexReplace(node, [(r"\s+", " "), (r"^ ", ""), (r" $", "")])
        in_ds = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = in_ds
        ds = rr.execute()
        doc = Document.from_row(ds.take(limit=1)[0])
        elems = doc.elements
        self.validateElems(elems)

    def validateElems(self, elems: list[Element]):
        assert len(elems) == 1
        assert elems[0].text_representation == "foo bar"
