import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms import SpreadProperties


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestSpreadProperties:
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
            ]
        },
    }

    def test_spread_properties(self):
        doc0 = Document(self.dict0)
        sp = SpreadProperties(None, [])
        spc = sp.SpreadPropertiesCallable(["path", "title"])
        doc1 = spc.spreadProperties(doc0)
        for elem in doc1.elements:
            assert elem.properties["filetype"] == "text/plain"
            assert elem.properties["path"] == "/docs/foo.txt"
            assert elem.properties["title"] == "bar"

    def test_via_execute(self):
        plan = FakeNode(self.dict0)
        sp = SpreadProperties(plan, ["path", "title"])
        ds = sp.execute()
        for row in ds.iter_rows():
            doc = Document(row)
            for elem in doc.elements:
                assert elem.properties["filetype"] == "text/plain"
                assert elem.properties["path"] == "/docs/foo.txt"
                assert elem.properties["title"] == "bar"
