import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms import SpreadProperties
from sycamore.transforms.base import take_separate


class TestSpreadProperties:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
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
            ],
        }
    )

    def test_spread_properties(self):
        doc1 = SpreadProperties(None, ["path", "title"]).run(self.doc)
        for elem in doc1.elements:
            assert elem.properties["filetype"] == "text/plain"
            assert elem.properties["path"] == "/docs/foo.txt"
            assert elem.properties["title"] == "bar"

    def test_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        sp = SpreadProperties(node, ["path", "title"])
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        ds = sp.execute()
        (docs, _) = take_separate(ds)
        assert len(docs) == 1
        for doc in docs:
            for elem in doc.elements:
                assert elem.properties["filetype"] == "text/plain"
                assert elem.properties["path"] == "/docs/foo.txt"
                assert elem.properties["title"] == "bar"

    def test_spread_dict_copies_dict(self):
        doc = self.doc.copy()
        doc.properties["copyme"] = {"key": "value"}
        sdoc = SpreadProperties(None, ["copyme"]).run(doc)
        sdoc.elements[0].properties["copyme"]["key"] = "othervalue"
        assert sdoc.properties["copyme"]["key"] == "value"
        assert sdoc.elements[0].properties["copyme"]["key"] == "othervalue"
        assert sdoc.elements[1].properties["copyme"]["key"] == "value"
