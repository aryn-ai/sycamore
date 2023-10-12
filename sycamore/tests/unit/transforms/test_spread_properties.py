from sycamore.data import Document
from sycamore.transforms import SpreadProperties


class TestSpreadProperties:
    def test_spread_properties(self):
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
        doc0 = Document(dict0)

        sp = SpreadProperties(None, [])
        spc = sp.SpreadPropertiesCallable(["path", "title"])
        doc1 = spc.spreadProperties(doc0)
        for elem in doc1.elements:
            assert elem.properties["filetype"] == "text/plain"
            assert elem.properties["path"] == "/docs/foo.txt"
            assert elem.properties["title"] == "bar"
