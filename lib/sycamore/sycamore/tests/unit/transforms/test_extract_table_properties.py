from sycamore.data import Document
from sycamore.llms import OpenAI
from sycamore.transforms.extract_table_properties import ExtractTableProperties
from sycamore.data.table import Table, TableCell


class TestExtractTableProperties:
    def table(self, str1, str2) -> Table:
        return Table(
            [
                TableCell(content="head1", rows=[0], cols=[0], is_header=True),
                TableCell(content="head2", rows=[0], cols=[1], is_header=True),
                TableCell(content=str1, rows=[1], cols=[0], is_header=False),
                TableCell(content=str2, rows=[1], cols=[1], is_header=False),
            ]
        )

    def test_extract_key_value_pair(self, mocker):
        self.doc = Document(
            {
                "doc_id": "doc_id",
                "type": "pdf",
                "text_representation": "text_representation",
                "bbox": (1, 2.3, 3.4, 4.5),
                "elements": [
                    {
                        "type": "table",
                        "bbox": (1, 2, 3, 4.0),
                        "properties": {"title": {"rows": None, "columns": None}},
                        "table": self.table("key1", "val1"),
                        "tokens": None,
                    },
                ],
                "properties": {"int": 0, "float": 3.14, "list": [1, 2, 3, 4], "tuple": (1, "tuple")},
            }
        )
        # print(self.doc)
        llm = mocker.Mock(sepc=OpenAI)
        generate = mocker.patch.object(llm, "generate",side_effect=["True", {"llm_response": '{"key1":"val1"}'}])
        generate.return_value = {"llm_response": '{"key1":"val1"}'}
        doc1 = ExtractTableProperties(None, parameters=["llm_response", llm]).run(self.doc)
        print(doc1)
        assert (doc1.elements[0].properties.get("llm_response")) == {"llm_response": '{"key1":"val1"}'}
