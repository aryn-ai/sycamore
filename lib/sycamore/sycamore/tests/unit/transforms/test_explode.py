import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms import Explode
from sycamore.transforms.explode import UnRoll


class TestExplode:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "content": {"binary": None, "text": "text"},
            "parent_id": None,
            "properties": {"path": "s3://path"},
            "embedding": {"binary": None, "text": None},
            "elements": [
                {
                    "type": "title",
                    "content": {"binary": None, "text": "text1"},
                    "properties": {"coordinates": [(1, 2)], "page_number": 1},
                },
                {
                    "type": "table",
                    "content": {"binary": None, "text": "text2"},
                    "properties": {"page_name": "name", "coordinates": [(1, 2)], "coordinate_system": "pixel"},
                },
            ],
        }
    )

    def test_explode_callable(self):
        docs = Explode(None).run(self.doc)
        assert len(docs) == 3

    def test_explode(self, mocker):
        node = mocker.Mock(spec=Node)
        explode = Explode(node)
        input_dataset = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        input_dataset.show()
        output_dataset = explode.execute()
        output_dataset.show()


class TestUnroll:
    def test_unroll(self):
        doc = Document()
        doc.properties["entities"] = "e1\ne2\ne3"
        docs = UnRoll(None, field="properties.entities").run(doc)
        assert len(docs) == 3
