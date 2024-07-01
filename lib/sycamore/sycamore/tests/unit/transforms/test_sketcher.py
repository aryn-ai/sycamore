import ray
import ray.data

from sycamore.data import Document
from sycamore.transforms.sketcher import Sketcher, SketchUniquify
from sycamore.plan_nodes import Node
from sycamore.transforms.base import take_separate


def tearDownModule():
    ray.shutdown()


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self, **kwargs) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestSketcher:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "The quick brown fox jumps over the lazy dog.",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [],
        }
    )

    def test_sketch(self):
        shingles = Sketcher(None, window=32, number=8).run(self.doc).shingles
        self.validateShingles(shingles)

    def test_sketch_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        sk = Sketcher(node, window=32, number=8)
        in_ds = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = in_ds
        ds = sk.execute()
        (docs, _) = take_separate(ds)
        shingles = docs[0].shingles
        self.validateShingles(shingles)

    def validateShingles(self, shingles: list[int]):
        assert len(shingles) == 8
        assert min(shingles) > 0  # generally true


class TestSketchUniquify:
    doc0 = Document(
        {
            "doc_id": "doc0",
            "type": "pdf",
            "text_representation": "The quick brown fox jumps over the lazy dog.",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/bar.txt", "title": "foo"},
            "elements": [],
            "shingles": [
                0x1111111111111111,
                0x2222222222222222,
                0x3333333333333333,
                0x4444444444444444,
                0x5555555555555555,
                0x6666666666666666,
                0x7777777777777777,
                0x8888888888888888,
            ],
        }
    )
    doc1 = Document(
        {
            "doc_id": "doc1",
            "type": "pdf",
            "text_representation": "The quick brown fox jumps ever the lazy dog.",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/bar.txt", "title": "foo"},
            "elements": [],
            "shingles": [
                0x0000000000000001,
                0x0000000000000002,
                0x0000000000000003,
                0x4444444444444444,
                0x5555555555555555,
                0x6666666666666666,
                0x7777777777777777,
                0x8888888888888888,
            ],
        }
    )

    def test_dedup_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        uq = SketchUniquify(node, threshold=0.4)
        in_ds = ray.data.from_items(
            [
                {"doc": self.doc0.serialize()},
                {"doc": self.doc1.serialize()},
            ]
        )
        execute = mocker.patch.object(node, "execute")
        execute.return_value = in_ds
        ds = uq.execute()
        (docs, _) = take_separate(ds)
        assert len(docs) == 1
        assert docs[0].doc_id == "doc0"

    def test_cleanup(self):
        ray.shutdown()
