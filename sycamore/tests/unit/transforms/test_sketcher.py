import ray.data

from sycamore.data import Document
from sycamore.transforms.sketcher import Sketcher, SketchUniquify
from sycamore.plan_nodes import Node


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self) -> ray.data.Dataset:
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
        sk = Sketcher(None)
        obj = sk.Callable(window=32, courses=15, tabs=8)
        doc = obj.run(self.doc)
        sims = doc.simHashes
        self.validateSimHashes(sims)

    def test_sketch_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        sk = Sketcher(node, window=32, courses=15, tabs=8)
        in_ds = ray.data.from_items([{"doc": self.doc.serialize()}])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = in_ds
        ds = sk.execute()
        doc = Document.from_row(ds.take(limit=1)[0])
        sims = doc.simHashes
        self.validateSimHashes(sims)

    def validateSimHashes(self, sims: list[int]):
        assert len(sims) == 8
        assert min(sims) > 0  # generally true


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
            "simHashes": [
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
            "simHashes": [
                0x0101010101010101,  # 8  bit diff
                0x0202020202020202,  # 8  bit diff
                0x0303030303030303,  # 16 bit diff
                0x0404040404040404,  # 8  bit diff
                0x0505050505050505,  # 16 bit diff
                0x0606060606060606,  # 16 bit diff
                0x0707070707070707,  # 24 bit diff
                0x0808080808080808,  # 8  bit diff
            ],
        }
    )

    def test_dedup_via_execute(self, mocker):
        node = mocker.Mock(spec=Node)
        uq = SketchUniquify(node, threshold=16)
        in_ds = ray.data.from_items(
            [
                {"doc": self.doc0.serialize()},
                {"doc": self.doc1.serialize()},
            ]
        )
        execute = mocker.patch.object(node, "execute")
        execute.return_value = in_ds
        ds = uq.execute()
        assert ds.count() == 1
        doc = Document.from_row(ds.take()[0])
        assert doc.doc_id == "doc0"
