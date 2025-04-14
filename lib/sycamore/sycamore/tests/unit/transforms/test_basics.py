import pytest
from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.transforms import Limit
import ray


class MockNode(Node):
    def __init__(self, docs):
        self._docs = docs

    def execute(self, **kwargs):
        return ray.data.from_items([{"doc": doc.serialize()} for doc in self._docs])


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_limit():
    docs = [Document({"text": f"Doc {i}"}) for i in range(10)]
    node = MockNode(docs)
    limit_transform = Limit(node, limit=5)
    result = limit_transform.execute()
    assert len(result.take_all()) == 5


def test_limit_empty_dataset():
    node = MockNode([])
    limit_transform = Limit(node, limit=5)
    result = limit_transform.execute()
    assert len(result.take_all()) == 0


def test_limit_with_metadata():
    docs = [Document({"id": i, "text": f"Doc {i}"}) for i in range(3)]
    docs.extend([MetadataDocument({"id": i, "text": f"Doc {i}"}) for i in range(3, 6)])
    docs.extend([Document({"id": i, "text": f"Doc {i}"}) for i in range(6, 9)])
    node = MockNode(docs)
    limit_transform = Limit(node, limit=4)
    result = limit_transform.execute()
    list_of_docs = [Document.deserialize(d["doc"]) for d in result.take_all()]
    documents = [doc for doc in list_of_docs if not isinstance(doc, MetadataDocument)]
    assert len(list_of_docs) == 4
    assert len(documents) == 4
