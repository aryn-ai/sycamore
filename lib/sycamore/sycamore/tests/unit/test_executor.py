import sycamore
from sycamore.docset import DocSet
from sycamore.data.document import Document
from sycamore.plan_nodes import Node


def test_finalize():
    finalize = set()

    class Finalize(Node):
        def __init__(self, children, name):
            super().__init__(children)
            self.name = name

        def execute(self, **kwargs):
            assert False

        def local_source(self):
            return [Document(), Document(), Document()]

        def local_execute(self, docs):
            return docs

        def finalize(self):
            assert self.name not in finalize
            finalize.add(self.name)

    plan = Finalize([], "source")
    for i in range(5):
        plan = Finalize([plan], f"step{i}")

    context = sycamore.init(exec_mode=sycamore.ExecMode.LOCAL)
    docset = DocSet(context, plan)

    docs = docset.take_all()
    assert len(docs) == 3
    assert "source" in finalize
    for i in range(5):
        assert f"step{i}" in finalize

    # 4 consumes the 3 documents and the "EOF"
    finalize = set()
    docs = docset.take(limit=4)
    assert len(docs) == 3
    assert "source" in finalize
    for i in range(5):
        assert f"step{i}" in finalize

    # 3 consumes
    finalize = set()
    docs = docset.take(limit=3)
    assert len(docs) == 3
    assert len(finalize) == 0
