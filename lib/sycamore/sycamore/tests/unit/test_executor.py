import unittest

import sycamore
from sycamore.context import ExecMode
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


class TestPrepare(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.LOCAL

    def test_prepare(self):
        from sycamore.plan_nodes import Node

        class PNode(Node):
            def __init__(self, target_count, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.count = 0
                self.target_count = target_count

            def prepare(self):
                count = self.count

                def loop():
                    assert self.count == count
                    self.count = self.count + 1
                    return self.prepare()

                if count < self.target_count:
                    return loop

            def local_source(self):
                return []

            def local_execute(self, docs):
                return []

            def execute(self):
                from sycamore.connectors.file import DocScan

                if len(self.children) > 0:
                    return self.children[0].execute()

                return DocScan([]).execute()

        a = PNode(children=[], target_count=7)
        b = PNode(children=[a], target_count=3)

        context = sycamore.init(exec_mode=self.exec_mode)
        docset = DocSet(context, b)
        docset.execute()

        assert a.count == 7
        assert b.count == 3
