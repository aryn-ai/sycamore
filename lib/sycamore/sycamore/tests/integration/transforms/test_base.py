import sycamore
import logging
import math
import time
import uuid
from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.map import Map


def make_docs(num):
    docs = []
    for i in range(num):
        doc = Document({"doc_id": f"doc_{i}"})
        docs.append(doc)

    return docs


ctx = sycamore.init()


def test_composite_transform_parallelism():
    class AgentMark:
        def __init__(self, key):
            self.key = key
            self.id = uuid.uuid4()
            logging.error(f"Start AgentMark {self.id}")

        def __call__(self, d):
            logging.error(f"Call AgentMark {self.id} on {d.doc_id}")
            time.sleep(1)
            d.properties[self.key] = self.id
            return d

    class AgentMark1(AgentMark):
        def __init__(self):
            super().__init__("op1")

    class AgentMark2(AgentMark):
        def __init__(self):
            super().__init__("op2")

    num_actors = 4
    num_docs = 20

    ops = [
        {
            "f": Map.wrap(AgentMark1),
            "parallelism": num_actors,
        },
        {
            "f": Map.wrap(AgentMark2),
            "parallelism": int(num_actors / 2),
        },
    ]
    ds = ctx.read.document(make_docs(num_docs))
    ds = DocSet(ctx, CompositeTransform(ds.plan, ops))
    docs = ds.take()

    op1_count = {}
    op2_count = {}
    for d in docs:
        a = d.properties["op1"]
        op1_count[a] = op1_count.get(a, 0) + 1
        a = d.properties["op2"]
        op2_count[a] = op2_count.get(a, 0) + 1

    assert len(op1_count) == num_actors
    assert len(op2_count) == num_actors / 2
    # Provide +-1 slop on perfectly even distribution.
    # given the sleep we probably will get perfect distribution
    min_count = math.floor(num_docs / num_actors - 1)
    max_count = math.ceil(num_docs / num_actors + 1)
    print("Expecting count to be between {min_count} and {max_count}")
    for a in op1_count:
        print(f"OP1 Actor {a} got {op1_count[a]} items")
        assert op1_count[a] >= min_count
        assert op1_count[a] <= max_count

    for a in op2_count:
        print(f"OP2 Actor {a} got {op2_count[a]} items")
        assert op2_count[a] >= 2 * min_count
        assert op2_count[a] <= 2 * max_count
