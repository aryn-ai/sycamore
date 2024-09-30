import sycamore
import logging
import math
import time
import uuid
from sycamore.data import Document


def make_docs(num):
    docs = []
    for i in range(num):
        doc = Document({"doc_id": f"doc_{i}"})
        docs.append(doc)

    return docs


ctx = sycamore.init()


def test_map_class_parallelism():
    class AgentMark:
        def __init__(self):
            self.id = uuid.uuid4()
            logging.error("Start AgentMark {self.id}")

        def __call__(self, d):
            logging.error(f"Call AgentMark {self.id} on {d.doc_id}")
            time.sleep(1)
            d.properties["agent"] = self.id
            return d

    num_actors = 4
    num_docs = 20
    docs = ctx.read.document(make_docs(num_docs)).map(AgentMark, parallelism=num_actors).take()

    count = {}
    for d in docs:
        a = d.properties["agent"]
        count[a] = count.get(a, 0) + 1

    assert len(count) == num_actors
    # Provide +-1 slop on perfectly even distribution.
    # given the sleep we probably will get perfect distribution
    min_count = math.floor(num_docs / num_actors - 1)
    max_count = math.ceil(num_docs / num_actors + 1)
    print("Expecting count to be between {min_count} and {max_count}")
    for a in count:
        print(f"Actor {a} got {count[a]} items")
        assert count[a] >= min_count
        assert count[a] <= max_count
