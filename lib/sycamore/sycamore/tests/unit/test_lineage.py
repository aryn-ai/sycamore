import uuid

import sycamore
from sycamore.context import ExecMode
from sycamore.data import Document, MetadataDocument


class TestLineage:
    def make_docs(self, num):
        docs = []
        for i in range(num):
            doc = Document({"doc_id": f"doc_{i}"})
            docs.append(doc)

        docs.append(
            MetadataDocument(
                lineage_links={"from_ids": ["root:" + str(uuid.uuid4())], "to_ids": [d.lineage_id for d in docs]}
            )
        )

        return docs

    @staticmethod
    def noop_fn(d):
        return d

    def test_simple(self):
        ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
        ctx.read.document(self.make_docs(3)).map(self.noop_fn).materialize().show()
