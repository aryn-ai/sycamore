import random

from sycamore.data import Document
from sycamore import DocSet
import sycamore


class TestUnion:
    @staticmethod
    def setup(exec_mode) -> tuple[list[DocSet], list[list[Document]]]:
        ctx = sycamore.init(exec_mode=exec_mode)
        docs_1 = [Document(doc_id="a"), Document(doc_id="b")]
        docs_2 = [Document(doc_id="c"), Document(doc_id="d")]
        docs_3 = [Document(doc_id="e"), Document(doc_id="f")]
        docses = [docs_1, docs_2, docs_3]
        random.shuffle(docses)
        dses = [ctx.read.document(docs) for docs in docses]
        random.shuffle(dses)
        return dses, docses

    def test_union_docset(self):
        dses, docses = self.setup(sycamore.EXEC_LOCAL)
        ds = dses[0].union(*dses[1:])
        out_docs = ds.take_all()
        assert set(od.doc_id for od in out_docs) == set("abcdef")
