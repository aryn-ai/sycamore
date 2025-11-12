import sycamore
import sycamore.tests.unit.transforms.test_union as unit


class TestUnion:
    def test_union_ray(self):
        dses, docses = unit.TestUnion.setup(sycamore.EXEC_RAY)
        ds = dses[0].union(*dses[1:])
        docs = ds.take_all()
        in_ids = set(doc.doc_id for docs in docses for doc in docs)
        out_ids = set(doc.doc_id for doc in docs)
        assert in_ids == out_ids
