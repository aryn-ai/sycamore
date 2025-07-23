from typing import Callable
import pytest

import sycamore
from sycamore.data import Document
from sycamore.data.document import split_data_metadata
from sycamore.transforms.aggregation import AggBuilder

from sycamore.tests.unit.transforms.test_aggregation import Common, assert_lineage


@pytest.fixture(scope="module")
def docset():
    return sycamore.init(exec_mode=sycamore.EXEC_RAY).read.document(Common.docs)


class TestAggregation:
    @staticmethod
    def sum_aggregation() -> AggBuilder:

        def accumulate(docs: list[Document]) -> Document:
            d = Document(doc_id=".".join([doc.doc_id or "" for doc in docs]))
            d.properties["sum"] = sum([doc.properties["value"] for doc in docs])
            return d

        def combine(doc1: Document, doc2: Document) -> Document:
            d = Document(
                doc_id=f"{doc1.doc_id}.{doc2.doc_id}",
                properties={"sum": doc1.properties["sum"] + doc2.properties["sum"]},
            )
            return d

        def finalize(doc: Document) -> Document:
            return doc

        return AggBuilder(name="test_sum", accumulate_docs=accumulate, combine_partials=combine, finalize=finalize)

    def test_aggregation_no_grouping(self, docset):
        agg = self.sum_aggregation()
        out_docs = docset.aggregate(agg).take_all(include_metadata=True)

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 1
        assert real_out[0].properties["sum"] == 21

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs, to_docs=real_out[0], metadata=meta_out)

    def test_aggregation_gouping(self, docset):
        agg = self.sum_aggregation()
        out_docs = docset.groupby("properties.key").aggregate(agg).take_all(include_metadata=True)

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 3
        assert all(d.doc_id is not None for d in real_out)
        real_out.sort(key=lambda doc: doc.doc_id or "Unreachable, type narrowing")
        for doc in real_out:
            assert doc.doc_id is not None, "Unreachable, type narrowing"
            doc_ids = set(doc.doc_id.split("."))
            if "0" in doc_ids:
                assert doc_ids == {"0", "1", "2", "3"}
                assert doc.properties["sum"] == 6
            if "4" in doc_ids:
                assert doc_ids == {"4", "5"}
                assert doc.properties["sum"] == 9
            if "6" in doc_ids:
                assert doc_ids == {"6"}
                assert doc.properties["sum"] == 6

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs[:4], to_docs=real_out[0], metadata=meta_out)
        assert_lineage(from_docs=Common.docs[4:6], to_docs=real_out[1], metadata=meta_out)
        assert_lineage(from_docs=[Common.docs[6]], to_docs=real_out[2], metadata=meta_out)


class TestReduce:
    @staticmethod
    def sum_reduction() -> Callable[[list[Document]], Document]:

        def sum_reduce(docs: list[Document]) -> Document:
            d = Document(doc_id=".".join([doc.doc_id or "" for doc in docs]))
            d.properties["sum"] = sum([doc.properties["value"] for doc in docs])
            return d

        return sum_reduce

    def test_reduction_no_grouping(self, docset):
        red_fn = self.sum_reduction()
        out_docs = docset.reduce(red_fn).take_all(include_metadata=True)

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 1
        assert real_out[0].properties["sum"] == 21

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs, to_docs=real_out[0], metadata=meta_out)

    def test_reduction_grouping(self, docset):
        red_fn = self.sum_reduction()
        out_docs = docset.groupby("properties.key").reduce(red_fn).take_all(include_metadata=True)

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 3
        real_out.sort(key=lambda d: d.doc_id or "Unreachable, Type Narrowing")

        assert real_out[0].doc_id is not None, "Unreachable, Type Narrowing"
        assert real_out[0].properties["sum"] == 6
        assert {"0", "1", "2", "3"} == set(real_out[0].doc_id.split("."))
        assert real_out[1].doc_id is not None, "Unreachable, Type Narrowing"
        assert real_out[1].properties["sum"] == 9
        assert {"4", "5"} == set(real_out[1].doc_id.split("."))
        assert real_out[2].doc_id is not None, "Unreachable, Type Narrowing. grr."
        assert real_out[2].properties["sum"] == 6
        assert {"6"} == set(real_out[2].doc_id.split("."))

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs[:4], to_docs=real_out[0], metadata=meta_out)
        assert_lineage(from_docs=Common.docs[4:6], to_docs=real_out[1], metadata=meta_out)
        assert_lineage(from_docs=[Common.docs[6]], to_docs=real_out[2], metadata=meta_out)
