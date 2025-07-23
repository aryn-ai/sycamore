from typing import Union, Callable

from sycamore.data import Document, MetadataDocument
from sycamore.data.document import split_data_metadata
from sycamore.transforms.aggregation import Aggregation, Reduce


class CallCounts:
    def __init__(self):
        self.acc_calls = 0
        self.comb_calls = 0
        self.fin_calls = 0


class Common:
    docs = [
        Document(doc_id="0", properties={"key": "a", "value": 0}),
        Document(doc_id="1", properties={"key": "a", "value": 1}),
        Document(doc_id="2", properties={"key": "a", "value": 2}),
        Document(doc_id="3", properties={"key": "a", "value": 3}),
        Document(doc_id="4", properties={"key": "b", "value": 4}),
        Document(doc_id="5", properties={"key": "b", "value": 5}),
        Document(doc_id="6", properties={"key": "c", "value": 6}),
        MetadataDocument(keep="me around"),
    ]


def construct_lineage_adjacency_list(metadata: list[MetadataDocument]) -> dict[str, list[str]]:
    lineage = [md.metadata["lineage_links"] for md in metadata if "lineage_links" in md.metadata]
    lineage_adjacency_list = {}
    for ln in lineage:
        if len(ln["to_ids"]) > 1:
            assert len(ln["to_ids"]) == len(ln["from_ids"])
            for to, fro in zip(ln["to_ids"], ln["from_ids"]):
                assert to not in lineage_adjacency_list
                lineage_adjacency_list[to] = [fro]
        else:
            to = ln["to_ids"][0]
            assert to not in lineage_adjacency_list
            lineage_adjacency_list[to] = ln["from_ids"]
    return lineage_adjacency_list


def assert_lineage(
    from_docs: list[Document], to_docs: Union[Document, list[Document]], metadata: list[MetadataDocument]
):
    if not isinstance(to_docs, list):
        to_docs = [to_docs]
    lineage_adjacency_list = construct_lineage_adjacency_list(metadata)
    lineage_leaves = set()
    to_check = [td.lineage_id for td in to_docs]
    while len(to_check) > 0:
        lid = to_check.pop()
        if lid in lineage_adjacency_list:
            to_check.extend(lineage_adjacency_list[lid])
        else:
            lineage_leaves.add(lid)

    assert {doc.lineage_id for doc in from_docs if not isinstance(doc, MetadataDocument)} == lineage_leaves


class TestAggregation:
    @staticmethod
    def sum_aggregation() -> tuple[Aggregation, CallCounts]:
        call_counts = CallCounts()

        def accumulate(docs: list[Document]) -> Document:
            call_counts.acc_calls += 1
            d = Document(doc_id=".".join([doc.doc_id or "" for doc in docs]))
            d.properties["sum"] = sum([doc.properties["value"] for doc in docs])
            return d

        def combine(doc1: Document, doc2: Document) -> Document:
            call_counts.comb_calls += 1
            d = Document(
                doc_id=f"{doc1.doc_id}.{doc2.doc_id}",
                properties={"sum": doc1.properties["sum"] + doc2.properties["sum"]},
            )
            return d

        def finalize(doc: Document) -> Document:
            call_counts.fin_calls += 1
            return doc

        return (
            Aggregation(name="test_sum", accumulate_docs=accumulate, combine_partials=combine, finalize=finalize),
            call_counts,
        )

    def test_aggregation_no_grouping(self):
        agg, counts = self.sum_aggregation()
        agg_node = agg.build(None)
        out_docs = agg_node.local_execute(Common.docs)

        assert counts.acc_calls > 0
        assert counts.comb_calls > 0
        assert counts.fin_calls > 0

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 1
        assert real_out[0].properties["sum"] == 21

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs, to_docs=real_out[0], metadata=meta_out)

    def test_aggregation_gouping(self):
        agg, counts = self.sum_aggregation()
        agg_node = agg.build_grouped(None, lambda d: d.properties["key"])
        out_docs = agg_node.local_execute(Common.docs)

        assert counts.acc_calls > 0
        assert counts.comb_calls > 0
        assert counts.fin_calls > 0

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 3
        real_out.sort(key=lambda doc: doc.doc_id)
        for doc in real_out:
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
    def sum_reduction() -> tuple[Callable[[list[Document]], Document], CallCounts]:
        call_counts = CallCounts()

        def sum_reduce(docs: list[Document]) -> Document:
            call_counts.acc_calls += 1
            d = Document(doc_id=".".join([doc.doc_id or "" for doc in docs]))
            d.properties["sum"] = sum([doc.properties["value"] for doc in docs])
            return d

        return sum_reduce, call_counts

    def test_reduction_no_grouping(self):
        red_fn, counts = self.sum_reduction()
        reduce = Reduce(None, red_fn)
        out_docs = reduce.local_execute(Common.docs)

        assert counts.acc_calls == 1

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 1
        assert real_out[0].properties["sum"] == 21

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs, to_docs=real_out[0], metadata=meta_out)

    def test_reduction_grouping(self):
        red_fn, counts = self.sum_reduction()
        reduce = Reduce(None, red_fn, group_key_fn=lambda d: d.properties["key"])
        out_docs = reduce.local_execute(Common.docs)

        assert counts.acc_calls == 3

        real_out, meta_out = split_data_metadata(out_docs)
        assert len(real_out) == 3
        real_out.sort(key=lambda d: d.doc_id)

        assert real_out[0].properties["sum"] == 6
        assert {"0", "1", "2", "3"} == set(real_out[0].doc_id.split("."))
        assert real_out[1].properties["sum"] == 9
        assert {"4", "5"} == set(real_out[1].doc_id.split("."))
        assert real_out[2].properties["sum"] == 6
        assert {"6"} == set(real_out[2].doc_id.split("."))

        assert any("keep" in md.metadata for md in meta_out)
        assert_lineage(from_docs=Common.docs[:4], to_docs=real_out[0], metadata=meta_out)
        assert_lineage(from_docs=Common.docs[4:6], to_docs=real_out[1], metadata=meta_out)
        assert_lineage(from_docs=[Common.docs[6]], to_docs=real_out[2], metadata=meta_out)
