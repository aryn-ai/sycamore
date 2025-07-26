from typing import Callable, Optional, TYPE_CHECKING, Union

from sycamore.data.document import split_data_metadata
from sycamore.plan_nodes import UnaryNode, Node
from sycamore.data import Document, MetadataDocument

from sycamore.utils.lineage_utils import update_lineage
from sycamore.utils.thread_local import ADD_METADATA_TO_OUTPUT, ThreadLocal

if TYPE_CHECKING:
    from ray.data import Dataset


class AggregationNode(UnaryNode):
    def __init__(
        self,
        child: Optional[Node],
        aggregation: "Aggregation",
        group_key_fn: Callable[[Document], str] = lambda d: "single_group",
    ):
        super().__init__(child)
        self._agg = aggregation
        self._group_key_fn = group_key_fn

    def _to_key_val(self, row):
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            row["key"] = f"md-{doc.doc_id}"
        else:
            row["key"] = self._group_key_fn(doc)
        return row

    def _unpack(self, row):
        import pickle

        doc_n_meta = row[self._agg._name]
        doc = doc_n_meta["doc"]
        meta = pickle.loads(doc_n_meta["meta"])
        return [{"doc": doc}] + [m.to_row() for m in meta]

    def execute(self, **kwargs) -> "Dataset":
        from ray.data.aggregate import AggregateFnV2
        import pyarrow
        import pandas
        import pickle

        dataset = self.child().execute()

        class RayAggregation(AggregateFnV2):
            def __init__(
                self,
                syc_agg: Aggregation,
            ):
                def real_zero_factory():
                    return {"doc": syc_agg.zero_factory(), "meta": pickle.dumps([])}

                self._syc_agg = syc_agg
                super().__init__(self._syc_agg._name, real_zero_factory, on=None, ignore_nulls=True)

            def aggregate_block(self, block: Union[pyarrow.Table, pandas.DataFrame]):
                docs = [
                    # I don't control how ray chooses to represent a block of data internally,
                    # so handle either case.
                    Document.deserialize(
                        dbytes.as_py() if hasattr(dbytes, "as_py") else dbytes
                    )  # ^^ if pyarrow BinaryScalar convert to python bytes
                    for dbytes in block["doc"]
                ]
                key = block["key"][0]
                if all(isinstance(d, MetadataDocument) for d in docs):
                    assert len(docs) == 1, "Found multiple metadata documents in accumulate fn somehow"
                    return {"doc": docs[0].serialize(), "key": key, "meta": pickle.dumps([])}
                assert not any(
                    isinstance(d, MetadataDocument) for d in docs
                ), "Found mixed accumuation between Documents and Metadata"
                extra_metadata: list[MetadataDocument] = []
                with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
                    partial_result = self._syc_agg.accumulate(docs)
                meta = update_lineage(from_docs=docs, to_docs=[partial_result])
                meta.extend(extra_metadata)
                return {"doc": partial_result.serialize(), "key": key, "meta": pickle.dumps(meta)}

            def combine(self, current_accumulator, new):
                assert current_accumulator["key"] == new["key"]
                row1 = {"doc": current_accumulator["doc"]}
                row2 = {"doc": new["doc"]}
                doc1 = Document.from_row(row1)
                doc2 = Document.from_row(row2)
                assert not isinstance(doc1, MetadataDocument), "Tried to combine metadata documents"
                assert not isinstance(doc2, MetadataDocument), "Tried to combine metadata documents"
                meta = pickle.loads(current_accumulator["meta"]) + pickle.loads(new["meta"])
                extra_metadata: list[MetadataDocument] = []
                with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
                    combined = self._syc_agg.combine(doc1, doc2)
                meta.extend(update_lineage(from_docs=[doc1, doc2], to_docs=[combined]))
                meta.extend(extra_metadata)

                return {"doc": combined.serialize(), "key": new["key"], "meta": pickle.dumps(meta)}

            def _finalize(self, accumulator):
                row = {"doc": accumulator["doc"]}
                doc = Document.from_row(row)
                meta = pickle.loads(accumulator["meta"])
                if isinstance(doc, MetadataDocument):
                    return accumulator
                extra_metadata: list[MetadataDocument] = []
                with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
                    final_doc = self._syc_agg.finalize(doc)
                final_doc["key"] = accumulator["key"]
                meta.extend(update_lineage(from_docs=[doc], to_docs=[final_doc]))
                meta.extend(extra_metadata)
                return {"doc": final_doc.serialize(), "meta": pickle.dumps(meta)}

        ray_agg = RayAggregation(self._agg)
        ds = dataset.map(self._to_key_val).groupby("key").aggregate(ray_agg).flat_map(self._unpack)
        return ds

    def local_execute(self, all_docs: list[Document], do_combine: bool = True) -> list[Document]:
        import random

        documents, metadata = split_data_metadata(all_docs)
        key_to_docs: dict[str, list] = {}

        for d in documents:
            key = self._group_key_fn(d)
            if (docs := key_to_docs.get(key, None)) is not None:
                docs.append(d)
            else:
                key_to_docs[key] = [d]

        ret: list[Union[Document, MetadataDocument]] = metadata  # type: ignore
        extra_metadata: list[MetadataDocument] = []
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
            for key, docs in key_to_docs.items():
                if do_combine and len(docs) > 1:
                    # This path is mostly for testing. Otherwise combine is not exercised in local mode.
                    # Determinism requires that combine is commutative, associative, and has a zero; and that accumulate is order independent.
                    # Non-crashiness requires it to handle those orders without failing.
                    # We check a subset of the conditions here and will expand over time.
                    # TODO: check that the result of combine(a, b) == combine(b, a);
                    #       that combine(a, combine(b, c)) = combine(combine(a, b), c);
                    #       and that combine(a, zero) == a ==  combine(zero, a)
                    # TODO: check that accumulate(docs) == accumulate(shuffle(docs))
                    documents_beginning = docs[: len(docs) // 2]
                    documents_end = docs[len(docs) // 2 :]
                    partial_beginning = self._agg.accumulate(documents_beginning)
                    ret.extend(update_lineage(from_docs=documents_beginning, to_docs=[partial_beginning]))
                    partial_end = self._agg.accumulate(documents_end)
                    ret.extend(update_lineage(from_docs=documents_end, to_docs=[partial_end]))
                    if random.random() < 0.5:
                        partial_beginning, partial_end = partial_end, partial_beginning
                    partial = self._agg.combine(partial_beginning, partial_end)
                    ret.extend(update_lineage(from_docs=[partial_beginning, partial_end], to_docs=[partial]))
                else:
                    partial = self._agg.accumulate(docs)
                    ret.extend(update_lineage(from_docs=docs, to_docs=[partial]))
                final = self._agg.finalize(partial)
                ret.extend(update_lineage(from_docs=[partial], to_docs=[final]))
                ret.append(final)
        ret.extend(extra_metadata)
        return ret


class Aggregation:
    def __init__(
        self,
        name: str,
        accumulate_docs: Optional[Callable[[list[Document]], Document]] = None,
        combine_partials: Optional[Callable[[Document, Document], Document]] = None,
        finalize: Optional[Callable[[Document], Document]] = None,
        zero_factory: Optional[Callable[[], Document]] = None,
    ):
        self._name = name
        self._accumulate = accumulate_docs
        self._combine = combine_partials
        self._finalize = finalize
        self._zero_factory = zero_factory

    # Syntax: the / in the param list tells python that docs is positional only.
    # this allows using Callable-typed arguments to override them through the constructor.
    def accumulate(self, docs: list[Document]) -> Document:
        if self._accumulate is not None:
            return self._accumulate(docs)
        raise NotImplementedError("accumulate is not implemented in base aggregation")

    def combine(self, doc1: Document, doc2: Document) -> Document:
        if self._combine is not None:
            return self._combine(doc1, doc2)
        raise NotImplementedError("combine is not implemented in base aggregation")

    def finalize(self, doc: Document) -> Document:
        if self._finalize is not None:
            return self._finalize(doc)
        return doc

    def zero_factory(self) -> Document:
        if self._zero_factory is not None:
            return self._zero_factory()
        return Document()

    def build(self, child: Optional[Node]) -> AggregationNode:
        return AggregationNode(child, self)

    def build_grouped(self, child: Optional[Node], group_key_fn: Callable[[Document], str]) -> AggregationNode:
        return AggregationNode(
            child,
            self,
            group_key_fn=group_key_fn,
        )


class Reduce(Aggregation):
    def __init__(
        self,
        reduce_fn: Callable[[list[Document]], Document],
    ):
        super().__init__(name="reduce")
        self.reduce_fn = reduce_fn

    def accumulate(self, docs: list[Document]) -> Document:
        from sycamore.transforms.summarize import SummaryDocument

        return SummaryDocument(sub_docs=docs)

    def combine(self, doc1: Document, doc2: Document) -> Document:
        from sycamore.transforms.summarize import SummaryDocument

        assert isinstance(doc1, SummaryDocument)
        assert isinstance(doc2, SummaryDocument)
        doc1.sub_docs.extend(doc2.sub_docs)
        return doc1

    def finalize(self, doc: Document) -> Document:
        from sycamore.transforms.summarize import SummaryDocument

        assert isinstance(doc, SummaryDocument)
        doc.sub_docs.sort(key=lambda d: d.doc_id or "")
        return self.reduce_fn(doc.sub_docs)

    def zero_factory(self) -> Document:
        from sycamore.transforms.summarize import SummaryDocument

        return SummaryDocument()
