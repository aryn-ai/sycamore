from typing import Callable, Optional, TYPE_CHECKING, Union

from sycamore.plan_nodes import UnaryNode, Node
from sycamore.data import Document, MetadataDocument
import numpy as np

from sycamore.utils.lineage_utils import update_lineage
from sycamore.utils.thread_local import ADD_METADATA_TO_OUTPUT, ThreadLocal

if TYPE_CHECKING:
    from ray.data import Dataset


class Aggregation(UnaryNode):
    def __init__(
        self,
        child: Optional[Node],
        name: str,
        accumulate_docs: Callable[[list[Document]], Document],
        combine_partials: Callable[[Document, Document], Document],
        finalize: Callable[[Document], Document],
        group_key_fn: Callable[[Document], str] = lambda d: "nogrouping",
        zero_factory: Callable[[], Document] = Document,
    ):
        super().__init__(child)
        self._name = name
        self._accumulate = accumulate_docs
        self._combine = combine_partials
        self._finalize = finalize
        self._group_key_fn = group_key_fn
        self._zero_factory = zero_factory

    def _to_key_val(self, row):
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            row["key"] = f"md-{doc.doc_id}"
        else:
            row["key"] = self._group_key_fn(doc)
        return row

    def _unpack(self, row):
        import pickle

        doc_n_meta = row[self._name]
        doc = doc_n_meta["doc"]
        meta = pickle.loads(doc_n_meta["meta"])
        return [{"doc": doc}] + [m.to_row() for m in meta]

    def execute(self, **kwargs) -> "Dataset":
        from ray.data.aggregate import AggregateFnV2
        import pyarrow as pa
        import pandas as pd
        import pickle

        dataset = self.child().execute()

        class RayAggregation(AggregateFnV2):
            def __init__(
                self,
                syc_agg: Aggregation,
                name: str,
                zero_factory: Callable[[], Document],
            ):
                # Idk why I couldn't do super().__init__(...)
                def real_zero_factory():
                    return {"doc": zero_factory(), "meta": pickle.dumps([])}

                AggregateFnV2.__init__(self, name, real_zero_factory, on=None, ignore_nulls=True)
                self._syc_agg = syc_agg

            def aggregate_block(self, block: Union[pa.Table, pd.DataFrame]):
                docs = [
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
                    partial_result = self._syc_agg._accumulate(docs)
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
                    combined = self._syc_agg._combine(doc1, doc2)
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
                    final_doc = self._syc_agg._finalize(doc)
                final_doc["key"] = accumulator["key"]
                meta.extend(update_lineage(from_docs=[doc], to_docs=[final_doc]))
                meta.extend(extra_metadata)
                return {"doc": final_doc.serialize(), "meta": pickle.dumps(meta)}

        ray_agg = RayAggregation(self, self._name, self._zero_factory)
        ds = dataset.map(self._to_key_val).groupby("key").aggregate(ray_agg).flat_map(self._unpack)
        return ds

    def local_execute(self, all_docs: list[Document], do_combine: bool = True) -> list[Document]:
        import random

        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        documents = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        split_docs: dict[str, list] = {}

        for d in documents:
            key = self._group_key_fn(d)
            if (split := split_docs.get(key, None)) is not None:
                split.append(d)
            else:
                split_docs[key] = [d]

        ret: list[Union[Document, MetadataDocument]] = metadata  # type: ignore
        extra_metadata: list[MetadataDocument] = []
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
            for key, split in split_docs.items():
                if do_combine and len(split) > 1:
                    # This path is mostly for testing. Otherwise combine is not exercised
                    # in local mode. We also combine in a random order because if
                    # we're testing we're trying to break your assumptions.
                    documents_beginning = split[: len(split) // 2]
                    documents_end = split[len(split) // 2 :]
                    partial_beginning = self._accumulate(documents_beginning)
                    ret.extend(update_lineage(from_docs=documents_beginning, to_docs=[partial_beginning]))
                    partial_end = self._accumulate(documents_end)
                    ret.extend(update_lineage(from_docs=documents_end, to_docs=[partial_end]))
                    if random.random() < 0.5:
                        partial_beginning, partial_end = partial_end, partial_beginning
                    partial = self._combine(partial_beginning, partial_end)
                    ret.extend(update_lineage(from_docs=[partial_beginning, partial_end], to_docs=[partial]))
                else:
                    partial = self._accumulate(split)
                    ret.extend(update_lineage(from_docs=split, to_docs=[partial]))
                final = self._finalize(partial)
                ret.extend(update_lineage(from_docs=[partial], to_docs=[final]))
                ret.append(final)
        ret.extend(extra_metadata)
        return ret


class AggBuilder:
    def __init__(
        self,
        name: str,
        accumulate_docs: Callable[[list[Document]], Document],
        combine_partials: Callable[[Document, Document], Document],
        finalize: Callable[[Document], Document],
        zero_factory: Callable[[], Document] = Document,
    ):
        self._name = name
        self._accumulate = accumulate_docs
        self._combine = combine_partials
        self._finalize = finalize
        self._zero_factory = zero_factory

    def build(self, child: Optional[Node]) -> Aggregation:
        return Aggregation(
            child, self._name, self._accumulate, self._combine, self._finalize, zero_factory=self._zero_factory
        )

    def build_grouped(self, child: Optional[Node], group_key_fn: Callable[[Document], str]) -> Aggregation:
        return Aggregation(
            child,
            self._name,
            self._accumulate,
            self._combine,
            self._finalize,
            group_key_fn=group_key_fn,
            zero_factory=self._zero_factory,
        )


class Reduce(UnaryNode):

    def __init__(
        self,
        child: Optional[Node],
        reduce_fn: Callable[[list[Document]], Document],
        group_key_fn: Callable[[Document], str] = lambda d: "nogrouping",
    ):
        super().__init__(child)
        self._reduce_fn = reduce_fn
        self._group_key_fn = group_key_fn

    def _to_key_val(self, row):
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            row["key"] = f"md-{doc.doc_id}"
        else:
            row["key"] = self._group_key_fn(doc)
        return row

    def _group_reduce_ray(self, block: dict[str, np.ndarray]):
        key = block["key"][0]
        assert isinstance(key, str)
        if key.startswith("md-"):
            return block
        docs = [Document.deserialize(d) for d in block["doc"]]
        extra_metadata: list[MetadataDocument] = []
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
            reduced = self._reduce_fn(docs)
        extra_metadata.extend(update_lineage(from_docs=docs, to_docs=[reduced]))

        return {"doc": [reduced.serialize()] + [m.serialize() for m in extra_metadata]}

    def execute(self, **kwargs) -> "Dataset":
        dataset = self.child().execute()

        return dataset.map(self._to_key_val).groupby("key").map_groups(self._group_reduce_ray, batch_format="numpy")

    def local_execute(self, all_docs: list[Document]) -> list[Document]:

        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        documents = [d for d in all_docs if not isinstance(d, MetadataDocument)]

        split_docs = {}
        for d in documents:
            key = self._group_key_fn(d)
            if (split := split_docs.get(key, None)) is not None:
                split.append(d)
            else:
                split_docs[key] = [d]

        ret: list[Union[Document, MetadataDocument]] = metadata  # type: ignore
        extra_metadata: list[MetadataDocument] = []
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, extra_metadata):
            for key, split in split_docs.items():
                reduced = self._reduce_fn(split)
                ret.append(reduced)
                ret.extend(update_lineage(from_docs=split, to_docs=[reduced]))
        ret.extend(extra_metadata)
        return ret
