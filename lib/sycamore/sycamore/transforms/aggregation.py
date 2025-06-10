from typing import Callable, Optional, TYPE_CHECKING, Union

from sycamore.plan_nodes import UnaryNode, Node
from sycamore.data import Document, MetadataDocument

if TYPE_CHECKING:
    from ray.data import Dataset
    import pyarrow as pa
    import pandas as pd


class Aggregation(UnaryNode):
    def __init__(
        self,
        child: Optional[Node],
        name: str,
        accumulate_docs: Callable[[list[Document]], Document],
        combine_partials: Callable[[Document, Document], Document],
        finalize: Callable[[Document], Document],
    ):
        super().__init__(child)
        self._name = name
        self._accumulate = accumulate_docs
        self._combine = combine_partials
        self._finalize = finalize

    def _to_key_val(self, row):
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            row["key"] = f"md-{doc.doc_id}"
        else:
            row["key"] = "nogrouping"
        return row

    def execute(self, **kwargs) -> "Dataset":
        from ray.data import Dataset
        from ray.data.aggregate import AggregateFnV2

        dataset = self.child().execute()

        class RayAggregation(AggregateFnV2):
            def __init__(
                self,
                syc_agg: Aggregation,
                name: str,
                zero_factory: Callable[[], Document] = Document,
            ):
                super().__init__(self, name, zero_factory)
                self._syc_agg = syc_agg

            def aggregate_block(self, block: Union[pa.Table, pd.DataFrame]):
                docs = [Document.deserialize(dbytes) for dbytes in block["doc"]]
                if all(isinstance(d, MetadataDocument) for d in docs):
                    assert len(docs) == 1, "Found multiple metadata documents in accumulate fn somehow"
                    return {"doc": docs[0].serialize()}
                assert not any(
                    isinstance(d, MetadataDocument) for d in docs
                ), "Found mixed accumuation between Documents and Metadata"
                partial_result = self._syc_agg._accumulate(docs)
                return {"doc": partial_result.serialize()}

            def combine(self, row1, row2):
                doc1 = Document.from_row(row1)
                doc2 = Document.from_row(row2)
                assert not isinstance(doc1, MetadataDocument), "Tried to combine metadata documents"
                assert not isinstance(doc2, MetadataDocument), "Tried to combine metadata documents"
                combined = self._syc_agg._combine(doc1, doc2)
                return {"doc": combined.serialize()}

            def finalize(self, row):
                doc = Document.from_row(row)
                if isinstance(doc, MetadataDocument):
                    return row
                final_doc = self._syc_agg._finalize(doc)
                return {"row": final_doc.serialize()}

        ray_agg = RayAggregation(self, self._name)
        return dataset.map(self._to_key_val).groupby("key").aggregate(ray_agg)


class AggBuilder:
    def __init__(
        self,
        name: str,
        accumulate_docs: Callable[[list[Document]], Document],
        combine_partials: Callable[[Document, Document], Document],
        finalize: Callable[[Document], Document],
    ):
        self._name = name
        self._accumulate = accumulate_docs
        self._combine = combine_partials
        self._finalize = finalize

    def build(self, child: Optional[Node]) -> Aggregation:
        return Aggregation(child, self._name, self._accumulate, self._combine, self._finalize)
