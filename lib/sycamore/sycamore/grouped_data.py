from typing import TYPE_CHECKING, Callable

from sycamore import DocSet
from sycamore.data import Document, MetadataDocument
from sycamore.transforms.aggregation import Aggregation, Reduce
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Transform, Node

if TYPE_CHECKING:
    from ray.data import Dataset


def filter_meta(row):
    doc = Document.from_row(row)
    return not isinstance(doc, MetadataDocument)


class AggregateCount(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, key: str):
        super().__init__(child)
        self._grouped_key = key

    def group_udf(self, batch):
        import numpy as np

        result = {"count": np.array([len(batch["properties"])])}
        key = batch[self._grouped_key][0]
        result["key"] = np.array([key])
        return result

    def to_doc(self, row: dict):
        count = row.pop("count")
        key = row.pop("key") if "key" in row else None
        doc = Document(row)
        properties = doc.properties
        properties["count"] = count
        if key:
            properties["key"] = key
        doc.properties = properties
        return doc.to_row()

    def execute(self, **kwargs) -> "Dataset":
        dataset = self.child().execute()
        grouped = dataset.filter(filter_meta).map(Document.from_row).groupby(self._grouped_key)
        aggregated = grouped.map_groups(self.group_udf)
        serialized = aggregated.map(self.to_doc)
        return serialized


class AggregateCollect(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, key: str, entity: str):
        super().__init__(child)
        self._grouped_key = key
        self._entity = entity

    def group_udf(self, batch):
        import numpy as np

        result = {}
        key = batch[self._grouped_key][0]
        result["key"] = np.array([key])
        if self._entity:
            names = self._entity.split(".")
            base = batch[names[0]]
            entities = []
            for row in base:
                for name in names[1:]:
                    row = row[name]
                entities.append(row)

            result["values"] = np.array([", ".join(str(e) for e in entities if e is not None)])
        return result

    def to_doc(self, row: dict):
        values = row.pop("values")
        key = row.pop("key")
        doc = Document(row)
        properties = doc.properties
        properties["values"] = values
        properties["key"] = key
        doc.properties = properties
        return doc.to_row()

    def execute(self, **kwargs) -> "Dataset":
        dataset = self.child().execute()
        grouped = dataset.filter(filter_meta).map(Document.from_row).groupby(self._grouped_key)
        aggregated = grouped.map_groups(self.group_udf)
        serialized = aggregated.map(self.to_doc)
        return serialized


class GroupedData:
    def __init__(self, docset: DocSet, grouped_key, entity=None):
        self._docset = docset
        self._grouped_key = grouped_key
        self._entity = entity

    def count(self) -> DocSet:
        return DocSet(self._docset.context, AggregateCount(self._docset.plan, self._grouped_key))

    def collect(self) -> DocSet:
        return DocSet(self._docset.context, AggregateCollect(self._docset.plan, self._grouped_key, self._entity))

    def aggregate(self, agg: Aggregation) -> DocSet:
        def group_key_fn(doc: Document) -> str:
            return str(doc.field_to_value(self._grouped_key))

        aggregation = agg.build_grouped(self._docset.plan, group_key_fn)
        return DocSet(self._docset.context, aggregation)

    def reduce(self, reduce_fn: Callable[[list[Document]], Document]) -> DocSet:
        def group_key_fn(doc: Document) -> str:
            return str(doc.field_to_value(self._grouped_key))

        reduction = Reduce(self._docset.plan, reduce_fn, group_key_fn)
        return DocSet(self._docset.context, reduction)
