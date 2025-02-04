from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data.aggregate import AggregateFn

from sycamore import DocSet
from sycamore.data import Document, MetadataDocument


class GroupedData:
    def __init__(self, docset: DocSet, key):
        self._docset = docset
        self._key = key

    def aggregate(self, f: "AggregateFn") -> DocSet:
        dataset = self._docset.plan.execute()

        def filter_meta(row):
            doc = Document.from_row(row)
            return not isinstance(doc, MetadataDocument)

        grouped = dataset.filter(filter_meta).map(Document.from_row).groupby(self._key)
        aggregated = grouped.aggregate(f)

        def to_doc(row: dict):
            count = row.pop("count()")
            doc = Document(row)
            properties = doc.properties
            properties["count"] = count
            doc.properties = properties
            return doc.to_row()

        serialized = aggregated.map(to_doc)
        from sycamore.transforms import DatasetScan

        return DocSet(self._docset.context, DatasetScan(serialized))

    def count(self) -> DocSet:
        from ray.data._internal.aggregate import Count

        return self.aggregate(Count())
