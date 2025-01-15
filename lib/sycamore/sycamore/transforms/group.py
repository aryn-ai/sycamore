from ray.data._internal.aggregate import Count

from sycamore import DocSet
from sycamore.data import Document


class GroupedData:
    def __init__(self, docset: DocSet, key):
        self._docset = docset
        self._key = key

    def aggregate(self, f) -> DocSet:
        dataset = self._docset.plan.execute().map(Document.from_row)
        grouped = dataset.groupby(self._key)
        aggregated = grouped.aggregate(f)
        m = aggregated.take()

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
        return self.aggregate(Count())
