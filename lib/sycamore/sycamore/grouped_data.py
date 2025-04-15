from sycamore import DocSet
from sycamore.data import Document, MetadataDocument


class GroupedData:
    def __init__(self, docset: DocSet, grouped_key, entity=None):
        self._docset = docset
        self._grouped_key = grouped_key
        self._entity = entity

    def filter_meta(self, row):
        doc = Document.from_row(row)
        return not isinstance(doc, MetadataDocument)

    def count(self) -> DocSet:
        dataset = self._docset.plan.execute()

        def group_udf(batch):
            import numpy as np

            result = {"count": np.array([len(batch["properties"])])}
            if self._entity:
                names = self._entity.split(".")
                base = batch[names[0]][0]
                names = names[1:]
                while names:
                    base = base[names[0]]
                    names = names[1:]
                result["key"] = np.array([base])
            return result

        grouped = dataset.filter(self.filter_meta).map(Document.from_row).groupby(self._grouped_key)
        aggregated = grouped.map_groups(group_udf)

        def to_doc(row: dict):
            count = row.pop("count")
            key = row.pop("key") if "key" in row else None
            doc = Document(row)
            properties = doc.properties
            properties["count"] = count
            if key:
                properties["key"] = key
            doc.properties = properties
            return doc.to_row()

        serialized = aggregated.map(to_doc)
        from sycamore.transforms import DatasetScan

        return DocSet(self._docset.context, DatasetScan(serialized))

    def collect(self) -> DocSet:
        dataset = self._docset.plan.execute()

        def group_udf(batch):
            import numpy as np

            result = {"count": np.array([len(batch["properties"])])}
            if self._entity:
                names = self._entity.split(".")
                base = batch[names[0]]
                entities = []
                for row in base:
                    for name in names[1:]:
                        row = row[name]
                    entities.append(row)

                result["key"] = np.array([", ".join(str(e) for e in entities if e is not None)])
            return result

        grouped = dataset.filter(self.filter_meta).map(Document.from_row).groupby(self._grouped_key)
        aggregated = grouped.map_groups(group_udf)

        def to_doc(row: dict):
            count = row.pop("count")
            key = row.pop("key") if "key" in row else None
            doc = Document(row)
            properties = doc.properties
            properties["count"] = count
            if key:
                properties["key"] = key
            doc.properties = properties
            return doc.to_row()

        serialized = aggregated.map(to_doc)
        from sycamore.transforms import DatasetScan

        return DocSet(self._docset.context, DatasetScan(serialized))
