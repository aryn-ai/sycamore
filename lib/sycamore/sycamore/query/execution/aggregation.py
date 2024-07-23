from collections.abc import Callable
from typing import Any, Optional

from ray.data import Dataset
from ray.data.aggregate import AggregateFn
from ray.data.grouped_data import GroupedData

from sycamore.data import Document
from sycamore.docset import DocSet
from sycamore.plan_nodes import Scan


# Todo support multiple fields
def make_aggregation_map_fn(field: str, default_val: Any):
    def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
        doc = Document.from_row(input_dict)
        if field not in doc.properties:
            agg_val = default_val
        else:
            agg_val = doc.properties[field]

        new_doc = doc.to_row()
        new_doc[field] = agg_val

        return new_doc

    return ray_callable


def accumulate_row(doc, doc_bytes, fn):
    # print(f"In accumulate_row: {doc} {doc_bytes}")
    return fn(Document.deserialize(doc), Document.deserialize(doc_bytes["doc"])).serialize()


def merge(x, y, fn):
    # print(f"In merge: {Document.deserialize(x)} {Document.deserialize(y)}")
    return fn(Document.deserialize(x), Document.deserialize(y)).serialize()


# Most general version. Keeps everything as Documents. Lots of unnecessary serialization/deserializaiton.
class DocSetAggregate(AggregateFn):
    def __init__(
        self,
        row_fn: Callable[[Document, Document], Document],
        merge_fn: Optional[Callable[[Document, Document], Document]] = None,
        default_props={},
    ):
        if merge_fn is None:
            merge_fn = row_fn

        super().__init__(
            init=lambda x: Document(text_representation="", properties={"companyId": x, **default_props}).serialize(),
            accumulate_row=lambda doc, doc_bytes: accumulate_row(doc, doc_bytes, row_fn),
            merge=lambda x, y: merge(x, y, merge_fn),
            name="doc",
        )


def sum_val_agg(doc1: Document, doc2: Document) -> Document:
    doc3 = Document(**doc1.data)
    # print(f"{type(doc1)} {type(doc2)}")
    # print(f"{doc1} {doc2}")
    new_sum = doc1.properties["sum_val"] + doc2.properties["sum_val"]
    doc3.properties.update(doc1.properties)
    doc3.properties.update(doc2.properties)
    doc3.properties["sum_val"] = new_sum
    return doc3


def group_docset(docset: DocSet, field: str, default_val: Any) -> GroupedData:
    from sycamore import Execution

    execution = Execution(docset.context, docset.plan)
    dataset = execution.execute(docset.plan)

    map_fn = make_aggregation_map_fn(field, default_val)
    dataset2 = dataset.map(map_fn)
    return dataset2.groupby(field)


class DatasetScan(Scan):
    def __init__(self, dataset: Dataset, **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset

    def execute(self) -> Dataset:
        return self._dataset

    def format(self):
        return "dataset"
