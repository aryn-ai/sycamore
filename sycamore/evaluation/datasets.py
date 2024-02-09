from typing import Any, Union

from datasets import IterableDataset
from ray.data import Dataset, from_huggingface

from sycamore.evaluation import EvaluationDataPoint
from sycamore.scans import MaterializedScan
from sycamore import DocSet, Context


class HuggingFaceScan(MaterializedScan):
    def __init__(self, dataset: Union[Dataset, IterableDataset], field_mapping: dict[str, str], **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset
        self._field_mapping = field_mapping

    def _hf_to_qa_datapoint(self, data: dict[str, Any]) -> dict[str, Any]:
        document = EvaluationDataPoint()
        if self._field_mapping:
            for k, v in self._field_mapping.items():
                document[k] = data[v]
        document["raw"] = data
        return {"doc": document.serialize()}

    def execute(self) -> Dataset:
        ray_ds = from_huggingface(self._dataset)
        processed = ray_ds.map(self._hf_to_qa_datapoint)
        return processed

    def format(self):
        return "huggingface"


class EvaluationDataSetReader:
    def __init__(self, context: Context) -> None:
        super().__init__()
        self._context = context

    def huggingface(
        self, dataset: Union[Dataset, IterableDataset], field_mapping: dict[str, str], **resource_args
    ) -> DocSet:
        json_scan = HuggingFaceScan(dataset=dataset, field_mapping=field_mapping, **resource_args)
        return DocSet(self._context, json_scan)
