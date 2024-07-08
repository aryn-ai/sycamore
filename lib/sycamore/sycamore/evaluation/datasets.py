from typing import Union, Callable, Any

from datasets import IterableDataset
from ray.data import Dataset, from_huggingface

from sycamore.evaluation import EvaluationDataPoint
from sycamore import DocSet, Context
from sycamore.connectors.file import MaterializedScan


class HuggingFaceScan(MaterializedScan):
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        doc_extractor: Callable[[dict[str, Any]], dict[str, EvaluationDataPoint]],
        **resource_args
    ):
        super().__init__(**resource_args)
        self._dataset = dataset
        self._doc_extractor = doc_extractor

    def execute(self, **kwargs) -> Dataset:
        ray_ds = from_huggingface(self._dataset)
        processed = ray_ds.map(self._doc_extractor)
        return processed

    def format(self):
        return "huggingface"


class EvaluationDataSetReader:
    def __init__(self, context: Context) -> None:
        super().__init__()
        self._context = context

    def huggingface(
        self,
        dataset: Union[Dataset, IterableDataset],
        doc_extractor: Callable[[dict[str, Any]], dict[str, EvaluationDataPoint]],
        **resource_args
    ) -> DocSet:
        json_scan = HuggingFaceScan(dataset=dataset, doc_extractor=doc_extractor, **resource_args)
        return DocSet(self._context, json_scan)
