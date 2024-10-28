from typing import TYPE_CHECKING
from sycamore.plan_nodes import Scan

if TYPE_CHECKING:
    from ray.data import Dataset


class DatasetScan(Scan):
    """
    Scans a dataset.
    """

    def __init__(self, dataset: "Dataset", **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset

    def execute(self, **kwargs) -> "Dataset":
        return self._dataset

    def format(self):
        return "dataset"

    def __str__(self):
        return f"DatasetScan({self._dataset})"
