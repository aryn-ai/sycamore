from ray.data import Dataset
from sycamore.plan_nodes import Scan

class DatasetScan(Scan):
    """
    Scans a dataset.
    """

    def __init__(self, dataset: Dataset, **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset

    def execute(self, **kwargs) -> Dataset:
        return self._dataset

    def format(self):
        return "dataset"