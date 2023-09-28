from typing import Callable

from ray.data import Dataset

from sycamore.plan_nodes import Node, NonGPUUser, NonCPUUser, Transform

from sycamore.data import Document
from sycamore.plan_nodes import UnaryNode
from sycamore.transforms.map import generate_map_batch_filter_function


class Limit(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, limit: int):
        super().__init__(child)
        self._limit = limit

    def execute(self) -> "Dataset":
        dataset = self.child().execute()
        return dataset.limit(self._limit)


class Filter(UnaryNode):
    def __init__(self, child: Node, *, f: Callable[[Document], bool], **resource_args):
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        ray_callable = generate_map_batch_filter_function(self._f)
        return input_dataset.map_batches(ray_callable, **self.resource_args)
