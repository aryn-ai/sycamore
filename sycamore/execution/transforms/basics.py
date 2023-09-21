from ray.data import Dataset

from sycamore.execution import Node
from sycamore.execution import NonCPUUser, NonGPUUser
from sycamore.execution import Transform


class Limit(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, limit: int):
        super().__init__(child)
        self._limit = limit

    def execute(self) -> "Dataset":
        dataset = self.child().execute()
        return dataset.limit(self._limit)
