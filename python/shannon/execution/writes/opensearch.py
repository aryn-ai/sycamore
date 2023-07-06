from ray.data import Dataset

from shannon.execution import (Node, Write)


class OpenSearchWrite(Write):
    def __init__(
            self, child: Node, *, url: str, index: str, **resource_args):
        super().__init__(child, **resource_args)
        self._url = url
        self._index = index

    def execute(self) -> "Dataset":
        pass
