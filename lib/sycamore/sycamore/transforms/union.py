from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore import DocSet
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data import Dataset


class Union(Node):
    """
    Union of multiple docsets
    """

    def __init__(self, *children: DocSet):
        super().__init__([child.plan for child in children])

    def execute(self, **kwargs) -> "Dataset":
        child_datasets = [c.execute() for c in self.children if c is not None]
        return self.merge(child_datasets)
        ds = child_datasets[0]
        for cds in child_datasets[1:]:
            ds = ds.union(cds)
        return ds

    def merge(self, dses: list["Dataset"]) -> "Dataset":
        # binary tree of unions seems to have better perf than linear
        if len(dses) == 1:
            return dses[0]
        if len(dses) == 2:
            return dses[0].union(dses[1])
        first_half = dses[: len(dses) // 2]
        second_half = dses[len(dses) // 2 :]
        return self.merge(first_half).union(self.merge(second_half))

    def local_execute(self, docs: list[Document]) -> list[Document]:
        # union of 1 docset will go through here.
        return self.local_execute_many([docs])

    def local_execute_many(self, docses: list[list[Document]]) -> list[Document]:
        res = []
        for docs in docses:
            res.extend(docs)
        return res
