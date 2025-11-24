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
        # TODO: Change node.children from list[optional[Node]] to list[Node]
        child_datasets = [c.execute() for c in self.children if c is not None]
        return self.recursive_merge(child_datasets)

    def recursive_merge(self, dses: list["Dataset"]) -> "Dataset":
        # binary tree of unions seems to have better perf than linear
        if len(dses) == 1:
            return dses[0]
        if len(dses) == 2:
            return dses[0].union(dses[1])
        mid = len(dses) // 2
        return self.recursive_merge(dses[:mid]).union(self.recursive_merge(dses[mid:]))

    def local_execute(self, docs: list[Document]) -> list[Document]:
        # local execution with only 1 child will use this function
        return self.local_execute_many([docs])

    def local_execute_many(self, docses: list[list[Document]]) -> list[Document]:
        res = []
        for docs in docses:
            res.extend(docs)
        return res
