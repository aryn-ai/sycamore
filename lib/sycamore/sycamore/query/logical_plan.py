from typing import Any, Dict, List, Optional


class Node:

    def __init__(
        self, node_id: str, dependencies: Optional[List[Any]] = None, downstream_nodes: Optional[List[Any]] = None
    ) -> None:
        super().__init__()
        self.node_id = node_id
        self.dependencies = dependencies
        self.downstream_nodes = downstream_nodes

    def show(self, indent=0, verbose=False):
        pass

    def type(self) -> str:
        raise NotImplementedError


def print_dag(node: Node, indent: int = 0, verbose=False) -> None:
    node.show(indent=indent, verbose=verbose)
    if node.dependencies:

        for dep_node in node.dependencies:
            print(" " * indent + " | --->")
            print_dag(dep_node, indent + 4, verbose=verbose)


class LogicalPlan:
    def __init__(
        self, result_node: Node, nodes: Dict[str, Node], query: str, openai_plan: Optional[str] = None
    ) -> None:
        super().__init__()
        self._result_node = result_node
        self._query = query
        self._nodes = nodes
        self._openai_plan = openai_plan

    def nodes(self):
        return self._nodes

    def result_node(self):
        return self._result_node

    def show(self, verbose=False):
        print(f"Query: {self._query}")
        print_dag(self._result_node, verbose=verbose)

    def openai_plan(self):
        print(self._openai_plan)
