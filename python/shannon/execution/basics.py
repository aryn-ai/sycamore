from abc import ABC, abstractmethod
from ray.data import Dataset
from typing import (Callable, List, Optional, TypeVar, Union)


class Node(ABC):
    def __init__(self, **resource_args):
        self.resource_args = resource_args

    def __str__(self):
        return "node"

    def resource_args(self, **resource_args) -> None:
        self.resource_args = resource_args

    def child(self) -> Union[None, "Node", List["Node"]]:
        pass

    def execute(self) -> "Dataset":
        pass

    T = TypeVar('T')

    def traverse_down(self, f: Callable[["Node"], T]) -> None:
        f(self)
        if self.child() is None:
            return
        elif isinstance(self.child(), Node):
            self.child().traverse_down(f)
        else:
            for c in self.child():
                c.traverse_down(f)

    def traverse_up(self, f: Callable[["Node"], T]) -> None:
        if self.child() is None:
            pass
        elif isinstance(self.child(), Node):
            self.child().traverse_up(f)
        else:
            for c in self.child():
                c.traverse_up(f)
        f(self)


class LeafNode(Node):
    def __init__(self, **resource_args):
        super().__init__(**resource_args)

    def __str__(self):
        return "leaf"

    def child(self) -> Union[None, "Node", List["Node"]]:
        return None


class Scan(LeafNode):
    def __init__(self, **resource_args):
        super().__init__(**resource_args)

    def __str__(self):
        return "scan"

    @abstractmethod
    def format(self):
        pass


class UnaryNode(Node):
    def __init__(self, child: Node, **resource_args):
        super().__init__(**resource_args)
        self._child = child

    def __str__(self):
        return "unary"

    def child(self) -> Optional["Node"]:
        return self._child


class Transform(UnaryNode):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    def __str__(self):
        return "transform"


class Write(UnaryNode):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    def __str__(self):
        return "write"


class Rule:
    def __call__(self, plan: Node) -> Node:
        raise NotImplementedError
