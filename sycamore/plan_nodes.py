from abc import ABC, abstractmethod
from typing import Callable

from ray.data import Dataset


class Node(ABC):
    def __init__(self, children: list["Node"], **resource_args):
        self.children = children
        self.resource_args = resource_args

    def __str__(self):
        return "node"

    @abstractmethod
    def execute(self) -> Dataset:
        pass

    def traverse_down(self, f: Callable[["Node"], "Node"]) -> "Node":
        f(self)
        self.children = [c.traverse_down(f) for c in self.children]
        return self

    def traverse_up(self, f: Callable[["Node"], "Node"]) -> "Node":
        self.children = [c.traverse_up(f) for c in self.children]
        f(self)
        return self

    def clone(self) -> "Node":
        return self


class LeafNode(Node):
    def __init__(self, **resource_args):
        super().__init__([], **resource_args)

    def __str__(self, **resource_args):
        return "leaf"


class UnaryNode(Node):
    def __init__(self, child: Node, **resource_args):
        super().__init__([child], **resource_args)

    def __str__(self):
        return "unary"

    def child(self) -> Node:
        return self.children[0]


class NonCPUUser:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SingleThreadUser:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NonGPUUser:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Scan(SingleThreadUser, NonGPUUser, LeafNode):
    def __init__(self, **resource_args):
        super().__init__(**resource_args)

    def __str__(self):
        return "scan"

    @abstractmethod
    def format(self):
        pass


class Transform(UnaryNode):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    def __str__(self):
        return "transform"


class Write(SingleThreadUser, NonGPUUser, UnaryNode):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    def __str__(self):
        return "write"
