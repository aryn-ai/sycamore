from abc import ABC, abstractmethod
from typing import Callable, Optional

from ray.data import Dataset


class Node(ABC):
    """
    A Node is the abstract base unit of a Sycamore Transform, which allows DocSets to transform themselves into end
    results. Sycamore processes this as a directed tree graph, which allows transforms to be linked to each other
    and then implemented
    """

    def __init__(self, children: list[Optional["Node"]], **resource_args):
        self.children = children
        self.resource_args = resource_args

    def __str__(self):
        return "node"

    @abstractmethod
    def execute(self, **kwargs) -> Dataset:
        pass

    def traverse_down(self, f: Callable[["Node"], "Node"]) -> "Node":
        """
        Allows a function to be applied to a node first and then all of its children
        """
        f(self)
        self.children = [c.traverse_down(f) for c in self.children if c is not None]
        return self

    def traverse_up(self, f: Callable[["Node"], "Node"]) -> "Node":
        """
        Allows a function to be applied to all of a node's children first and then itelf
        """
        self.children = [c.traverse_up(f) for c in self.children if c is not None]
        f(self)
        return self

    def clone(self) -> "Node":
        raise Exception("Unimplemented")


class LeafNode(Node):
    def __init__(self, **resource_args):
        super().__init__([], **resource_args)

    def __str__(self, **resource_args):
        return "leaf"


class UnaryNode(Node):
    def __init__(self, child: Optional[Node], **resource_args):
        super().__init__([child], **resource_args)

    def __str__(self):
        return "unary"

    def child(self) -> Node:
        assert self.children[0] is not None
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
