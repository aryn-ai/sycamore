import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ray import Dataset
    from sycamore.context import Context


class NodeTraverse:
    """NodeTraverse allows for complicated traversals

    For simple use cases, call node.traverse({before,visit,after}=fn)

    - before is called before traversing children.
    - after is called after traversing children.
    - visit is called over each node in an unspecified order, and is easier to use since the
      function returns nothing.
    - once is called one time at the very start, and enables multi-pass transforms.
    """

    def __init__(
        self,
        before: Optional[Callable[["Node"], "Node"]] = None,
        visit: Optional[Callable[["Node"], None]] = None,
        after: Optional[Callable[["Node"], "Node"]] = None,
    ):
        self.before_fn = before
        self.visit_fn = visit
        self.after_fn = after

    def once(self, context: "Context", node: "Node") -> "Node":
        # Called one time at the start of rewriting on the root of the tree.
        # Enables multi-pass traversals
        return node

    # Called before traversing children
    def before(self, node: "Node") -> "Node":
        if self.before_fn is None:
            return node
        return self.before_fn(node)

    # Called before traversing children, convenience function for single node mutating operations
    def visit(self, node: "Node") -> None:
        if self.visit_fn is not None:
            self.visit_fn(node)

    # Called after traversing children
    def after(self, node: "Node") -> "Node":
        if self.after_fn is None:
            return node
        return self.after_fn(node)


class Node(ABC):
    """
    A Node is the abstract base unit of a Sycamore Transform, which allows DocSets to transform themselves into end
    results. Sycamore processes this as a directed tree graph, which allows transforms to be linked to each other
    and then implemented
    """

    def __init__(
        self,
        children: list[Optional["Node"]],
        materialize: dict = {},
        parallelism: Optional[int] = None,
        **resource_args,
    ):
        self.children = children
        assert parallelism is None or parallelism > 0
        self.parallelism = parallelism
        self.resource_args = resource_args
        self.properties = {}
        # copy because of https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument
        self.properties["materialize"] = materialize.copy()

    def __str__(self):
        return "node"

    @abstractmethod
    def execute(self, **kwargs) -> "Dataset":
        pass

    def prepare(self) -> Optional[Callable]:
        """Override this method to run something at the beginning of execution after rules have
        been applied. The entire tree will be traversed in before mode and then any returned
        callables will be called in the order they were returned. Each callable can return
        another callable."""

        pass

    def finalize(self) -> None:
        """Override this method to run something at the end of execution after all documents have
        been returned."""
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

    def traverse(
        self,
        obj: Optional[NodeTraverse] = None,
        before: Optional[Callable[["Node"], "Node"]] = None,
        visit: Optional[Callable[["Node"], None]] = None,
        after: Optional[Callable[["Node"], "Node"]] = None,
    ) -> "Node":
        """
        Traverse the node tree, functions will be converted to an object.
        See NodeTraverse for the semantics.
        """
        if obj is None:
            assert before is not None or visit is not None or after is not None
            obj = NodeTraverse(before=before, visit=visit, after=after)
        else:
            assert before is None and visit is None and after is None

        return self._traverse(obj)

    def _traverse(self, obj: NodeTraverse) -> "Node":
        n = obj.before(self)
        obj.visit(self)
        n.children = [c._traverse(obj) for c in n.children if c is not None]
        return obj.after(n)

    def clone(self) -> "Node":
        raise Exception("Unimplemented")

    def last_node_reliability_assertor(self) -> bool:
        from sycamore.materialize import Materialize
        from sycamore.connectors.file import BinaryScan, JsonScan

        if not self.children or all(c is None for c in self.children):
            return (isinstance(self, Materialize) and self._reliability is None) or (
                (isinstance(self, JsonScan) or isinstance(self, BinaryScan)) and self.filter_paths is not None
            )
        assert not isinstance(
            self, Materialize
        ), "For ensuring reliability, only first node must be materialize or first node must be binary scan with filter_paths set"

        for child in self.children:
            if child is not None:
                return child.last_node_reliability_assertor()
        return False


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


def print_plan(node: Node, stream=sys.stdout) -> None:
    """Utility function for printing plans.

    Prints the node and all its children. Indentation is used
    to indicate the parent-child relation, e.g.

    RootNode
      InternalNode1
        LeafNode1
      InternalNode2
        LeafNode2
    """

    class PrintTraverse(NodeTraverse):
        def __init__(self):
            super().__init__()
            self.indent = 0
            self.stream = stream

        def before(self, n: Node) -> Node:
            self.stream.write(" " * self.indent)
            self.stream.write(f"{n.__class__.__name__} {n.properties}\n")
            self.indent += 2
            return n

        def after(self, n: Node) -> Node:
            if self.indent >= 0:
                self.indent -= 2
            return n

    node.traverse(obj=PrintTraverse())
