import pytest
import logging

from sycamore.plan_nodes import Node, NodeTraverse


class MockNode(Node):
    def __init__(self, **kwargs):
        super().__init__([], **kwargs)
        self.value = 0

    def execute(self, **kwargs):
        pass


def set_value_function(node: MockNode) -> None:
    node.value = 1


class SetValueClass:
    def __init__(self):
        self.value = 0

    def set_value(self, node: MockNode) -> None:
        node.value = 1
        self.value += 1


@pytest.fixture
def mock_a_dag(mocker):
    root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 = (
        MockNode(),
        MockNode(),
        MockNode(),
        MockNode(),
        MockNode(),
    )

    mocker.patch.object(root_node, "children", [inter_node1, inter_node2])
    mocker.patch.object(inter_node1, "children", [leaf_node1])
    mocker.patch.object(inter_node2, "children", [leaf_node2])
    mocker.patch.object(leaf_node1, "children", [])
    mocker.patch.object(leaf_node2, "children", [])

    return root_node, inter_node1, inter_node2, leaf_node1, leaf_node2


class TestNode:
    @pytest.mark.parametrize("direction", ["traverse_down", "traverse_up"])
    def test_traverse_function(self, mock_a_dag, direction):
        root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 = mock_a_dag
        traverse = getattr(root_node, direction)
        traverse(f=set_value_function)
        assert root_node.value == 1
        assert inter_node1.value == 1 and inter_node2.value == 1
        assert leaf_node1.value == 1 and leaf_node2.value == 1

    @pytest.mark.parametrize("direction", ["traverse_down", "traverse_up"])
    def test_traverse_class(self, mock_a_dag, direction):
        root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 = mock_a_dag
        set_value_class = SetValueClass()
        traverse = getattr(root_node, direction)
        traverse(f=set_value_class.set_value)
        assert set_value_class.value == 5
        assert root_node.value == 1
        assert inter_node1.value == 1 and inter_node2.value == 1
        assert leaf_node1.value == 1 and leaf_node2.value == 1


class TestTraverse:
    class Noop(Node):
        def __init__(self):
            super().__init__(children=[])

        def execute(self, **kwargs):
            return None

    class One(Noop):
        pass

    class Two(Noop):
        @staticmethod
        def from_one(o):
            n = TestTraverse.Two()
            n.children = o.children
            return n

    @staticmethod
    def assert_one(n):
        assert isinstance(n, TestTraverse.One)
        return n

    @staticmethod
    def assert_two(n):
        assert isinstance(n, TestTraverse.Two)
        return n

    def test_reclass(self):
        One = TestTraverse.One
        n = One()
        n.children = [One(), One()]

        n.traverse(before=self.assert_one)

        n2 = n.traverse(before=self.Two.from_one)
        n.traverse(before=self.assert_one)  # should not change original
        n2.traverse(before=self.assert_two)  # should change new tree

    def test_mutate(self):
        One = TestTraverse.One
        n = One()
        n.children = [One(), One()]
        n.children[0].children = [One()]

        def set_n(n, v):
            n.val = v
            return n

        def add_1(n):
            n.val = n.val + 1
            return n

        def assert_n(n, v):
            assert n.val == v
            return n

        n.traverse(after=lambda n: set_n(n, 1))
        n.traverse(before=lambda n: assert_n(n, 1))
        n.traverse(before=add_1)
        n.traverse(after=lambda n: assert_n(n, 2))

    class Number(NodeTraverse):
        def __init__(self):
            super().__init__()
            self.v_before = 0
            self.v_after = 0

        def before(self, n):
            n.v_before = self.v_before
            self.v_before = self.v_before + 1
            return n

        def after(self, n):
            n.v_after = self.v_after
            self.v_after = self.v_after + 1
            return n

    def test_number(self):
        One = TestTraverse.One
        n = One()
        n.children = [One(), One()]
        n.children[0].children = [One()]

        n.traverse(obj=self.Number())
        assert n.v_before == 0
        assert n.v_after == 3

        ca = n.children[0]
        assert ca.v_before == 1
        assert ca.v_after == 1

        cb = n.children[1]
        assert cb.v_before == 3
        assert cb.v_after == 2

        cc = ca.children[0]
        assert cc.v_before == 2
        assert cc.v_after == 0

    def test_wrap(self):
        One = TestTraverse.One
        Two = TestTraverse.Two

        def mark_wrapped(n):
            if isinstance(n, Two):
                logging.debug("isinstance(Two)")
                assert len(n.children) <= 1
                for c in n.children:
                    c._is_wrapped = True

            return n

        # Note: can not call this function before, it will just push node 0 down repeatedly.
        def wrap(n):
            if isinstance(n, Two):
                return n
            assert isinstance(n, One)
            if hasattr(n, "_is_wrapped") and n._is_wrapped:
                logging.debug(f"skipping wrapping {n.v_before}")
                return n
            logging.debug(f"  wrapping {n.v_before}")
            ret = Two()
            ret.children = [n]
            for c in n.children:
                c._is_wrapped = False
            return ret

        n = One()
        n.children = [One(), One()]
        n.children[0].children = [One()]

        n.traverse(obj=self.Number())
        logging.debug("Wrapping first time")
        t = n.traverse(before=mark_wrapped, after=wrap)

        def assert_alternate(n):
            if isinstance(n, One):
                for c in n.children:
                    assert isinstance(c, Two)
            else:
                for c in n.children:
                    assert isinstance(c, One)
            return n

        assert isinstance(t, Two)
        t.traverse(before=assert_alternate)

        # Because we haven't implemented clone, we are doing in-place mutation
        assert isinstance(n, One)
        n.traverse(before=assert_alternate)

        logging.debug("Rewrapping...")
        # wrap a second time to test idempotency
        t = t.traverse(before=mark_wrapped, after=wrap)

        assert isinstance(t, Two)
        t.traverse(before=assert_alternate)

        assert isinstance(n, One)
        n.traverse(before=assert_alternate)
