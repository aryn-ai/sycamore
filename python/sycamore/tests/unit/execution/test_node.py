import pytest

from sycamore.execution import Node


class MockNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0


def set_value_function(node: Node) -> None:
    node.value = 1


class SetValueClass:
    def __init__(self):
        self.value = 0

    def set_value(self, node: Node) -> None:
        node.value = 1
        self.value += 1


@pytest.fixture
def mock_a_dag(mocker):
    root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 = \
        MockNode(), MockNode(), MockNode(), MockNode(), MockNode()

    mocker.patch.object(
        root_node, "child", new=lambda: [inter_node1, inter_node2])
    mocker.patch.object(inter_node1, "child", new=lambda: leaf_node1)
    mocker.patch.object(inter_node2, "child", new=lambda: leaf_node2)
    mocker.patch.object(leaf_node1, "child", new=lambda: None)
    mocker.patch.object(leaf_node2, "child", new=lambda: None)

    return root_node, inter_node1, inter_node2, leaf_node1, leaf_node2


class TestNode:

    @pytest.mark.parametrize("direction", ["traverse_down", "traverse_up"])
    def test_traverse_function(self, mock_a_dag, direction):
        root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 =\
            mock_a_dag
        traverse = getattr(root_node, direction)
        traverse(f=set_value_function)
        assert (root_node.value == 1)
        assert (inter_node1.value == 1 and inter_node2.value == 1)
        assert (leaf_node1.value == 1 and leaf_node2.value == 1)

    @pytest.mark.parametrize("direction", ["traverse_down", "traverse_up"])
    def test_traverse_class(self, mock_a_dag, direction):
        root_node, inter_node1, inter_node2, leaf_node1, leaf_node2 =\
            mock_a_dag
        set_value_class = SetValueClass()
        traverse = getattr(root_node, direction)
        traverse(f=set_value_class.set_value)
        assert (set_value_class.value == 5)
        assert (root_node.value == 1)
        assert (inter_node1.value == 1 and inter_node2.value == 1)
        assert (leaf_node1.value == 1 and leaf_node2.value == 1)
