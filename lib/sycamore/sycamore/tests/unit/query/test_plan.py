from typing import Any, Dict

from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.logical_operator import LogicalOperator


class DummyOperator(LogicalOperator):
    def __init__(self, node_id: str):
        super().__init__(node_id, {"description": "Dummy operator for testing"})

    @staticmethod
    def description() -> str:
        return "Dummy operator for testing"

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"DummyOperator"',
            "description": "string",
        }
        return schema


def test_plan():
    root_node = DummyOperator("root")
    node_2 = DummyOperator("node_2")
    node_3 = DummyOperator("node_3")
    node_4 = DummyOperator("final")

    root_node.downstream_nodes = [node_2, node_3]

    node_2.dependencies = [root_node]
    node_2.downstream_nodes = [node_4]

    node_3.dependencies = [root_node]
    node_3.downstream_nodes = [node_4]

    node_4.dependencies = [node_2, node_3]

    nodes = {
        "root": root_node,
        "node_2": node_2,
        "node_3": node_3,
        "final": node_4,
    }

    plan = LogicalPlan(result_node=node_4, nodes=nodes, query="Test query plan")
    assert plan.result_node() == node_4
    assert plan.nodes() == nodes


def test_count_operator():
    c = Count(node_id="test_id", description="Count operator", field="test_field")
    assert c.node_id == "test_id"
    assert c.description == "Count operator"
    assert c.field == "test_field"
    assert c.usage().startswith("**Count**: Determines the length of a particular database")
    schema = c.input_schema()
    assert "description" in schema
    assert schema["description"] == "typing.Optional[str]"
    assert "node_id" in schema
    assert schema["node_id"] == "<class 'str'>"
    assert "field" in schema
    assert schema["field"] == "typing.Optional[str]"
    assert "primaryField" in schema
    assert schema["primaryField"] == "typing.Optional[str]"

