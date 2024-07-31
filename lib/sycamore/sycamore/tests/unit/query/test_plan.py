from typing import Optional

from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.logical_operator import LogicalOperator


class DummyOperator(LogicalOperator):
    dummy: Optional[str] = None
    """A dummy field for testing purposes."""


def test_plan():
    node_1 = DummyOperator(node_id=1, description="node_1")
    node_2 = DummyOperator(node_id=1, description="node_2")
    node_3 = DummyOperator(node_id=2, description="node_3")
    node_4 = DummyOperator(node_id=3, description="final")

    node_1._downstream_nodes = [node_2, node_3]

    node_2._dependencies = [node_1]
    node_2._downstream_nodes = [node_4]

    node_3._dependencies = [node_1]
    node_3._downstream_nodes = [node_4]

    node_4._dependencies = [node_2, node_3]

    nodes = {
        1: node_1,
        2: node_2,
        3: node_3,
        4: node_4,
    }

    plan = LogicalPlan(result_node=node_4, nodes=nodes, query="Test query plan")
    assert plan.result_node == node_4
    assert plan.nodes == nodes


def test_count_operator():
    c = Count(node_id=77, description="Count operator", field="test_field")
    assert c.node_id == 77
    assert c.description == "Count operator"
    assert c.field == "test_field"
    assert c.usage().startswith("**Count**: Determines the length of a particular database")
    schema = c.input_schema()

    assert "description" in schema
    assert schema["description"].field_name == "description"
    assert (
        schema["description"].description
        == "A detailed description of why this operator was chosen for this query plan."
    )
    assert schema["description"].type_hint == "typing.Optional[str]"

    assert "input" in schema
    assert schema["input"].field_name == "input"
    assert schema["input"].description == "A list of node IDs that this operation depends on."
    assert schema["input"].type_hint == "typing.Optional[typing.List[int]]"

    assert "node_id" in schema
    assert schema["node_id"].field_name == "node_id"
    assert schema["node_id"].description == "A unique integer ID representing this node."
    assert schema["node_id"].type_hint == "<class 'int'>"

    assert "field" in schema
    assert schema["field"].field_name == "field"
    assert schema["field"].description == "Non-primary database field to return a count based on."
    assert schema["field"].type_hint == "typing.Optional[str]"

    assert "primary_field" in schema
    assert schema["primary_field"].field_name == "primary_field"
    assert (
        schema["primary_field"].description
        == "Primary field that represents what a unique entry is considered for the data provided."
    )
    assert schema["primary_field"].type_hint == "typing.Optional[str]"

    assert "_dependencies" not in schema
    assert "_downstream_nodes" not in schema
    assert "dependencies" not in schema
    assert "downstream_nodes" not in schema
