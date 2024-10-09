from typing import Optional

import pytest

from sycamore.query.logical_plan import Node, LogicalPlan, LogicalNodeDiffType
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.planner import process_json_plan


class DummyOperator(LogicalOperator):
    dummy: Optional[str] = None
    """A dummy field for testing purposes."""


def test_node_serialize_deserialize():
    node = DummyOperator(node_id=1, description="Test node", dummy="Dummy value")
    serialized = node.model_dump()
    assert serialized["node_id"] == 1
    assert serialized["description"] == "Test node"
    assert serialized["node_type"] == "DummyOperator"
    assert serialized["dummy"] == "Dummy value"

    # Explicit deserialization as DummyOperator works.
#    deserialized = DummyOperator.model_validate(serialized)
#    assert isinstance(deserialized, DummyOperator)
#    assert deserialized.node_id == 1
#    assert deserialized.description == "Test node"
#    assert deserialized.dummy == "Dummy value"

    # Duck-typed deserialization as Node works.
    deserialized = Node.deserialize(serialized)
    assert isinstance(deserialized, DummyOperator)
    assert deserialized.node_id == 1
    assert deserialized.description == "Test node"
    assert deserialized.dummy == "Dummy value"
    assert deserialized.node_type == "DummyOperator"


def test_plan_serialize_deserialize():
    node_1 = DummyOperator(node_id=1, description="node_1")
    node_2 = DummyOperator(node_id=2, description="node_2", dummy="test2")
    node_3 = DummyOperator(node_id=3, description="node_3", dummy="test3")
    node_4 = DummyOperator(node_id=4, description="final")

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

    plan = LogicalPlan(
        result_node=node_4, nodes=nodes, query="Test query plan", llm_prompt="Test LLM prompt", llm_plan="Test LLM plan"
    )
    assert plan.result_node == node_4
    assert plan.nodes == nodes

    serialized = plan.model_dump()
    assert serialized["result_node"] == node_4.model_dump()
    assert serialized["nodes"] == {k: v.model_dump() for k, v in nodes.items()}

    # Ensure serialized nodes have the correct fields and references to one another.
    node_1_serialized = serialized["nodes"][1]
    assert node_1_serialized == node_1.model_dump()
    assert "_dependencies" not in node_1_serialized
    assert "_downstream_nodes" not in node_1_serialized
    assert node_1_serialized["node_type"] == "DummyOperator"
    assert node_1_serialized["dependencies"] == []
    assert node_1_serialized["downstream_nodes"] == [2, 3]

    node_2_serialized = serialized["nodes"][2]
    assert node_2_serialized == node_2.model_dump()
    assert "_dependencies" not in node_2_serialized
    assert "_downstream_nodes" not in node_2_serialized
    assert node_2_serialized["dependencies"] == [1]
    assert node_2_serialized["downstream_nodes"] == [4]

    node_3_serialized = serialized["nodes"][3]
    assert node_3_serialized == node_3.model_dump()
    assert "_dependencies" not in node_3_serialized
    assert "_downstream_nodes" not in node_3_serialized
    assert node_3_serialized["dependencies"] == [1]
    assert node_3_serialized["downstream_nodes"] == [4]

    node_4_serialized = serialized["nodes"][4]
    assert node_4_serialized == node_4.model_dump()
    assert "_dependencies" not in node_4_serialized
    assert "_downstream_nodes" not in node_4_serialized
    assert node_4_serialized["dependencies"] == [2, 3]
    assert node_4_serialized["downstream_nodes"] == []

    print("MDW: DOING VALIDATION")
    deserialized = LogicalPlan.model_validate(serialized)
    print("MDW: DONE WITH VALIDATION")
    print(f"MDW: GOT BACK: {deserialized}")
    assert deserialized.query == "Test query plan"
    assert deserialized.llm_prompt == "Test LLM prompt"
    assert deserialized.llm_plan == "Test LLM plan"
    assert deserialized.result_node == deserialized.nodes[4]
    assert len(deserialized.nodes) == 4

    assert isinstance(deserialized.nodes[1], DummyOperator)


def test_count_operator():
    c = Count(node_id=77, description="Count operator", distinct_field="test_field")
    assert c.node_id == 77
    assert c.description == "Count operator"
    assert c.distinct_field == "test_field"
    assert c.usage().startswith("**Count**: Returns a count")
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

    assert "distinct_field" in schema
    assert schema["distinct_field"].field_name == "distinct_field"
    assert schema["distinct_field"].description.startswith("If specified, returns the count")
    assert schema["distinct_field"].type_hint == "typing.Optional[str]"

    assert "_dependencies" not in schema
    assert "_downstream_nodes" not in schema
    assert "dependencies" not in schema
    assert "downstream_nodes" not in schema


@pytest.fixture
def llm_filter_plan():
    return [
        {
            "operatorName": "QueryDatabase",
            "description": "Get all the airplane incidents",
            "index": "ntsb",
            "query": {"match_all": {}},
            "node_id": 0,
        },
        {
            "operatorName": "LlmFilter",
            "description": "Filter to only include Piper aircraft incidents",
            "question": "Did this incident occur in a Piper aircraft?",
            "field": "properties.entity.aircraft",
            "input": [0],
            "node_id": 1,
        },
        {
            "operatorName": "Count",
            "description": "Determine how many incidents occurred in Piper aircrafts",
            "countUnique": False,
            "field": None,
            "input": [1],
            "node_id": 2,
        },
        {
            "operatorName": "SummarizeData",
            "description": "Generate an English response to the question",
            "question": "How many Piper aircrafts were involved in accidents?",
            "input": [2],
            "node_id": 3,
        },
    ]


@pytest.fixture
def vector_search_filter_plan():
    return [
        {
            "operatorName": "QueryVectorDatabase",
            "description": "Get all the airplane incidents",
            "index": "ntsb",
            "query_phrase": "Get all the airplane incidents",
            "filter": {"properties.entity.aircraft": "Piper"},
            "node_id": 0,
        },
        {
            "operatorName": "Count",
            "description": "Determine how many incidents occurred in Piper aircrafts",
            "countUnique": False,
            "field": None,
            "input": [0],
            "node_id": 1,
        },
        {
            "operatorName": "SummarizeData",
            "description": "Generate an English response to the question",
            "question": "How many Piper aircrafts were involved in accidents?",
            "input": [1],
            "node_id": 2,
        },
    ]


def get_logical_plan(plan):
    result_node, nodes = process_json_plan(plan)
    plan = LogicalPlan(result_node=result_node, nodes=nodes, query="", llm_prompt="", llm_plan="")
    return plan


def test_compare_plans(llm_filter_plan, vector_search_filter_plan):
    diff = get_logical_plan(llm_filter_plan).compare(get_logical_plan(vector_search_filter_plan))
    assert len(diff) == 7
    assert diff[0].diff_type == LogicalNodeDiffType.OPERATOR_TYPE
    assert isinstance(diff[0].node_a, QueryDatabase)
    assert isinstance(diff[0].node_b, QueryVectorDatabase)
    assert diff[1].diff_type == LogicalNodeDiffType.OPERATOR_DATA

    assert diff[2].diff_type == LogicalNodeDiffType.OPERATOR_TYPE
    assert isinstance(diff[2].node_a, LlmFilter)
    assert isinstance(diff[2].node_b, Count)
    assert diff[3].diff_type == LogicalNodeDiffType.OPERATOR_DATA

    assert diff[4].diff_type == LogicalNodeDiffType.OPERATOR_TYPE
    assert isinstance(diff[4].node_a, Count)
    assert isinstance(diff[4].node_b, SummarizeData)
    assert diff[5].diff_type == LogicalNodeDiffType.OPERATOR_DATA

    assert diff[6].diff_type == LogicalNodeDiffType.PLAN_STRUCTURE


def test_compare_plans_diff_llm_filter_string(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1].question = "this is another question"
    diff = get_logical_plan(llm_filter_plan).compare(llm_filter_plan_modified)
    assert len(diff) == 0


def test_compare_plans_data_changed(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1].field = "different_field"
    diff = get_logical_plan(llm_filter_plan).compare(llm_filter_plan_modified)
    assert len(diff) == 1
    assert diff[0].diff_type == LogicalNodeDiffType.OPERATOR_DATA
    assert isinstance(diff[0].node_a, LlmFilter)
    assert isinstance(diff[0].node_b, LlmFilter)
    assert diff[0].node_a.field == "properties.entity.aircraft"
    assert diff[0].node_b.field == "different_field"


def test_compare_plans_structure_changed(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1]._downstream_nodes = []
    diff = get_logical_plan(llm_filter_plan).compare(llm_filter_plan_modified)
    assert len(diff) == 1
    assert diff[0].diff_type == LogicalNodeDiffType.PLAN_STRUCTURE
    assert isinstance(diff[0].node_a, LlmFilter)
    assert isinstance(diff[0].node_b, LlmFilter)
    assert len(diff[0].node_a.get_downstream_nodes()) == 1
    assert diff[0].node_a.get_downstream_nodes()[0].node_id == 2
    assert len(diff[0].node_b.get_downstream_nodes()) == 0
