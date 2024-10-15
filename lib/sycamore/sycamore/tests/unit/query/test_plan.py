from typing import Optional

import pytest

from sycamore.query.logical_plan import Node, LogicalPlan, LogicalNodeDiffType
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.operators.summarize_data import SummarizeData


class DummyOperator(Node):
    dummy: Optional[str] = None
    """A dummy field for testing purposes."""


def test_node_serialize_deserialize():
    node = DummyOperator(node_id=1, description="Test node", dummy="Dummy value")

    serialized = node.model_dump()
    assert serialized["node_id"] == 1
    assert serialized["description"] == "Test node"
    assert serialized["node_type"] == "DummyOperator"
    assert serialized["dummy"] == "Dummy value"

    # Explicit validation as DummyOperator works.
    deserialized = DummyOperator.model_validate(serialized)
    assert isinstance(deserialized, DummyOperator)
    assert deserialized.node_id == 1
    assert deserialized.description == "Test node"
    assert deserialized.dummy == "Dummy value"

    # Explicit deserialization as DummyOperator works.
    deserialized = DummyOperator.deserialize(serialized)
    assert isinstance(deserialized, DummyOperator)
    assert deserialized.node_id == 1
    assert deserialized.description == "Test node"
    assert deserialized.dummy == "Dummy value"

    # Duck-typed deserialization via Node works.
    deserialized = Node.deserialize(serialized)
    assert isinstance(deserialized, DummyOperator)
    assert deserialized.node_id == 1
    assert deserialized.description == "Test node"
    assert deserialized.dummy == "Dummy value"
    assert deserialized.node_type == "DummyOperator"


def test_plan_serialize_deserialize():

    nodes = {
        1: DummyOperator(node_id=1, description="node_1"),
        2: DummyOperator(node_id=2, description="node_2", dummy="test2", inputs=[1]),
        3: DummyOperator(node_id=3, description="node_3", dummy="test3", inputs=[1]),
        4: DummyOperator(node_id=4, description="final", inputs=[2, 3]),
    }
    plan = LogicalPlan(
        result_node=4, nodes=nodes, query="Test query plan", llm_prompt="Test LLM prompt", llm_plan="Test LLM plan"
    )

    assert plan.result_node == 4
    assert plan.nodes == nodes
    assert plan.nodes[1] == nodes[1]
    assert plan.nodes[2] == nodes[2]
    assert plan.nodes[3] == nodes[3]
    assert plan.nodes[4] == nodes[4]

    assert plan.nodes[1].inputs == []
    assert plan.nodes[2].inputs == [1]
    assert plan.nodes[3].inputs == [1]
    assert plan.nodes[4].inputs == [2, 3]

    assert plan.nodes[1].input_nodes() == []
    assert plan.nodes[2].input_nodes() == [nodes[1]]

    serialized = plan.model_dump()
    assert serialized["result_node"] == 4
    assert serialized["nodes"] == {k: v.model_dump() for k, v in nodes.items()}

    for node_id in nodes.keys():
        assert serialized["nodes"][node_id] == nodes[node_id].model_dump()

    # Deserialize the plan.
    deserialized = LogicalPlan.model_validate(serialized)

    assert deserialized.query == "Test query plan"
    assert deserialized.llm_prompt == "Test LLM prompt"
    assert deserialized.llm_plan == "Test LLM plan"
    assert deserialized.result_node == 4
    assert len(deserialized.nodes) == 4

    assert isinstance(deserialized.nodes[1], DummyOperator)
    assert deserialized.nodes[1].node_id == 1
    assert deserialized.nodes[1].description == "node_1"

    assert isinstance(deserialized.nodes[2], DummyOperator)
    assert deserialized.nodes[2].node_id == 2
    assert deserialized.nodes[2].description == "node_2"
    assert deserialized.nodes[2].dummy == "test2"
    assert deserialized.nodes[2].inputs == [1]
    assert deserialized.nodes[2].input_nodes() == [deserialized.nodes[1]]

    assert isinstance(deserialized.nodes[3], DummyOperator)
    assert deserialized.nodes[3].node_id == 3
    assert deserialized.nodes[3].description == "node_3"
    assert deserialized.nodes[3].dummy == "test3"
    assert deserialized.nodes[3].inputs == [1]
    assert deserialized.nodes[3].input_nodes() == [deserialized.nodes[1]]

    assert isinstance(deserialized.nodes[4], DummyOperator)
    assert deserialized.nodes[4].node_id == 4
    assert deserialized.nodes[4].description == "final"
    assert deserialized.nodes[4].inputs == [2, 3]
    assert deserialized.nodes[4].input_nodes() == [deserialized.nodes[2], deserialized.nodes[3]]


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

    assert "inputs" in schema
    assert schema["inputs"].field_name == "inputs"
    assert schema["inputs"].description == "A list of node IDs that this operation depends on."
    assert schema["inputs"].type_hint == "typing.List[int]"

    assert "node_id" in schema
    assert schema["node_id"].field_name == "node_id"
    assert schema["node_id"].description == "A unique integer ID representing this node."
    assert schema["node_id"].type_hint == "<class 'int'>"

    assert "distinct_field" in schema
    assert schema["distinct_field"].field_name == "distinct_field"
    assert schema["distinct_field"].description.startswith("If specified, returns the count")
    assert schema["distinct_field"].type_hint == "typing.Optional[str]"

    assert "_inputs" not in schema


@pytest.fixture
def llm_filter_plan():
    nodes = {
        0: QueryDatabase(
            node_id=0, description="Get all the airplane incidents", index="ntsb", query={"match_all": {}}
        ),
        1: LlmFilter(
            node_id=1,
            description="Filter to only include Piper aircraft incidents",
            question="Did this incident occur in a Piper aircraft?",
            field="properties.entity.aircraft",
            inputs=[0],
        ),
        2: Count(
            node_id=2,
            description="Determine how many incidents occurred in Piper aircrafts",
            countUnique=False,
            field=None,
            inputs=[1],
        ),
        3: SummarizeData(
            node_id=3,
            description="Generate an English response to the question",
            question="How many Piper aircrafts were involved in accidents?",
            inputs=[2],
        ),
    }
    return LogicalPlan(result_node=3, nodes=nodes, query="", llm_prompt="", llm_plan="")


@pytest.fixture
def vector_search_filter_plan():
    nodes = {
        0: QueryVectorDatabase(
            node_id=0,
            description="Get all the airplane incidents",
            index="ntsb",
            query_phrase="Get all the airplane incidents",
            filter={"properties.entity.aircraft": "Piper"},
        ),
        1: Count(
            node_id=1,
            description="Determine how many incidents occurred in Piper aircrafts",
            countUnique=False,
            field=None,
            inputs=[0],
        ),
        2: SummarizeData(
            node_id=2,
            description="Generate an English response to the question",
            question="How many Piper aircrafts were involved in accidents?",
            inputs=[1],
        ),
    }
    return LogicalPlan(result_node=2, nodes=nodes, query="", llm_prompt="", llm_plan="")


def test_compare_plans(llm_filter_plan, vector_search_filter_plan):
    diff = llm_filter_plan.compare(vector_search_filter_plan)

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
    llm_filter_plan_modified = LogicalPlan(**llm_filter_plan.model_dump())
    llm_filter_plan_modified.nodes[1].question = "this is another question"
    diff = llm_filter_plan.compare(llm_filter_plan_modified)
    assert len(diff) == 0


def test_compare_plans_data_changed(llm_filter_plan):
    llm_filter_plan_modified = LogicalPlan(**llm_filter_plan.model_dump())
    llm_filter_plan_modified.nodes[1].field = "different_field"
    diff = llm_filter_plan.compare(llm_filter_plan_modified)
    assert len(diff) == 1
    assert diff[0].diff_type == LogicalNodeDiffType.OPERATOR_DATA
    assert isinstance(diff[0].node_a, LlmFilter)
    assert isinstance(diff[0].node_b, LlmFilter)
    assert diff[0].node_a.field == "properties.entity.aircraft"
    assert diff[0].node_b.field == "different_field"


def test_compare_plans_structure_changed(llm_filter_plan):
    llm_filter_plan_modified = LogicalPlan(**llm_filter_plan.model_dump())
    llm_filter_plan_modified.nodes[2].inputs = []
    diff = llm_filter_plan.compare(llm_filter_plan_modified)
    assert len(diff) == 1
    assert diff[0].diff_type == LogicalNodeDiffType.PLAN_STRUCTURE
    assert isinstance(diff[0].node_a, LlmFilter)
    assert isinstance(diff[0].node_b, LlmFilter)
