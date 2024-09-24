import pytest

from sycamore.query.compare import compare_logical_plans_from_query_source
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner import process_json_plan


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
    diff = compare_logical_plans_from_query_source(
        get_logical_plan(llm_filter_plan), get_logical_plan(vector_search_filter_plan)
    )
    assert len(diff["node_type_diff_result"]) > 0
    assert len(diff["node_data_diff_result"]) > 0
    assert len(diff["structural_diff_result"]) > 0


def test_compare_plans_diff_llm_filter_string(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1].question = "this is another question"
    diff = compare_logical_plans_from_query_source(get_logical_plan(llm_filter_plan), llm_filter_plan_modified)
    assert len(diff["node_type_diff_result"]) == 0
    assert len(diff["node_data_diff_result"]) == 0
    assert len(diff["structural_diff_result"]) == 0


def test_compare_plans_data_changed(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1].field = "different_field"
    diff = compare_logical_plans_from_query_source(get_logical_plan(llm_filter_plan), llm_filter_plan_modified)
    assert len(diff["node_type_diff_result"]) == 0
    assert len(diff["node_data_diff_result"]) == 1
    assert len(diff["structural_diff_result"]) == 0


def test_compare_plans_structure_changed(llm_filter_plan):
    llm_filter_plan_modified = get_logical_plan(llm_filter_plan)
    llm_filter_plan_modified.nodes[1]._downstream_nodes = []
    diff = compare_logical_plans_from_query_source(get_logical_plan(llm_filter_plan), llm_filter_plan_modified)
    assert len(diff["node_type_diff_result"]) == 0
    assert len(diff["node_data_diff_result"]) == 0
    assert len(diff["structural_diff_result"]) == 1
