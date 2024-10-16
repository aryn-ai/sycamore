from sycamore.query.planner import postprocess_plan
from sycamore.query.logical_plan import LogicalPlan


class DummyLLMClient:
    def generate(prompt_kwargs, llm_kwargs):
        return "Dummy response from an LLM Client"


def vector_search_filter_plan_with_opensearch_filter():
    json_plan = {
        "query": "How many incidents involving Piper Aircrafts in California",
        "nodes": {
            "0": {
                "node_type": "QueryVectorDatabase",
                "node_id": 0,
                "description": "Get all the airplane incidents in California",
                "index": "ntsb",
                "inputs": [],
                "query_phrase": "Get all the airplane incidents",
                "opensearch_filter": {"match": {"properties.entity.location": "California"}},
            },
            "1": {
                "node_type": "Count",
                "description": "Determine how many incidents occurred in Piper aircrafts",
                "countUnique": False,
                "field": None,
                "inputs": [0],
                "node_id": 1,
            },
            "2": {
                "node_type": "SummarizeData",
                "description": "Generate an English response to the question",
                "question": "How many Piper aircrafts were involved in accidents?",
                "inputs": [1],
                "node_id": 2,
            },
        },
        "result_node": 2,
        "llm_prompt": None,
        "llm_plan": None,
    }
    return LogicalPlan.model_validate(json_plan)


def vector_search_filter_plan_without_opensearch_filter():
    json_plan = {
        "query": "How many incidents involving Piper Aircrafts",
        "nodes": {
            "0": {
                "node_type": "QueryVectorDatabase",
                "node_id": 0,
                "description": "Get all the airplane incidents involving Piper Aircrafts",
                "index": "ntsb",
                "inputs": [],
                "query_phrase": "piper aircrafts",
            },
            "1": {
                "node_type": "Count",
                "description": "Count the number of incidents",
                "node_id": 1,
                "inputs": [0],
            },
            "2": {
                "node_type": "SummarizeData",
                "description": "Generate an English response to the question",
                "question": "How many Piper aircrafts were involved in accidents?",
                "node_id": 2,
                "inputs": [1],
            },
        },
        "result_node": 2,
    }
    return LogicalPlan.model_validate(json_plan)


def test_postprocess_plan():
    llm_client = DummyLLMClient

    plan1 = vector_search_filter_plan_with_opensearch_filter()
    plan2 = vector_search_filter_plan_without_opensearch_filter()

    for index, plan in enumerate([plan1, plan2]):
        print(plan.nodes)
        copy_plan = plan.model_copy()
        modified_plan = postprocess_plan(copy_plan, llm_client)

        print(modified_plan.nodes)

        assert len(modified_plan.nodes) == 4

        assert modified_plan.nodes[0].node_type == "QueryDatabase"
        if index == 0:
            assert "match" in modified_plan.nodes[0].query
        else:
            assert "match_all" in modified_plan.nodes[0].query

        assert modified_plan.nodes[1].node_type == "LlmFilter"
        assert modified_plan.nodes[1].node_id == 1
        assert modified_plan.nodes[1].field == "text_representation"
        assert modified_plan.nodes[1].inputs[0] == 0
