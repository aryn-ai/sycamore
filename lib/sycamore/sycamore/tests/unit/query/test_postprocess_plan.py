from sycamore.query.planner import postprocess_plan


class DummyLLMClient:
    def generate(prompt_kwargs, llm_kwargs):
        return "Dummy response from an LLM Client"


def vector_search_filter_plan_with_opensearch_filter():
    return [
        {
            "operatorName": "QueryVectorDatabase",
            "description": "Get all the airplane incidents in California",
            "index": "ntsb",
            "query_phrase": "Get all the airplane incidents",
            "opensearch_filter": {"match": {"properties.entity.location": "California"}},
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


def vector_search_filter_plan_without_opensearch_filter():
    return [
        {
            "operatorName": "QueryVectorDatabase",
            "description": "Get all the airplane incidents",
            "index": "ntsb",
            "query_phrase": "Get all the airplane incidents",
            "node_id": 0,
        },
        {
            "operatorName": "LlmExtractEntity",
            "description": "Extract the salient reasons for the incidents",
            "field": "text_representation",
            "new_field": "salient_reasons",
            "new_field_type": "string",
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


def test_postprocess_plan():
    llm_client = DummyLLMClient

    for index, plan in enumerate(
        [vector_search_filter_plan_without_opensearch_filter(), vector_search_filter_plan_with_opensearch_filter()]
    ):
        modified_plan = postprocess_plan(plan, llm_client)

        assert len(modified_plan) == 4

        assert modified_plan[0]["operatorName"] == "QueryDatabase"
        if index == 0:
            assert "match_all" in modified_plan[0]["query"]
        else:
            assert "match" in modified_plan[0]["query"]

        assert modified_plan[1]["operatorName"] == "LlmFilter"
        assert modified_plan[1]["node_id"] == 1
        assert modified_plan[1]["field"] == "text_representation"
        assert modified_plan[1]["input"][0] == 0

        for field_name in plan[1]:
            if field_name not in ["input", "node_id"]:
                assert plan[1][field_name] == modified_plan[2][field_name]
            elif field_name == "input":
                assert modified_plan[2][field_name][0] == 1
            else:
                assert modified_plan[2][field_name] == 2

        for field_name in plan[2]:
            if field_name not in ["input", "node_id"]:
                assert plan[2][field_name] == modified_plan[3][field_name]
            elif field_name == "input":
                assert modified_plan[3][field_name][0] == 2
            else:
                assert modified_plan[3][field_name] == 3
