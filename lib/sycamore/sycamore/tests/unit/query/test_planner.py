from unittest.mock import MagicMock
import pytest
import json

from sycamore.query.planner import LlmPlanner


@pytest.fixture
def mock_os_config():
    return MagicMock()


@pytest.fixture
def mock_os_client():
    return MagicMock()


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_schema():
    schema = {
        "incidentId": ("string", {"A1234, B1234, C1234"}),
        "date": ("string", {"2022-01-01", "2024-02-10"}),
    }
    return schema


def test_generate_system_prompt(mock_schema):
    index = "test_index"
    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        natural_language_response=True,
    )
    prompt = planner.generate_system_prompt("Test query")
    assert "The last step of each plan *MUST* be a **SummarizeData** operation" in prompt

    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        natural_language_response=False,
    )
    prompt = planner.generate_system_prompt("Test query")
    assert "The last step of each plan should return the raw data" in prompt


def test_llm_planner(mock_os_config, mock_os_client, mock_llm_client, mock_schema, monkeypatch):
    index = "test_index"

    # Mock the generate_from_llm method to return a static JSON object
    def mock_generate_from_llm(self, query):
        return "Mock LLM prompt", json.dumps(
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the airplane incidents",
                    "index": "ntsb",
                    "query": "airplane incidents",
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
        )

    monkeypatch.setattr(LlmPlanner, "generate_from_llm", mock_generate_from_llm)

    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        natural_language_response=True,
    )

    plan = planner.plan("Dummy query")
    assert plan.result_node.node_id == 3
    assert plan.result_node.description == "Generate an English response to the question"
    assert len(plan.result_node.dependencies) == 1
    assert plan.result_node.dependencies[0].node_id == 2
    assert plan.result_node.dependencies[0].description == "Determine how many incidents occurred in Piper aircrafts"
    assert len(plan.result_node.dependencies[0].dependencies) == 1
    assert plan.result_node.dependencies[0].dependencies[0].node_id == 1
    assert (
        plan.result_node.dependencies[0].dependencies[0].description
        == "Filter to only include Piper aircraft incidents"
    )
    assert len(plan.result_node.dependencies[0].dependencies[0].dependencies) == 1
    assert plan.result_node.dependencies[0].dependencies[0].dependencies[0].node_id == 0
    assert (
        plan.result_node.dependencies[0].dependencies[0].dependencies[0].description == "Get all the airplane incidents"
    )
