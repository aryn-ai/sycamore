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
def mock_openai_client():
    return MagicMock()


def test_llm_planner(mock_os_config, mock_os_client, mock_openai_client, monkeypatch):
    index = "test_index"
    schema = {
        "description": "Database of airplane incidents",
        "incidentId": "(string) e.g. A123G73",
        "date": "(string: YYYY-MM-DD) e.g. 2022-01-01, 2024-02-10",
    }

    # Mock the generate_from_openai method to return a static JSON object
    def mock_generate_from_openai(self, query):
        return json.dumps(
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the airplane incidents",
                    "index": "ntsb",
                    "query": "airplane incidents",
                    "id": 0,
                },
                {
                    "operatorName": "LlmFilter",
                    "description": "Filter to only include Piper aircraft incidents",
                    "question": "Did this incident occur in a Piper aircraft?",
                    "field": "properties.entity.aircraft",
                    "input": [0],
                    "id": 1,
                },
                {
                    "operatorName": "Count",
                    "description": "Determine how many incidents occurred in Piper aircrafts",
                    "countUnique": False,
                    "field": None,
                    "input": [1],
                    "id": 2,
                },
                {
                    "operatorName": "LlmGenerate",
                    "description": "Generate an English response to the question",
                    "question": "How many Piper aircrafts were involved in accidents?",
                    "input": [2],
                    "id": 3,
                },
            ]
        )

    monkeypatch.setattr(LlmPlanner, "generate_from_openai", mock_generate_from_openai)

    planner = LlmPlanner(
        index,
        data_schema=schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        openai_client=mock_openai_client,
    )

    plan = planner.plan("Dummy query")
    assert plan.result_node().node_id == 3
    assert plan.result_node().data.get("description") == "Generate an English response to the question"
    assert len(plan.result_node().dependencies) == 1
    assert plan.result_node().dependencies[0].node_id == 2
    assert (
        plan.result_node().dependencies[0].data.get("description")
        == "Determine how many incidents occurred in Piper aircrafts"
    )
    assert len(plan.result_node().dependencies[0].dependencies) == 1
    assert plan.result_node().dependencies[0].dependencies[0].node_id == 1
    assert (
        plan.result_node().dependencies[0].dependencies[0].data.get("description")
        == "Filter to only include Piper aircraft incidents"
    )
    assert len(plan.result_node().dependencies[0].dependencies[0].dependencies) == 1
    assert plan.result_node().dependencies[0].dependencies[0].dependencies[0].node_id == 0
    assert (
        plan.result_node().dependencies[0].dependencies[0].dependencies[0].data.get("description")
        == "Get all the airplane incidents"
    )
