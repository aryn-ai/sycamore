from unittest.mock import MagicMock
import pytest

from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.planner import LlmPlanner
from sycamore.query.planner_prompt import PlannerPrompt
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaField


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
def mock_schema() -> OpenSearchSchema:
    return OpenSearchSchema(
        fields={
            "incidentId": OpenSearchSchemaField(field_type="str", examples=["A1234, B1234, C1234"]),
            "date": OpenSearchSchemaField(field_type="str", examples=["2022-01-01", "2024-02-10"]),
        }
    )


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
    prompt = planner.generate_prompt("Test query")
    system = prompt.messages[0].content
    assert "The last step of each plan *MUST* be a **SummarizeData** operation" in system

    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        natural_language_response=False,
    )

    prompt = planner.generate_prompt("Test query")
    system = prompt.messages[0].content
    assert "The last step of each plan should return the raw data" in system


def test_generate_prompt_with_custom_prompt(
    mock_schema,
    mock_os_config,
    mock_os_client,
    mock_llm_client,
):
    class CustomPrompt(PlannerPrompt):
        def __init__(self, tail: str):
            super().__init__()
            self.tail = tail

        def generate_system_prompt(self, _query: str) -> str:
            return super().generate_system_prompt(_query) + self.tail

    index = "dummy"
    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        prompt=CustomPrompt("OOGABOOGA"),
    )
    prompt = planner.generate_prompt("Test query")
    system = prompt.messages[0].content
    assert system.endswith("OOGABOOGA")
    assert "The last step of each plan *MUST* be a **SummarizeData** operation" in system


def test_llm_planner(mock_os_config, mock_os_client, mock_llm_client, mock_schema, monkeypatch):
    index = "test_index"

    llm_plan = LogicalPlan(
        query="Dummy query",
        result_node=1,
        nodes={
            0: QueryDatabase(
                node_id=0,
                description="Get all the airplane incidents",
                index="ntsb",
                query={"match_all": {}},
            ),
            1: Count(
                node_id=1,
                description="Count the number of incidents",
                distinct_field=None,
                inputs=[0],
            ),
        },
    )
    llm_plan.llm_plan = llm_plan.model_dump_json()

    # Mock the llm to return a static JSON object
    mock_llm_client.generate.return_value = llm_plan.model_dump_json()

    planner = LlmPlanner(
        index,
        data_schema=mock_schema,
        os_config=mock_os_config,
        os_client=mock_os_client,
        llm_client=mock_llm_client,
        natural_language_response=True,
    )

    plan = planner.plan("Dummy query")

    assert plan.query == "Dummy query"
    assert plan.result_node == 1
    assert len(plan.nodes) == 2
    assert plan.nodes[0].model_dump() == llm_plan.nodes[0].model_dump()
    assert plan.nodes[1].model_dump() == llm_plan.nodes[1].model_dump()
