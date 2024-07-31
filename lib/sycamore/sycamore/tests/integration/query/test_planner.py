from opensearchpy import OpenSearch

from sycamore.tests.integration.query.conftest import OS_CLIENT_ARGS, OS_CONFIG
from sycamore.query.planner import LlmPlanner


def test_simple_openai_planner(query_integration_test_index: str):
    """
    Simple test ensuring nodes are being creating and dependencies are being set.
    Using a simple query here for consistent query plans.
    """
    os_client = OpenSearch(OS_CLIENT_ARGS)

    schema = {"location": "string", "airplaneType": "string"}
    planner = LlmPlanner(query_integration_test_index, data_schema=schema, os_config=OS_CONFIG, os_client=os_client)
    plan = planner.plan("How many locations did incidents happen in?")

    assert len(plan.nodes) == 3
    assert type(plan.nodes[0]).__name__ == "LoadData"
    assert type(plan.nodes[1]).__name__ == "Count"
    assert type(plan.nodes[2]).__name__ == "LlmGenerate"

    assert [plan.nodes[1]] == plan.nodes[0]._downstream_nodes
    assert [plan.nodes[2]] == plan.nodes[1]._downstream_nodes

    assert [plan.nodes[0]] == plan.nodes[1]._dependencies
    assert [plan.nodes[1]] == plan.nodes[2]._dependencies
