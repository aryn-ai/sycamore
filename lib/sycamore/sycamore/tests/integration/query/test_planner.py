from typing import List

from opensearchpy import OpenSearch

from sycamore.tests.integration.query.conftest import OS_CLIENT_ARGS, OS_CONFIG
from sycamore.query.logical_plan import Node
from sycamore.query.planner import LlmPlanner


def test_simple_openai_planner(query_integration_test_index: str):
    """
    Simple test ensuring nodes are being creating and dependencies are being set.
    Using a simple query here for consistent query plans.
    """
    os_client = OpenSearch(**OS_CLIENT_ARGS)

    schema = {"location": "string", "airplaneType": "string"}
    planner = LlmPlanner(query_integration_test_index, data_schema=schema, os_config=OS_CONFIG, os_client=os_client)
    plan = planner.plan("How many locations did incidents happen in?")
    plan.show()

    nodes: List[Node] = plan.nodes()

    assert len(nodes) == 3
    assert type(nodes[0]).__name__ == "LoadData"
    assert type(nodes[1]).__name__ == "Count"
    assert type(nodes[2]).__name__ == "LlmGenerate"

    assert [nodes[1]] == nodes[0].downstream_nodes
    assert [nodes[2]] == nodes[1].downstream_nodes

    assert [nodes[0]] == nodes[1].dependencies
    assert [nodes[1]] == nodes[2].dependencies
