from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging
from sycamore.tests.integration.query.conftest import OS_CLIENT_ARGS, OS_CONFIG
from sycamore.query.planner import LlmPlanner
from sycamore.schema import SchemaV2 as Schema, make_named_property


def test_simple_llm_planner(query_integration_test_index: str):
    """
    Simple test ensuring nodes are being created and dependencies are being set.
    Using a simple query here for consistent query plans.
    """
    os_client = OpenSearchClientWithLogging(OS_CLIENT_ARGS)

    schema = Schema(
        properties=[
            make_named_property(name="location", type="string", examples=["New York", "Seattle"]),
            make_named_property(name="airplaneType", type="string", examples=["Boeing 747", "Airbus A380"]),
        ]
    )
    planner = LlmPlanner(query_integration_test_index, data_schema=schema, os_config=OS_CONFIG, os_client=os_client)
    plan = planner.plan("How many locations did incidents happen in?")

    assert len(plan.nodes) == 2
    assert type(plan.nodes[0]).__name__ == "QueryDatabase"
    assert type(plan.nodes[1]).__name__ == "Count"

    assert [plan.nodes[0]] == plan.nodes[1].input_nodes()

    # Just ensure we can run the planner with a Schema object as well
    planner = LlmPlanner(query_integration_test_index, data_schema=schema, os_config=OS_CONFIG, os_client=os_client)
    planner.plan("How many locations did incidents happen in?")
