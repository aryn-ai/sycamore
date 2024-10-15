import pytest

from sycamore.query.client import SycamoreQueryClient
from sycamore.query.operators.query_database import QueryVectorDatabase


class TestSycamoreQuery:

    @pytest.mark.parametrize("codegen_mode", [True, False])
    def test_simple(self, query_integration_test_index: str, codegen_mode: bool):
        """
        Simple test that ensures we can run a query end to end and get a text response.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "How many incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        query_id, result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("codegen_mode", [True, False])
    def test_forked(self, query_integration_test_index: str, codegen_mode: bool):
        """
        Test that has a fork in the DAG, ensures we can support multiple execution paths.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "What fraction of all incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        query_id, result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "0" in result

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_dry_run(self, query_integration_test_index: str, dry_run: bool):
        """
        Test that asserts dry_run mode doesn't execute the query. We use ray initialization as a proxy here,
        we can also look at LLM calls but this is a little more reliable in the current state.
        """
        import ray

        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "What fraction of all incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        ray.shutdown()
        query_id, result = client.run_plan(plan, dry_run=dry_run)
        assert isinstance(result, str)
        assert len(result) > 0
        if dry_run:
            assert not ray.is_initialized()
        else:
            assert ray.is_initialized()

    @pytest.mark.parametrize("codegen_mode", [True, False])
    def test_vector_search(self, query_integration_test_index: str, codegen_mode: bool):
        """ """

        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "were there any environmentally caused incidents?",
            query_integration_test_index,
            schema,
            natural_language_response=False,
        )
        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], QueryVectorDatabase)
        query_id, result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result, str)
        assert "No" in result or "no" in result
