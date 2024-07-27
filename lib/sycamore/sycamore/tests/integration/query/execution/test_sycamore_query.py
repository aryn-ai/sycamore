from sycamore.query.client import SycamoreQueryClient


class TestSycamoreQuery:

    def test_simple(self, query_integration_test_index: str):
        """
        Simple test that ensures we can run a query end to end and get a text response.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan("How many incidents happened in california?", query_integration_test_index, schema)
        query_id, result = client.run_plan(plan)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_forked(self, query_integration_test_index: str):
        """
        Test that has a fork in the DAG, ensures we can support multiple execution paths.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "What percentage of all incidents happened in california", query_integration_test_index, schema
        )
        query_id, result = client.run_plan(plan)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "%" in result
