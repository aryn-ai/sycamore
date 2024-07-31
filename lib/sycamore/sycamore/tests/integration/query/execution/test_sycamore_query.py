import contextlib
import io

import pytest

from sycamore.query.client import SycamoreQueryClient


class TestSycamoreQuery:

    @pytest.mark.parametrize("codegen", [True, False])
    def test_simple(self, query_integration_test_index: str, codegen: bool):
        """
        Simple test that ensures we can run a query end to end and get a text response.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan("How many incidents happened in california?", query_integration_test_index, schema)
        query_id, result = client.run_plan(plan, dry_run=codegen)
        assert isinstance(result, str)
        assert len(result) > 0
        if codegen:
            print(result)
            exec(result)

    @pytest.mark.parametrize("codegen", [True, False])
    def test_forked(self, query_integration_test_index: str, codegen: bool):
        """
        Test that has a fork in the DAG, ensures we can support multiple execution paths.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema("ntsb")
        plan = client.generate_plan(
            "What percent of  environmentally caused incidents were due to wind?", "ntsb", schema
        )
        plan.show()
        query_id, result = client.run_plan(plan, dry_run=codegen)
        assert isinstance(result, str)
        assert len(result) > 0
        if codegen:
            print(result)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(result)
            result = output.getvalue()
        assert "%" in result
