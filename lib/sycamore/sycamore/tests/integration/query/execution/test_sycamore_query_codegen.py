import contextlib
import io

from sycamore.query.client import SycamoreQueryClient


class TestSycamoreQuery:

    def test_simple(self, query_integration_test_index: str):
        """
        Simple test that ensures we can generate code for a query end to end and get a text response.
        """
        client = SycamoreQueryClient()

        result = client.query("How many incidents happened in california?", query_integration_test_index, dry_run=True)

        print(result)
        assert isinstance(result, str)
        assert len(result) > 0

        exec(result)

    def test_forked(self, query_integration_test_index: str):
        """
        Test that has a fork in the DAG, ensures we can codegen and support multiple execution paths.
        """
        client = SycamoreQueryClient()

        result = client.query(
            "What percentage of all incidents happened in california", query_integration_test_index, dry_run=True
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(result)
        assert "%" in output.getvalue()
