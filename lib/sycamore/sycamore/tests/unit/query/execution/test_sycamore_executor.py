import os
import tempfile
from unittest.mock import patch

import pytest

import sycamore
from sycamore.query.execution.sycamore_executor import SycamoreExecutor
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.result import SycamoreQueryResult


@pytest.fixture
def test_count_docs_query_plan() -> LogicalPlan:
    """A simple query plan which only counts the number of documents."""

    plan = LogicalPlan(
        query="Test query",
        result_node=1,
        nodes={
            0: QueryDatabase(node_id=0, description="Load data", index="test_index"),
            1: Count(node_id=1, description="Count number of documents", inputs=[0]),
        },
    )
    return plan


def test_run_plan(test_count_docs_query_plan, mock_sycamore_docsetreader, mock_opensearch_num_docs):
    with patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader):
        context = sycamore.init(
            params={
                "opensearch": {
                    "os_client_args": {
                        "hosts": [{"host": "localhost", "port": 9200}],
                        "http_compress": True,
                        "http_auth": ("admin", "admin"),
                        "use_ssl": True,
                        "verify_certs": False,
                        "ssl_assert_hostname": False,
                        "ssl_show_warn": False,
                        "timeout": 120,
                    }
                }
            }
        )

        executor = SycamoreExecutor(context)
        result = executor.execute(test_count_docs_query_plan, query_id="test_query_id")
        expected = SycamoreQueryResult(
            plan=test_count_docs_query_plan,
            result=mock_opensearch_num_docs,
            query_id="test_query_id",
        )
        assert result == expected


def test_run_plan_with_caching(test_count_docs_query_plan, mock_sycamore_docsetreader, mock_opensearch_num_docs):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader):
            context = sycamore.init(
                params={
                    "opensearch": {
                        "os_client_args": {
                            "hosts": [{"host": "localhost", "port": 9200}],
                            "http_compress": True,
                            "http_auth": ("admin", "admin"),
                            "use_ssl": True,
                            "verify_certs": False,
                            "ssl_assert_hostname": False,
                            "ssl_show_warn": False,
                            "timeout": 120,
                        }
                    }
                }
            )

            # First run should populate cache.
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            result = executor.execute(test_count_docs_query_plan, query_id="test_query_id")
            expected = SycamoreQueryResult(
                plan=test_count_docs_query_plan,
                result=mock_opensearch_num_docs,
                query_id="test_query_id",
                execution=result.execution,
            )
            assert result == expected

            # Check that a directory was created for each node.
            cache_dirs = [
                os.path.join(temp_dir, node.cache_key()) for node in test_count_docs_query_plan.nodes.values()
            ]
            for cache_dir in cache_dirs:
                assert os.path.exists(cache_dir)

            # Second run should use the cache.
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            result = executor.execute(test_count_docs_query_plan, query_id="test_query_id")
            assert result == expected

            # No new directories should have been created.
            existing_dirs = [os.path.join(temp_dir, x) for x in os.listdir(temp_dir)]
            assert set(existing_dirs) == set(cache_dirs)
