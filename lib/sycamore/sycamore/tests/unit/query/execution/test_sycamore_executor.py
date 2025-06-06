import os
import tempfile
from unittest.mock import patch
from typing import Optional

import pytest

import sycamore
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.query.execution.sycamore_executor import SycamoreExecutor
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.result import SycamoreQueryResult
from structlog.contextvars import bind_contextvars


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="dummy", default_mode=LLMMode.SYNC)

    def is_chat_mode(self):
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return ""


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


def test_run_summarize_data_plan(mock_sycamore_docsetreader):

    plan = LogicalPlan(
        query="Test query",
        result_node=1,
        nodes={
            0: QueryDatabase(node_id=0, description="Load data", index="test_index"),
            1: SummarizeData(
                node_id=1,
                description="Summarize data",
                question="Summarize this data",
                inputs=[0],
            ),
        },
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader),
            patch("sycamore.tests.unit.query.conftest.MOCK_SCAN_NUM_DOCUMENTS", new=1000),
        ):
            context = sycamore.init(
                params={
                    "default": {"llm": MockLLM()},
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
                    },
                }
            )

            # First run should populate cache.
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            result = executor.execute(plan, query_id="test_query_id")
            assert result.plan == plan
            assert result.query_id == "test_query_id"

            # Check that a directory was created for each node.
            cache_dirs = [os.path.join(temp_dir, node.cache_key()) for node in plan.nodes.values()]
            for cache_dir in cache_dirs:
                assert os.path.exists(cache_dir)

            # Check that the materialized data is complete.
            assert os.path.exists(os.path.join(cache_dirs[0], "materialize.success"))
            assert os.path.exists(os.path.join(cache_dirs[0], "materialize.clean"))
            # 1000 docs + 200 metadata docs + 2 'materialize' files
            assert len(os.listdir(cache_dirs[0])) == 1200 + 2


def test_pause_materialization_after_sort_node(mock_sycamore_docsetreader):
    plan = LogicalPlan(
        query="Test query",
        result_node=4,
        nodes={
            0: QueryDatabase(
                node_id=0,
                description="Load data",
                index="test_index",
            ),
            1: LlmExtractEntity(
                node_id=1,
                description="Extract entity",
                field="text_representation",
                new_field="test_field",
                new_field_type="int",
                question="Extract an integer from this text",
                inputs=[0],
            ),
            2: Sort(
                node_id=2,
                description="Sort data",
                field="test_field",
                default_value=1,
                inputs=[1],
            ),
            3: LlmFilter(
                node_id=3,
                description="Filter data",
                field="test_field",
                question="Filter out documents where test_field is not 1",
                inputs=[2],
            ),
            4: LlmExtractEntity(
                node_id=4,
                description="Extract entity",
                field="text_representation",
                new_field="test_field2",
                new_field_type="float",
                question="Extract a float from this text",
                inputs=[3],
            ),
        },
    )

    # Plan with no Sort Node
    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader),
            patch("sycamore.tests.unit.query.conftest.MOCK_SCAN_NUM_DOCUMENTS", new=1000),
        ):
            context = sycamore.init(
                params={
                    "default": {"llm": MockLLM()},
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
                    },
                }
            )
            # First run should populate cache.
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            bind_contextvars(query_id="test_query_id")

            plan.result_node = 1
            result = SycamoreQueryResult(query_id="test_query_id", plan=plan, result=None)
            res, disable_materialization = executor.process_node(
                plan.nodes[plan.result_node], result, is_result_node=True
            )
            assert (
                not disable_materialization
            ), "Materialization should not be disabled for result node when Sort is not present"
            materialized_node = res.plan.get_plan_nodes(sycamore.materialize.Materialize)[-1]
            assert (
                materialized_node._source_mode == sycamore.MATERIALIZE_USE_STORED
            ), "Incorrect source mode for materialization"

    # Plan where result node is Sort Node
    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader),
            patch("sycamore.tests.unit.query.conftest.MOCK_SCAN_NUM_DOCUMENTS", new=1000),
        ):
            context = sycamore.init(
                params={
                    "default": {"llm": MockLLM()},
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
                    },
                }
            )
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            bind_contextvars(query_id="test_query_id")
            # First run should populate cache.
            plan.result_node = 2
            result = SycamoreQueryResult(query_id="test_query_id", plan=plan, result=None)
            res, disable_materialization = executor.process_node(
                plan.nodes[plan.result_node], result, is_result_node=True
            )
            assert disable_materialization, "Materialization should be disabled for result node when Sort is present"
            materialized_node = res.plan.get_plan_nodes(sycamore.materialize.Materialize)[-1]
            assert (
                materialized_node._source_mode == sycamore.MATERIALIZE_RECOMPUTE
            ), "Incorrect source mode for materialization"

    # Plan where sort node is present in the middle of the plan
    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader),
            patch("sycamore.tests.unit.query.conftest.MOCK_SCAN_NUM_DOCUMENTS", new=1000),
        ):
            context = sycamore.init(
                params={
                    "default": {"llm": MockLLM()},
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
                    },
                },
            )
            executor = SycamoreExecutor(context, cache_dir=temp_dir)
            bind_contextvars(query_id="test_query_id")
            # First run should populate cache.
            plan.result_node = 4
            result = SycamoreQueryResult(query_id="test_query_id", plan=plan, result=None)
            res, disable_materialization = executor.process_node(
                plan.nodes[plan.result_node], result, is_result_node=True
            )
            assert disable_materialization, "Materialization should be disabled for result node when Sort is present"
            materialized_node = res.plan.get_plan_nodes(sycamore.materialize.Materialize)[-1]
            assert (
                materialized_node._source_mode == sycamore.MATERIALIZE_RECOMPUTE
            ), "Incorrect source mode for materialization"
