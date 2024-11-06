import pytest
from unittest.mock import MagicMock, patch

from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaField

from queryeval.driver import QueryEvalDriver
from queryeval.types import (
    QueryEvalQuery,
    QueryEvalResult,
    QueryEvalMetrics,
)
from sycamore.query.result import SycamoreQueryResult
from sycamore.query.logical_plan import LogicalPlan, Node


@pytest.fixture
def mock_client():
    client = MagicMock()
    schema = OpenSearchSchema(
        fields={
            "field1": OpenSearchSchemaField(field_type="<class 'str'>", description="text"),
            "field2": OpenSearchSchemaField(field_type="<class 'str'>", description="keyword"),
        }
    )
    client.get_opensearch_schema.return_value = schema
    return client


@pytest.fixture
def mock_plan():
    return LogicalPlan(query="test query", nodes={0: Node(node_id=0, node_type="test_op")}, result_node=0)


@pytest.fixture
def test_input_file(tmp_path):
    input_file = tmp_path / "test_input.yaml"
    input_file.write_text(
        """
config:

  index: test-index
  results_file: test-results.yaml
queries:
  - query: "test query 1"
    tags: ["test"]
    expected_docs: ["doc1.pdf", "doc2.pdf"]
  - query: "test query 2"
"""
    )
    return str(input_file)


def test_driver_init(test_input_file, mock_client):
    with patch("queryeval.driver.SycamoreQueryClient", return_value=mock_client):
        driver = QueryEvalDriver(input_file_path=test_input_file, index="test-index", doc_limit=10, tags=["test"])

        assert driver.config.config.index == "test-index"
        assert driver.config.config.doc_limit == 10
        assert driver.config.queries[0].tags == ["test"]
        assert driver.config.queries[1].tags is None
        assert driver.config.queries[0].expected_docs == {"doc1.pdf", "doc2.pdf"}
        assert driver.config.queries[1].expected_docs is None


def test_driver_do_plan(test_input_file, mock_client, mock_plan):
    with patch("queryeval.driver.SycamoreQueryClient", return_value=mock_client):
        driver = QueryEvalDriver(input_file_path=test_input_file)

        query = QueryEvalQuery(query="test query")
        result = QueryEvalResult(query=query, metrics=QueryEvalMetrics())

        mock_client.generate_plan.return_value = mock_plan

        result = driver.do_plan(query, result)
        assert result.plan is not None
        mock_client.generate_plan.assert_called_once()


def test_driver_do_query(test_input_file, mock_client, mock_plan):
    with patch("queryeval.driver.SycamoreQueryClient", return_value=mock_client):
        driver = QueryEvalDriver(input_file_path=test_input_file)

        query = QueryEvalQuery(query="test query")
        result = QueryEvalResult(query=query, plan=mock_plan, metrics=QueryEvalMetrics())

        mock_query_result = SycamoreQueryResult(query_id="test", plan=result.plan, result="test result")
        mock_client.run_plan.return_value = mock_query_result

        result = driver.do_query(query, result)
        driver.eval_all()  # we are only asserting that this runs
        assert result.result == "test result"


def test_driver_do_eval(test_input_file, mock_client, mock_plan):
    with patch("queryeval.driver.SycamoreQueryClient", return_value=mock_client):
        driver = QueryEvalDriver(input_file_path=test_input_file)
        expected_docs = {"doc1.pdf", "doc2.pdf"}
        query = QueryEvalQuery(query="test query", expected_plan=mock_plan, expected_docs=expected_docs)
        result = QueryEvalResult(query=query, plan=mock_plan, retrieved_docs=expected_docs)

        result = driver.do_eval(query, result)
        assert result.metrics.plan_similarity == 1.0
        assert result.metrics.doc_retrieval_recall == 1.0

        result = QueryEvalResult(query=query, plan=mock_plan, retrieved_docs={"doc1.pdf"})

        result = driver.do_eval(query, result)
        assert result.metrics.plan_similarity == 1.0
        assert result.metrics.doc_retrieval_recall == 0.5
