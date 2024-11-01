from unittest.mock import patch, MagicMock
import pytest

import sycamore
from sycamore.data import Document
from sycamore.query.result import SycamoreQueryResult, NodeExecution
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.math import Math


@pytest.fixture
def fake_docset():
    """Return a docset with some fake documents."""
    doc_dicts = [
        {"doc_id": 1, "properties": {"path": "path1.txt"}},
        {"doc_id": 2, "properties": {"path": "path2.txt"}},
        {"doc_id": 3, "properties": {"path": "path3.txt"}},
        {"doc_id": 4, "properties": {}},
    ]
    context = sycamore.init()
    docset = context.read.document([Document(d) for d in doc_dicts])
    return docset


@pytest.fixture
def fake_docset_2():
    """Return a second docset with some fake documents."""
    doc_dicts = [
        {"doc_id": 5, "properties": {"path": "path5.txt"}},
        {"doc_id": 6, "properties": {"path": "path6.txt"}},
        {"doc_id": 3, "properties": {"path": "path3.txt"}},  # Intentional Duplicate.
        {"doc_id": 8, "properties": {}},
    ]
    context = sycamore.init()
    docset = context.read.document([Document(d) for d in doc_dicts])
    return docset


def test_get_source_docs(fake_docset):
    """Test that get_source_docs returns the correct set of documents."""

    plan = LogicalPlan(
        query="Dummy query",
        result_node=1,
        nodes={
            0: QueryDatabase(
                node_id=0,
                description="Get all the airplane incidents",
                index="ntsb",
                query={"match_all": {}},
            ),
            1: Count(
                node_id=1,
                description="Count the number of incidents",
                field=None,
                inputs=[0],
            ),
        },
    )

    execution = {
        0: NodeExecution(node_id=0, trace_dir="test_trace_dir_0"),
        1: NodeExecution(node_id=1, trace_dir="test_trace_dir_1"),
    }

    def mock_materialize_result(trace_dir):
        if trace_dir == "test_trace_dir_1":
            return fake_docset
        else:
            return None

    context = sycamore.init()
    mock_context = MagicMock(wraps=context)
    mock_context.exec_mode = sycamore.ExecMode.RAY
    mock_context.read.materialize.side_effect = mock_materialize_result
    with patch("sycamore.init", return_value=mock_context):
        result = SycamoreQueryResult(
            query_id="test_query_id", plan=plan, result="Test query result", execution=execution
        )
        retrieved_docs = result.retrieved_docs()

    assert retrieved_docs == {"path1.txt", "path2.txt", "path3.txt"}


def test_get_source_docs_complex(fake_docset, fake_docset_2):
    """Test that get_source_docs returns the correct set of documents for a complex query plan."""

    plan = LogicalPlan(
        query="Dummy query",
        result_node=5,
        nodes={
            0: QueryDatabase(
                node_id=0,
                description="Get all the airplane incidents",
                index="ntsb",
                query={"match_all": {}},
            ),
            1: LlmFilter(
                node_id=1,
                description="Filter by test_field_1",
                inputs=[0],
                field="test_field_1",
                question="test_question",
            ),
            2: Count(node_id=2, description="Test count 1", field=None, inputs=[1]),
            3: LlmFilter(
                node_id=3,
                description="Filter by test_field_2",
                inputs=[0],
                field="test_field_2",
                question="test_question",
            ),
            4: Count(node_id=4, description="Test count 2", field=None, inputs=[3]),
            5: Math(node_id=5, description="Sum counts", operation="add", inputs=[2, 4]),
        },
    )

    execution = {
        0: NodeExecution(node_id=0, trace_dir="test_trace_dir_0"),
        1: NodeExecution(node_id=1, trace_dir="test_trace_dir_1"),
        2: NodeExecution(node_id=2, trace_dir=None),
        3: NodeExecution(node_id=3, trace_dir="test_trace_dir_3"),
        4: NodeExecution(node_id=4, trace_dir=None),
        5: NodeExecution(node_id=5, trace_dir=None),
    }

    def mock_materialize_result(trace_dir):
        if trace_dir == "test_trace_dir_1":
            return fake_docset
        elif trace_dir == "test_trace_dir_3":
            return fake_docset_2
        else:
            return None

    context = sycamore.init()
    mock_context = MagicMock(wraps=context)
    mock_context.exec_mode = sycamore.ExecMode.RAY
    mock_context.read.materialize.side_effect = mock_materialize_result
    with patch("sycamore.init", return_value=mock_context):
        result = SycamoreQueryResult(
            query_id="test_query_id", plan=plan, result="Test query result", execution=execution
        )
        retrieved_docs = result.retrieved_docs()

    print(f"\nMDW: retrieved_docs is {retrieved_docs}")

    assert mock_context.read.materialize.call_count == 2
    assert retrieved_docs == {"path1.txt", "path2.txt", "path3.txt", "path5.txt", "path6.txt"}
