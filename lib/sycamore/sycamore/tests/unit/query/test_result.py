from unittest.mock import patch, MagicMock
import pytest

import sycamore
from sycamore.data import Document, MetadataDocument
from sycamore.query.result import SycamoreQueryResult, NodeExecution
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.math import Math


@pytest.fixture
def fake_docset_with_scores():
    """Return a docset with some fake documents with ranking scores."""
    doc_dicts = [
        {"doc_id": 1, "properties": {"path": "path1.txt", "_rerank_score": 1, "score": 10}},
        {"doc_id": 2, "properties": {"path": "path2.txt", "_rerank_score": 5}},
        {"doc_id": 3, "properties": {"path": "path3.txt", "score": 3}},
        {"doc_id": 4, "properties": {"path": "path4.txt", "score": 9}},
        {"doc_id": 5, "properties": {"score": 4}},
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


@pytest.fixture
def fake_docset_with_metadata():
    """Return a docset with some metadata in it."""
    docs = [
        Document({"doc_id": "5", "properties": {"path": "path5.txt"}}),
        Document({"doc_id": "6", "properties": {"path": "path6.txt"}}),
        MetadataDocument(key="value"),
    ]
    context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
    docset = context.read.document(docs)
    return docset


@pytest.fixture
def tmpdir():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        yield d


def simple_plan() -> LogicalPlan:
    return LogicalPlan(
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
                distinct_field=None,
                inputs=[0],
            ),
        },
    )


def complex_plan() -> LogicalPlan:
    return LogicalPlan(
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
            2: Count(node_id=2, description="Test count 1", distinct_field=None, inputs=[1]),
            3: LlmFilter(
                node_id=3,
                description="Filter by test_field_2",
                inputs=[0],
                field="test_field_2",
                question="test_question",
            ),
            4: Count(node_id=4, description="Test count 2", distinct_field=None, inputs=[3]),
            5: Math(node_id=5, description="Sum counts", operation="add", inputs=[2, 4]),
        },
    )


def forked_plan() -> LogicalPlan:
    return LogicalPlan(
        query="Dummy query",
        result_node=5,
        nodes={
            0: QueryDatabase(
                node_id=0,
                description="Get all the airplane incidents",
                index="ntsb",
                query={"match_all": {}},
            ),
            2: Count(node_id=2, description="Test count 1", distinct_field=None, inputs=[0]),
            4: Count(node_id=4, description="Test count 2", distinct_field=None, inputs=[0]),
            5: Math(node_id=5, description="Sum counts", operation="add", inputs=[2, 4]),
        },
    )


def test_get_source_docs(fake_docset_with_scores):
    """Test that get_source_docs returns the correct set of documents."""

    plan = simple_plan()

    execution = {
        0: NodeExecution(node_id=0, trace_dir="test_trace_dir_0"),
        1: NodeExecution(node_id=1, trace_dir="test_trace_dir_1"),
    }

    def mock_materialize_result(trace_dir):
        if trace_dir == "test_trace_dir_1":
            return fake_docset_with_scores
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

    assert ["path2.txt", "path1.txt", "path4.txt", "path3.txt"] == retrieved_docs


def test_get_source_docs_complex(fake_docset_with_scores, fake_docset_2):
    """Test that get_source_docs returns the correct set of documents for a complex query plan."""

    plan = complex_plan()

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
            return fake_docset_with_scores
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

    assert mock_context.read.materialize.call_count == 2
    assert retrieved_docs == ["path2.txt", "path1.txt", "path4.txt", "path3.txt", "path5.txt", "path6.txt"]


def test_get_metadata_simpleplan(fake_docset_with_metadata, tmpdir):
    fake_docset_with_metadata.materialize(path=tmpdir).execute()
    plan = simple_plan()
    execution = {
        0: NodeExecution(node_id=0, trace_dir=tmpdir),
        1: NodeExecution(node_id=1, trace_dir="no_trace"),
    }
    result = SycamoreQueryResult(
        query_id="test_get_metadata_simple", plan=plan, result="Test get metadata", execution=execution
    )
    metadata = result.get_metadata()
    assert len(metadata) == 1
    metadata_list = metadata[0]
    assert len(metadata_list) == 1
    assert metadata_list[0].metadata["key"] == "value"


def test_get_metadata_complexplan(fake_docset_with_metadata, tmpdir):
    fake_docset_with_metadata.materialize(path=tmpdir).execute()
    plan = complex_plan()

    execution = {
        0: NodeExecution(node_id=0, trace_dir="dont_see_this"),
        1: NodeExecution(node_id=1, trace_dir=tmpdir),
        2: NodeExecution(node_id=2, trace_dir=None),
        3: NodeExecution(node_id=3, trace_dir=tmpdir),
        4: NodeExecution(node_id=4, trace_dir=None),
        5: NodeExecution(node_id=5, trace_dir=None),
    }
    result = SycamoreQueryResult(
        query_id="test_get_metadata_complex", plan=plan, result="Test get metadata", execution=execution
    )
    metadata = result.get_metadata()
    assert isinstance(metadata, dict)
    assert len(metadata) == 2
    md1 = metadata[1]
    md2 = metadata[3]
    assert len(md1) == 1
    assert md1[0].metadata["key"] == "value"
    assert len(md2) == 1
    assert md2[0].metadata["key"] == "value"


def test_get_metadata_forked(fake_docset_with_metadata, tmpdir):
    fake_docset_with_metadata.materialize(path=tmpdir).execute()
    plan = forked_plan()

    execution = {
        0: NodeExecution(node_id=0, trace_dir=tmpdir),
        2: NodeExecution(node_id=2, trace_dir=None),
        4: NodeExecution(node_id=4, trace_dir=None),
        5: NodeExecution(node_id=5, trace_dir=None),
    }
    result = SycamoreQueryResult(
        query_id="test_get_metadata_simple", plan=plan, result="Test get metadata", execution=execution
    )
    metadata = result.get_metadata()
    assert len(metadata) == 1
    metadata_list = metadata[0]
    assert len(metadata_list) == 1
    assert metadata_list[0].metadata["key"] == "value"
