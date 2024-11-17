import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from queryserver.main import app, Query
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.result import SycamoreQueryResult
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaField


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_schema():
    return OpenSearchSchema(fields={"text": OpenSearchSchemaField(field_type="string")})


@pytest.fixture
def mock_indices():
    return ["test_index1", "test_index2"]


def test_list_indices(test_client, mock_schema, mock_indices):
    with patch("queryserver.main.sqclient.get_opensearch_indices", return_value=mock_indices), patch(
        "queryserver.main.sqclient.get_opensearch_schema", return_value=mock_schema
    ):
        response = test_client.get("/v1/indices")
        assert response.status_code == 200

        indices = response.json()
        assert len(indices) == 2
        assert {index["index"] for index in indices} == {"test_index1", "test_index2"}
        assert indices[0]["index_schema"] == mock_schema.model_dump()
        assert indices[1]["index_schema"] == mock_schema.model_dump()


def test_get_index(test_client, mock_schema):
    with patch("queryserver.main.sqclient.get_opensearch_schema", return_value=mock_schema):
        response = test_client.get("/v1/index/test_index1")
        assert response.status_code == 200

        index = response.json()
        assert index["index"] == "test_index1"
        assert index["index_schema"] == mock_schema.model_dump()


def test_generate_plan(test_client, mock_schema):
    mock_plan = LogicalPlan(query="test query", result_node=1, nodes={})
    with patch("queryserver.main.sqclient.generate_plan", return_value=mock_plan):
        with patch("queryserver.main.sqclient.get_opensearch_schema", return_value=mock_schema):
            query = Query(query="test query", index="test_index")
            response = test_client.post("/v1/plan", json=query.model_dump())

        assert response.status_code == 200
        assert response.json() == mock_plan.model_dump()


def test_generate_plan_with_no_query(test_client):
    response = test_client.post("/v1/plan", json={})
    assert response.status_code == 422


def test_generate_plan_with_plan(test_client):
    response = test_client.post("/v1/plan", json={"query": "test query", "plan": "test plan"})
    assert response.status_code == 422


def test_query_with_no_query_or_plan(test_client):
    response = test_client.post("/v1/query", json={})
    assert response.status_code == 422


def test_query_with_query_and_plan(test_client):
    response = test_client.post("/v1/query", json={"query": "test query", "plan": "test plan"})
    assert response.status_code == 422


def test_run_plan(test_client):
    mock_plan = LogicalPlan(query="test query", result_node=1, nodes={})

    mock_result = MagicMock(spec=SycamoreQueryResult)
    mock_result.query_id = "test_id"
    mock_result.plan = mock_plan
    mock_result.result = "test result"
    mock_result.retrieved_docs.return_value = ["doc1.txt", "doc2.txt"]

    with patch("queryserver.main.sqclient.run_plan", return_value=mock_result):
        query = Query(plan=mock_plan, index="test_index1")
        response = test_client.post("/v1/query", json=query.model_dump())

        assert response.status_code == 200
        result = response.json()
        assert result["query_id"] == "test_id"
        assert result["plan"] == mock_plan.model_dump()
        assert result["result"] == "test result"
        assert set(result["retrieved_docs"]) == {"doc1.txt", "doc2.txt"}


def test_run_query(test_client, mock_schema):
    mock_plan = LogicalPlan(query="test query", result_node=1, nodes={})

    class MockSycamoreQueryResult(SycamoreQueryResult):
        def retrieved_docs(self):
            return {"doc1.txt", "doc2.txt"}

    mock_result = MagicMock(spec=SycamoreQueryResult)
    mock_result.query_id = "test_id"
    mock_result.plan = mock_plan
    mock_result.result = "test result"
    mock_result.retrieved_docs.return_value = ["doc1.txt", "doc2.txt"]

    with patch("queryserver.main.sqclient.generate_plan", return_value=mock_plan), patch(
        "queryserver.main.sqclient.run_plan", return_value=mock_result
    ), patch(
        "queryserver.main.sqclient.get_opensearch_schema",
        return_value=mock_schema,
    ):

        query = Query(query="test query", index="test_index")
        response = test_client.post("/v1/query", json=query.model_dump())

        assert response.status_code == 200
        result = response.json()
        assert result["plan"] == mock_plan.model_dump()
        assert result["result"] == "test result"
        assert set(result["retrieved_docs"]) == {"doc1.txt", "doc2.txt"}


def test_run_query_stream(test_client, mock_schema):
    mock_plan = LogicalPlan(query="test query", result_node=1, nodes={})

    class MockSycamoreQueryResult(SycamoreQueryResult):
        def retrieved_docs(self):
            return {"doc1.txt", "doc2.txt"}

    mock_result = MagicMock(spec=SycamoreQueryResult)
    mock_result.query_id = "test_id"
    mock_result.plan = mock_plan
    mock_result.result = "test result"
    mock_result.retrieved_docs.return_value = ["doc1.txt", "doc2.txt"]

    with patch("queryserver.main.sqclient.generate_plan", return_value=mock_plan), patch(
        "queryserver.main.sqclient.run_plan", return_value=mock_result
    ), patch(
        "queryserver.main.sqclient.get_opensearch_schema",
        return_value=mock_schema,
    ):

        query = Query(query="test query", index="test_index", stream=True)
        response = test_client.post("/v1/query", json=query.model_dump())

        assert response.status_code == 200
        retrieved_docs = []
        event = None
        data = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
            if event is not None and data is not None:
                if event == "plan":
                    assert json.loads(data) == mock_plan.model_dump()
                elif event == "result":
                    assert data == "test result"
                elif event == "retrieved_doc":
                    retrieved_docs.append(data)
                elif event != "status":
                    raise ValueError(f"Unknown event: {event}")
                event = None
                data = None

        assert set(retrieved_docs) == {"doc1.txt", "doc2.txt"}
