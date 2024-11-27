import os
import pytest
from unittest import mock
from sycamore.data import Document
from sycamore.data.document import DocumentPropertyTypes, DocumentSource

from sycamore.connectors.duckdb.duckdb_reader import (
    DuckDBReaderClient,
    DuckDBReaderClientParams,
    DuckDBReaderQueryParams,
    DuckDBReaderQueryResponse,
    DuckDBReader,
)


@pytest.fixture
def mock_duckdb_connection():
    with mock.patch("duckdb.connect") as mock_connect:
        mock_conn = mock.Mock()
        mock_connect.return_value = mock_conn
        yield mock_conn


def test_duckdb_reader_client_from_client_params(mock_duckdb_connection):
    os.makedirs("tmp", exist_ok=True)
    params = DuckDBReaderClientParams(db_url="tmp/test_db")
    with mock.patch("duckdb.connect", return_value=mock_duckdb_connection) as mock_connect:
        client = DuckDBReaderClient.from_client_params(params)
        assert isinstance(client, DuckDBReaderClient)
        mock_connect.assert_called_once_with(database="tmp/test_db", read_only=True)


def test_duckdb_reader_client_read_records(mock_duckdb_connection):
    client = DuckDBReaderClient(mock_duckdb_connection)
    query_params = DuckDBReaderQueryParams(table_name="test_table", query=None, create_hnsw_table=None)
    mock_duckdb_connection.execute.return_value = mock.Mock()
    response = client.read_records(query_params)
    assert isinstance(response, DuckDBReaderQueryResponse)
    mock_duckdb_connection.execute.assert_called_once_with("SELECT * from test_table")


def test_duckdb_reader_client_read_records_with_query(mock_duckdb_connection):
    client = DuckDBReaderClient(mock_duckdb_connection)
    query_params = DuckDBReaderQueryParams(
        table_name="test_table", query="SELECT * FROM test_table", create_hnsw_table=None
    )
    mock_duckdb_connection.execute.return_value = mock.Mock()
    response = client.read_records(query_params)
    assert isinstance(response, DuckDBReaderQueryResponse)
    mock_duckdb_connection.execute.assert_called_once_with("SELECT * FROM test_table")


def test_duckdb_reader_client_read_records_with_create_hnsw_table(mock_duckdb_connection):
    client = DuckDBReaderClient(mock_duckdb_connection)
    query_params = DuckDBReaderQueryParams(
        table_name="test_table", query=None, create_hnsw_table="CREATE TABLE hnsw AS SELECT * FROM test_table"
    )
    mock_duckdb_connection.execute.return_value = mock.Mock()
    response = client.read_records(query_params)
    assert isinstance(response, DuckDBReaderQueryResponse)
    mock_duckdb_connection.execute.assert_any_call("CREATE TABLE hnsw AS SELECT * FROM test_table")
    mock_duckdb_connection.execute.assert_any_call("SELECT * from test_table")


def test_duckdb_reader_client_check_target_presence(mock_duckdb_connection):
    client = DuckDBReaderClient(mock_duckdb_connection)
    query_params = DuckDBReaderQueryParams(table_name="test_table", query=None, create_hnsw_table=None)
    mock_duckdb_connection.sql.return_value = mock.Mock()
    assert client.check_target_presence(query_params) is True
    mock_duckdb_connection.sql.assert_called_once_with("SELECT * FROM test_table")


def test_duckdb_reader_client_check_target_presence_not_found(mock_duckdb_connection):
    client = DuckDBReaderClient(mock_duckdb_connection)
    query_params = DuckDBReaderQueryParams(table_name="non_existent_table", query=None, create_hnsw_table=None)
    mock_duckdb_connection.sql.side_effect = Exception("Table not found")
    assert client.check_target_presence(query_params) is False
    mock_duckdb_connection.sql.assert_called_once_with("SELECT * FROM non_existent_table")


def test_duckdb_reader_query_response_to_docs():
    mock_output = mock.Mock()
    mock_output.df.return_value.to_dict.return_value = [{"properties": {"key": "value"}, "embedding": 0.0}]
    response = DuckDBReaderQueryResponse(output=mock_output)
    query_params = DuckDBReaderQueryParams(table_name="test_table", query=None, create_hnsw_table=None)
    docs = response.to_docs(query_params)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].properties[DocumentPropertyTypes.SOURCE] == DocumentSource.DB_QUERY
    assert docs[0].properties["key"] == "value"
    assert docs[0].embedding == []


def test_duckdb_reader_query_response_to_docs_empty():
    mock_output = mock.Mock()
    mock_output.df.return_value.to_dict.return_value = []
    response = DuckDBReaderQueryResponse(output=mock_output)
    query_params = DuckDBReaderQueryParams(table_name="test_table", query=None, create_hnsw_table=None)
    docs = response.to_docs(query_params)
    assert len(docs) == 0


def test_duckdb_reader():
    assert DuckDBReader.Client == DuckDBReaderClient
    assert DuckDBReader.Record == DuckDBReaderQueryResponse
    assert DuckDBReader.ClientParams == DuckDBReaderClientParams
    assert DuckDBReader.QueryParams == DuckDBReaderQueryParams
