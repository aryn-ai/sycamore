import pytest
from unittest import mock
from sycamore.connectors.duckdb.duckdb_writer import (
    DuckDBWriterClientParams,
    DuckDBWriterTargetParams,
    DuckDBClient,
    DuckDBDocumentRecord,
)
from sycamore.data.document import Document
from sycamore.connectors.duckdb.duckdb_writer import _narrow_list_of_doc_records
from sycamore.connectors.base_writer import BaseDBWriter


@pytest.fixture
def mock_duckdb(mocker):
    mock_duckdb = mocker.patch("duckdb.connect")
    mock_conn = mocker.Mock()
    mock_duckdb.return_value = mock_conn
    return mock_conn


@pytest.fixture
def target_params():
    return DuckDBWriterTargetParams(dimensions=128)


@pytest.fixture
def client_params():
    return DuckDBWriterClientParams()


@pytest.fixture
def duckdb_client(client_params):
    return DuckDBClient(client_params)


@pytest.fixture
def document():
    return Document(
        doc_id="doc1",
        parent_id="parent1",
        properties={"key": "value"},
        type="type1",
        text_representation="text",
        bbox=(0.0, 0.0, 1.0, 1.0),
        shingles=[1, 2, 3],
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


def test_duckdb_client_init(client_params):
    client = DuckDBClient(client_params)
    assert isinstance(client, DuckDBClient)


def test_duckdb_client_from_client_params(client_params):
    client = DuckDBClient.from_client_params(client_params)
    assert isinstance(client, DuckDBClient)


def test_duckdb_writer_create_target_idempotent(mock_duckdb, target_params):
    client = DuckDBClient(DuckDBWriterClientParams())
    client.create_target_idempotent(target_params)
    mock_duckdb.sql.assert_called_once()


def test_duckdb_writer_write_many_records(mock_duckdb, target_params, document):
    client = DuckDBClient(DuckDBWriterClientParams())
    records = [DuckDBDocumentRecord.from_doc(document, target_params)]
    client.write_many_records(records, target_params)
    assert mock_duckdb.sql.call_count > 0


def test_duckdb_writer_get_existing_target_params(mock_duckdb, target_params):
    client = DuckDBClient(DuckDBWriterClientParams())
    mock_duckdb.sql.return_value = mock.Mock(columns=["doc_id"], dtypes=["VARCHAR"])
    params = client.get_existing_target_params(target_params)
    assert params.schema == {"doc_id": "VARCHAR"}


def test_duckdb_document_record_from_doc(document, target_params):
    record = DuckDBDocumentRecord.from_doc(document, target_params)
    assert record.doc_id == document.doc_id
    assert record.parent_id == document.parent_id
    assert record.properties == document.properties
    assert record.type == document.type
    assert record.text_representation == document.text_representation
    assert record.bbox == document.bbox.coordinates
    assert record.shingles == document.shingles
    assert record.embedding == document.embedding


def test_narrow_list_of_doc_records(document, target_params):
    records = [DuckDBDocumentRecord.from_doc(document, target_params)]
    assert _narrow_list_of_doc_records(records)


def test_duckdb_writer_target_params_compatible_with():
    params1 = DuckDBWriterTargetParams(dimensions=128)
    params2 = DuckDBWriterTargetParams(dimensions=128)
    assert params1.compatible_with(params2)

    params3 = DuckDBWriterTargetParams(dimensions=64)
    assert not params1.compatible_with(params3)

    params4 = DuckDBWriterTargetParams(dimensions=128, db_url="different.db")
    assert not params1.compatible_with(params4)

    params5 = DuckDBWriterTargetParams(dimensions=128, table_name="different_table")
    assert not params1.compatible_with(params5)

    params7 = DuckDBWriterTargetParams(dimensions=128, schema={"doc_id": "VARCHAR", "embedding": "FLOAT[128]"})
    assert not params1.compatible_with(params7)

    params8 = DuckDBWriterTargetParams(dimensions=128, schema={"doc_id": "VARCHAR", "embedding": "FLOAT"})
    assert params8.compatible_with(params7)


def test_duckdb_writer_get_existing_target_params_table_not_exist(mock_duckdb, target_params):
    client = DuckDBClient(DuckDBWriterClientParams())
    mock_duckdb.sql.side_effect = Exception("Table does not exist")
    params = client.get_existing_target_params(target_params)
    assert params.schema == target_params.schema


def test_duckdb_writer_write_many_records_empty(mock_duckdb, target_params):
    client = DuckDBClient(DuckDBWriterClientParams())
    client.write_many_records([], target_params)
    assert mock_duckdb.sql.call_count == 0


def test_duckdb_writer_write_many_records_invalid_record(mock_duckdb, target_params, document):
    client = DuckDBClient(DuckDBWriterClientParams())
    invalid_record = mock.Mock(spec=BaseDBWriter.Record)
    with pytest.raises(AssertionError):
        client.write_many_records([invalid_record], target_params)


def test_duckdb_document_record_from_doc_no_doc_id(target_params):
    document = Document(
        doc_id=None,
        parent_id="parent1",
        properties={"key": "value"},
        type="type1",
        text_representation="text",
        bbox=(0.0, 0.0, 1.0, 1.0),
        shingles=[1, 2, 3],
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    with pytest.raises(ValueError):
        DuckDBDocumentRecord.from_doc(document, target_params)


def test_duckdb_writer_create_target_idempotent_no_schema(mock_duckdb, target_params):
    client = DuckDBClient(DuckDBWriterClientParams())
    target_params.schema = None
    client.create_target_idempotent(target_params)
    assert mock_duckdb.sql.call_count == 0
