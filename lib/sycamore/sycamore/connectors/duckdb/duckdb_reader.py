from sycamore.data import Document

from dataclasses import dataclass
from typing import Optional

from sycamore.connectors.base_reader import BaseDBReader
import duckdb


@dataclass
class DuckDBReaderClientParams(BaseDBReader.ClientParams):
    db_url: str


@dataclass
class DuckDBReaderQueryParams(BaseDBReader.QueryParams):
    table_name: str
    query: Optional[str]


class DuckDBReaderClient(BaseDBReader.Client):
    def __init__(self, client_params: DuckDBReaderClientParams):
        self._client = duckdb.connect(database=client_params.db_url, read_only=True)

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "DuckDBReaderClient":
        assert isinstance(params, DuckDBReaderClientParams)
        return DuckDBReaderClient(params)

    def read_records(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(
            query_params, DuckDBReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        if query_params.query:
            results = DuckDBReaderDocumentRecord(self._client.execute(query_params.query))
        else:
            results = DuckDBReaderDocumentRecord(self._client.execute(f"SELECT * from {query_params.table_name}"))
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, DuckDBReaderQueryParams)
        try:
            self._client.sql(f"SELECT * FROM {query_params.table_name}")
            return True
        except Exception:
            return False


@dataclass
class DuckDBReaderDocumentRecord(BaseDBReader.Record):
    output: duckdb.DuckDBPyConnection

    @classmethod
    def to_doc(cls, record: "BaseDBReader.Record", query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(record, DuckDBReaderDocumentRecord)
        data = record.output.fetchdf().to_dict(orient="records")
        result = []
        for object in data:
            result.append(Document(object))
        return result


class DuckDBReader(BaseDBReader):
    Client = DuckDBReaderClient
    Record = DuckDBReaderDocumentRecord
    ClientParams = DuckDBReaderClientParams
    QueryParams = DuckDBReaderQueryParams
