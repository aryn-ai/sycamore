from sycamore.data import Document

from dataclasses import dataclass
from typing import Optional, Any

from sycamore.data.document import Document
from sycamore.connectors.base_reader import BaseDBReader
import duckdb


@dataclass
class DuckDBReaderClientParams(BaseDBReader.ClientParams):
    pass


@dataclass
class DuckDBReaderQueryParams(BaseDBReader.QueryParams):
    db_url: str
    table_name: str
    on_input_docs: bool
    query: Optional[str]


class DuckDBReaderClient(BaseDBReader.Client):
    def __init__(self, client_params: DuckDBReaderClientParams):
        pass

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "DuckDBReaderClient":
        assert isinstance(params, DuckDBReaderClientParams)
        return DuckDBReaderClient(params)

    def read_records(self, input_docs: list[Document], query_params: BaseDBReader.QueryParams):
        assert isinstance(
            query_params, DuckDBReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        con = duckdb.connect(database=query_params.db_url, read_only=True)
        results = []
        if query_params.on_input_docs and query_params.query:
            for doc in input_docs:  # noqa
                results.append(DuckDBReaderDocumentRecord(output=con.execute(f"{query_params.query}")))
        else:
            if query_params.query:
                results = [DuckDBReaderDocumentRecord(con.execute(query_params.query))]
            else:
                results = [DuckDBReaderDocumentRecord(con.execute(f"SELECT * from {query_params.table_name}"))]
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, DuckDBReaderQueryParams)
        client = duckdb.connect(query_params.db_url, read_only=True)
        try:
            client.sql(f"SELECT * FROM {query_params.table_name}")
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
