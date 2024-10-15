from sycamore.data import Document

from dataclasses import dataclass
import typing
from typing import Optional
from sycamore.connectors.common import convert_from_str_dict

from sycamore.connectors.base_reader import BaseDBReader
from sycamore.utils.import_utils import requires_modules

if typing.TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


@dataclass
class DuckDBReaderClientParams(BaseDBReader.ClientParams):
    db_url: str


@dataclass
class DuckDBReaderQueryParams(BaseDBReader.QueryParams):
    table_name: str
    query: Optional[str]
    create_hnsw_table: Optional[str]


class DuckDBReaderClient(BaseDBReader.Client):
    def __init__(self, client: "DuckDBPyConnection"):
        self._client = client

    @classmethod
    @requires_modules("duckdb", extra="duckdb")
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "DuckDBReaderClient":
        import duckdb

        assert isinstance(params, DuckDBReaderClientParams)
        client = duckdb.connect(database=params.db_url, read_only=True)
        return DuckDBReaderClient(client)

    def read_records(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(
            query_params, DuckDBReaderQueryParams
        ), f"Wrong kind of query parameters found: {query_params}"
        if query_params.create_hnsw_table:
            self._client.execute(query_params.create_hnsw_table)
        if query_params.query:
            results = DuckDBReaderQueryResponse(self._client.execute(query_params.query))
        else:
            results = DuckDBReaderQueryResponse(self._client.execute(f"SELECT * from {query_params.table_name}"))
        return results

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, DuckDBReaderQueryParams)
        try:
            self._client.sql(f"SELECT * FROM {query_params.table_name}")
            return True
        except Exception:
            return False


@dataclass
class DuckDBReaderQueryResponse(BaseDBReader.QueryResponse):
    output: "DuckDBPyConnection"

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        assert isinstance(self, DuckDBReaderQueryResponse)
        data = self.output.df()
        data = data.to_dict(orient="records")
        result = []
        for object in data:
            val = object.get("properties")
            if val is not None:
                object["properties"] = convert_from_str_dict(val)
            result.append(Document(object))
        return result


class DuckDBReader(BaseDBReader):
    Client = DuckDBReaderClient
    Record = DuckDBReaderQueryResponse
    ClientParams = DuckDBReaderClientParams
    QueryParams = DuckDBReaderQueryParams
