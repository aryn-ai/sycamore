from dataclasses import dataclass
from typing import Optional, Union, Any
from sycamore.writers.common import drop_types
from typing_extensions import TypeGuard, TypeAlias

from sycamore.data.document import Document
from sycamore.writers.base import BaseDBWriter
import duckdb
from duckdb import CatalogException

@dataclass
class DuckDBClientParams(BaseDBWriter.ClientParams):
    db_name: Optional[str] = None

@dataclass
class DuckDBTargetParams(BaseDBWriter.TargetParams):
    table_name: str

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, DuckDBTargetParams):
            return False
        if self.table_name != other.table_name:
            return False
        return True


class DuckDBClient(BaseDBWriter.Client):
    def __init__(self, db_name: str = None):
        db_name = db_name if db_name else ":memory:"
        self._client = duckdb.connect(db_name)

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "DuckDBClient":
        assert isinstance(params, DuckDBClientParams)
        return DuckDBClient(params)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBTargetParams)
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        with self._client:
            for r in records:
                self._client.execute(f"INSERT INTO {target_params.table_name} VALUES ({r.uuid}, {r.properties}, {r.embeddings})")

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBTargetParams)
        with self._client:
                self._client.execute("CREATE TABLE IF NOT EXISTS {target_params.table_name} (uuid TEXT PRIMARY KEY, properties STRUCT, embeddings ARRAY(FLOAT))")

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "DuckDBTargetParams":
        assert isinstance(target_params, DuckDBTargetParams)
        with self._client:
            return DuckDBTargetParams(name=target_params.table_name, embeddings=target_params.embeddings)


@dataclass
class DuckDBDocumentRecord(BaseDBWriter.Record):
    uuid: str
    properties: Optional[dict[str, Any]] = None
    embeddings: list[float] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "DuckDBDocumentRecord":
        assert isinstance(target_params, DuckDBTargetParams)
        uuid = document.doc_id
        if uuid is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        embedding = document.embedding
        if embedding is None:
            raise ValueError(f"Cannot write documents without a embedding. Found {document}")
        properties = {
            "properties": document.properties,
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.coordinates if document.bbox else None,
            "shingles": document.shingles,
        }
        droperties = drop_types(properties, drop_empty_lists=True)
        assert isinstance(droperties, dict)
        return DuckDBDocumentRecord(uuid=uuid, properties=droperties, embeddings=embedding)


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[DuckDBDocumentRecord]]:
    return all(isinstance(r, DuckDBDocumentRecord) for r in records)

class DuckDBDocumentWriter(BaseDBWriter):
    Client = DuckDBClient
    Record = DuckDBDocumentRecord
    ClientParams = DuckDBClientParams
    TargetParams = DuckDBTargetParams

