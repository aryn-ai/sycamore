from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, Any
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.writers.base import BaseDBWriter
from sycamore.writers.common import convert_to_str_dict
import pyarrow as pa
import duckdb


@dataclass
class DuckDBClientParams(BaseDBWriter.ClientParams):
    pass


@dataclass
class DuckDBTargetParams(BaseDBWriter.TargetParams):
    db_url: Optional[str] = ":default:"
    table_name: Optional[str] = "default_table"

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, DuckDBTargetParams):
            return False
        if self.db_url != other.db_url:
            return False
        if self.table_name != other.table_name:
            return False
        return True


class DuckDBClient(BaseDBWriter.Client):
    def __init__(self, client_params: DuckDBClientParams):
        pass

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "DuckDBClient":
        assert isinstance(params, DuckDBClientParams)
        return DuckDBClient(params)

    def write_many_records(
        self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams, batch_size: int = 1000
    ):
        N = 1000 ^ 2  # Bit lesser than 1 MB
        dict_params = (
            asdict(target_params)
            if is_dataclass(target_params)
            else (target_params.__dict__ if hasattr(target_params, "__dict__") else {})
        )
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        assert isinstance(target_params, DuckDBTargetParams)

        headers = ["uuid", "embeddings", "properties", "text_representation", "bbox", "shingles", "type"]
        creation = True
        # schema = pa.schema(
        #     [
        #         ("uuid", pa.string()),
        #         ("embeddings", pa.list_(pa.float32())),
        #         ("properties", pa.map_(pa.string(), pa.string())),
        #         ("text_representation", pa.string()),
        #         ("bbox", pa.list_(pa.float32())),
        #         ("shingles", pa.list_(pa.int64())),
        #         ("type", pa.string()),
        #     ]
        # )

        def write_batch(batch_data: dict):
            pa_table = pa.Table.from_pydict(batch_data)  # noqa
            client = duckdb.connect(str(dict_params.get("db_url")))
            nonlocal creation
            if creation:
                client.sql(f"CREATE TABLE {dict_params.get('table_name')} AS SELECT * FROM pa_table")
                creation = False
            else:
                client.sql(f"INSERT INTO {dict_params.get('table_name')} SELECT * FROM pa_table")
            for key in batch_data:
                batch_data[key].clear()

        batch_data: dict[str, list[Any]] = {key: [] for key in headers}

        for r in records:
            # Append the new data to the batch
            batch_data["uuid"].append(r.uuid)
            batch_data["embeddings"].append(r.embeddings)
            batch_data["properties"].append(r.properties)
            batch_data["text_representation"].append(r.text_representation)
            batch_data["bbox"].append(r.bbox)
            batch_data["shingles"].append(r.shingles)
            batch_data["type"].append(r.type)

            # If we've reached the batch size, write to the database
            if batch_data.__sizeof__() >= N:
                batch_data["properties"] = (
                    [convert_to_str_dict(i) for i in batch_data["properties"]] if batch_data["properties"] else []
                )
                write_batch(batch_data)
        # Write any remaining records
        if len(batch_data["uuid"]) > 0:
            write_batch(batch_data)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBTargetParams)

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "DuckDBTargetParams":
        assert isinstance(target_params, DuckDBTargetParams)
        return DuckDBTargetParams(db_url=target_params.db_url, table_name=target_params.table_name)


@dataclass
class DuckDBDocumentRecord(BaseDBWriter.Record):
    uuid: str
    embeddings: Optional[list[float]] = None
    properties: Optional[dict[str, Any]] = None
    text_representation: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    shingles: Optional[list[int]] = None
    type: Optional[str] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "DuckDBDocumentRecord":
        assert isinstance(target_params, DuckDBTargetParams)
        uuid = document.doc_id
        if uuid is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        embedding = document.embedding
        return DuckDBDocumentRecord(
            uuid=uuid,
            properties=document.properties,
            type=document.type,
            text_representation=document.text_representation,
            bbox=document.bbox.coordinates if document.bbox else None,
            shingles=document.shingles,
            embeddings=embedding,
        )


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[DuckDBDocumentRecord]]:
    return all(isinstance(r, DuckDBDocumentRecord) for r in records)


class DuckDBWriter(BaseDBWriter):
    Client = DuckDBClient
    Record = DuckDBDocumentRecord
    ClientParams = DuckDBClientParams
    TargetParams = DuckDBTargetParams
