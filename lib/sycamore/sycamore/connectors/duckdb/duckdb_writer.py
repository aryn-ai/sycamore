from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Dict
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.connectors.writers.base import BaseDBWriter
from sycamore.connectors.writers.common import convert_to_str_dict
import pyarrow as pa
import duckdb
import os


@dataclass
class DuckDBClientParams(BaseDBWriter.ClientParams):
    pass


@dataclass
class DuckDBTargetParams(BaseDBWriter.TargetParams):
    db_url: Optional[str] = "tmp.db"
    table_name: Optional[str] = "default_table"
    batch_size: int = 1000
    schema: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            "doc_id": "VARCHAR",
            "embeddings": "DOUBLE[]",
            "properties": "MAP(VARCHAR, VARCHAR)",
            "text_representation": "VARCHAR",
            "bbox": "DOUBLE[]",
            "shingles": "BIGINT[]",
            "type": "VARCHAR",
        }
    )

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, DuckDBTargetParams):
            return False
        if self.db_url != other.db_url:
            return False
        if self.table_name != other.table_name:
            return False
        if other.schema:
            return self.schema == other.schema
        return True


class DuckDBClient(BaseDBWriter.Client):
    def __init__(self, client_params: DuckDBClientParams):
        pass

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "DuckDBClient":
        assert isinstance(params, DuckDBClientParams)
        return DuckDBClient(params)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        assert isinstance(target_params, DuckDBTargetParams), f"Wrong kind of target parameters found: {target_params}"
        dict_params = asdict(target_params)
        N = target_params.batch_size * 1024  # Around 1 MB
        headers = ["doc_id", "embeddings", "properties", "text_representation", "bbox", "shingles", "type"]
        schema = pa.schema(
            [
                ("doc_id", pa.string()),
                ("embeddings", pa.list_(pa.float32())),
                ("properties", pa.map_(pa.string(), pa.string())),
                ("text_representation", pa.string()),
                ("bbox", pa.list_(pa.float32())),
                ("shingles", pa.list_(pa.int64())),
                ("type", pa.string()),
            ]
        )

        def write_batch(batch_data: dict):
            pa_table = pa.Table.from_pydict(batch_data, schema=schema)  # noqa
            client = duckdb.connect(str(dict_params.get("db_url")))
            client.sql(f"INSERT INTO {dict_params.get('table_name')} SELECT * FROM pa_table")
            for key in batch_data:
                batch_data[key].clear()

        batch_data: dict[str, list[Any]] = {key: [] for key in headers}

        for r in records:
            # Append the new data to the batch
            batch_data["doc_id"].append(r.doc_id)
            batch_data["embeddings"].append(r.embeddings)
            batch_data["properties"].append(convert_to_str_dict(r.properties) if r.properties else {})
            batch_data["text_representation"].append(r.text_representation)
            batch_data["bbox"].append(r.bbox)
            batch_data["shingles"].append(r.shingles)
            batch_data["type"].append(r.type)

            # If we've reached the batch size, write to the database
            if batch_data.__sizeof__() >= N:
                write_batch(batch_data)
        # Write any remaining records
        if len(batch_data["doc_id"]) > 0:
            write_batch(batch_data)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBTargetParams)
        dict_params = asdict(target_params)
        schema = dict_params.get("schema")
        client = duckdb.connect(str(dict_params.get("db_url")))
        try:
            if schema:
                client.sql(
                    f"""CREATE TABLE {dict_params.get('table_name')} (doc_id {schema.get('doc_id')},
                      embeddings {schema.get('embeddings')}, properties {schema.get('properties')}, 
                      text_representation {schema.get('text_representation')}, bbox {schema.get('bbox')}, 
                      shingles {schema.get('shingles')}, type {schema.get('type')})"""
                )
            else:
                print(
                    f"""Error creating table {dict_params.get('table_name')} 
                    in database {dict_params.get('db_url')}: no schema provided"""
                )
        except Exception as e:
            print(f"Error creating table {dict_params.get('table_name')} in database {dict_params.get('db_url')}: {e}")

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "DuckDBTargetParams":
        assert isinstance(target_params, DuckDBTargetParams)
        dict_params = asdict(target_params)
        schema = None
        if target_params.db_url and target_params.table_name and os.path.exists(target_params.db_url):
            client = duckdb.connect(str(dict_params.get("db_url")))
            try:
                table = client.sql(f"SELECT * FROM {target_params.table_name}")
                schema = dict(zip(table.columns, [repr(i) for i in table.dtypes]))
            except Exception as e:
                print(
                    f"""Table {dict_params.get('table_name')} 
                    does not exist in database {dict_params.get('table_name')}: {e}"""
                )
        return DuckDBTargetParams(
            db_url=target_params.db_url,
            table_name=target_params.table_name,
            batch_size=target_params.batch_size,
            schema=schema,
        )


@dataclass
class DuckDBDocumentRecord(BaseDBWriter.Record):
    doc_id: str
    embeddings: Optional[list[float]] = None
    properties: Optional[dict[str, Any]] = None
    text_representation: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    shingles: Optional[list[int]] = None
    type: Optional[str] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "DuckDBDocumentRecord":
        assert isinstance(target_params, DuckDBTargetParams)
        doc_id = document.doc_id
        if doc_id is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        embedding = document.embedding
        return DuckDBDocumentRecord(
            doc_id=doc_id,
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
