from dataclasses import dataclass, asdict, field
from typing import Optional, Any
from typing_extensions import TypeGuard

import pyarrow as pa
import logging
import duckdb

from sycamore.data.document import Document
from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.utils.import_utils import requires_modules
from sycamore.connectors.common import convert_to_str_dict, _get_pyarrow_type


@dataclass
class DuckDBWriterClientParams(BaseDBWriter.ClientParams):
    db_url: str


@dataclass
class DuckDBWriterTargetParams(BaseDBWriter.TargetParams):
    dimensions: int
    table_name: Optional[str] = "default_table"
    batch_size: int = 1000
    schema: dict[str, str] = field(
        default_factory=lambda: {
            "doc_id": "VARCHAR",
            "parent_id": "VARCHAR",
            "embedding": "FLOAT",
            "properties": "MAP(VARCHAR, VARCHAR)",
            "text_representation": "VARCHAR",
            "bbox": "DOUBLE[]",
            "shingles": "BIGINT[]",
            "type": "VARCHAR",
        }
    )

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, DuckDBWriterTargetParams):
            return False
        if self.dimensions != other.dimensions:
            return False
        if self.table_name != other.table_name:
            return False
        if other.schema and self.schema:
            if (
                "embedding" in other.schema
                and "embedding" in self.schema
                and self.schema["embedding"] != other.schema["embedding"]
            ):
                self.schema["embedding"] = self.schema["embedding"] + "[" + str(self.dimensions) + "]"
            return self.schema == other.schema
        return True


class DuckDBClient(BaseDBWriter.Client):
    @requires_modules("duckdb", extra="duckdb")
    def __init__(self, client_params: DuckDBWriterClientParams):
        if not client_params.db_url:
            raise ValueError(f"Must provide valid database url. Location Specified: {client_params.db_url}")
        self._client = duckdb.connect(database=client_params.db_url)

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "DuckDBClient":
        assert isinstance(params, DuckDBWriterClientParams)
        return DuckDBClient(params)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        assert isinstance(
            target_params, DuckDBWriterTargetParams
        ), f"Wrong kind of target parameters found: {target_params}"
        dict_params = asdict(target_params)
        N = target_params.batch_size * 1024  # Around 1 MB

        # Validate schema and create pyarrow schema
        headers = []
        pa_fields = []
        for key, dtype in target_params.schema.items():
            headers.append(key)
            try:
                pa_dtype = _get_pyarrow_type(key, dtype)
                pa_fields.append((key, pa_dtype))
            except Exception as e:
                raise ValueError(f"Invalid schema attribute or datatype for {key}: {e}")

        schema = pa.schema(pa_fields)

        def write_batch(batch_data: dict):
            pa_table = pa.Table.from_pydict(batch_data, schema=schema)  # noqa
            self._client.sql(f"INSERT INTO {dict_params.get('table_name')} SELECT * FROM pa_table")
            for key in batch_data:
                batch_data[key].clear()

        batch_data: dict[str, list[Any]] = {key: [] for key in headers}

        for r in records:
            for key in headers:
                value = getattr(r, key, None)
                if isinstance(value, dict) and value:
                    value = convert_to_str_dict(value)
                batch_data[key].append(value)

            # If we've reached the batch size, write to the database
            if batch_data.__sizeof__() >= N:
                write_batch(batch_data)
        # Write any remaining records
        if len(batch_data[headers[0]]) > 0:
            write_batch(batch_data)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBWriterTargetParams)
        dict_params = asdict(target_params)
        schema = dict_params.get("schema")
        try:
            if schema:
                columns = []
                for key, dtype in schema.items():
                    if key == "embedding":
                        dtype += f"[{dict_params.get('dimensions')}]"
                    columns.append(f"{key} {dtype}")
                columns_str = ", ".join(columns)
                self._client.sql(f"CREATE TABLE {dict_params.get('table_name')} ({columns_str})")
            else:
                logging.warning(
                    f"""Error creating table {dict_params.get('table_name')}
                    in database {dict_params.get('db_url')}: no schema provided"""
                )
        except Exception as e:
            logging.debug(f"Table {dict_params.get('table_name')} could not be created: {e}")

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "DuckDBWriterTargetParams":
        assert isinstance(target_params, DuckDBWriterTargetParams)
        dict_params = asdict(target_params)
        schema = target_params.schema
        if target_params.table_name:
            try:
                table = self._client.sql(f"SELECT * FROM {target_params.table_name}")
                schema = dict(zip(table.columns, [str(i) for i in table.dtypes]))
            except Exception as e:
                logging.warning(
                    f"""Table {dict_params.get('table_name')}
                    does not exist in database {dict_params.get('table_name')}: {e}"""
                )
        return DuckDBWriterTargetParams(
            dimensions=target_params.dimensions,
            table_name=target_params.table_name,
            batch_size=target_params.batch_size,
            schema=schema,
        )

    def close(self):
        self._client.close()


@dataclass
class DuckDBDocumentRecord(BaseDBWriter.Record):
    doc_id: str
    parent_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    properties: Optional[dict[str, Any]] = None
    text_representation: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    shingles: Optional[list[int]] = None
    type: Optional[str] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "DuckDBDocumentRecord":
        assert isinstance(target_params, DuckDBWriterTargetParams)
        doc_id = document.doc_id
        if doc_id is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        return DuckDBDocumentRecord(
            doc_id=doc_id,
            parent_id=document.parent_id,
            properties=document.properties,
            type=document.type,
            text_representation=document.text_representation,
            bbox=document.bbox.coordinates if document.bbox else None,
            shingles=document.shingles,
            embedding=document.embedding,
        )


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[DuckDBDocumentRecord]]:
    return all(isinstance(r, DuckDBDocumentRecord) for r in records)


class DuckDBWriter(BaseDBWriter):
    Client = DuckDBClient
    Record = DuckDBDocumentRecord
    ClientParams = DuckDBWriterClientParams
    TargetParams = DuckDBWriterTargetParams
