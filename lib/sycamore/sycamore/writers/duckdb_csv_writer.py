from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, Any
from typing_extensions import TypeGuard

from sycamore.data.document import Document
from sycamore.writers.base import BaseDBWriter
import random
import string
import os
import csv


@dataclass
class DuckDBClientParams(BaseDBWriter.ClientParams):
    pass


@dataclass
class DuckDBTargetParams(BaseDBWriter.TargetParams):
    parquet_location: str

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, DuckDBTargetParams):
            return False
        if self.parquet_location != other.parquet_location:
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
        N = 10
        dict_params = (
            asdict(target_params)
            if is_dataclass(target_params)
            else (target_params.__dict__ if hasattr(target_params, "__dict__") else {})
        )
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        assert isinstance(target_params, DuckDBTargetParams)

        # Determine file location
        parquet_location = dict_params.get("parquet_location", "./tmp/duckdb")
        file_location = os.path.join(parquet_location, "".join(random.choices(string.ascii_uppercase, k=N)) + ".csv")
        while os.path.isfile(file_location):
            file_location = os.path.join(
                parquet_location, "".join(random.choices(string.ascii_uppercase, k=N)) + ".csv"
            )
        # Prepare the CSV file and directory
        directory = os.path.dirname(file_location)
        os.makedirs(directory, exist_ok=True)

        headers = ["uuid", "embeddings", "properties", "text_representation", "bbox", "shingles", "type"]

        def write_batch(batch_data: dict):
            # If the file doesn't exist, write headers first
            with open(file_location, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                for i in range(len(batch_data["uuid"])):
                    row = {key: batch_data[key][i] for key in batch_data}
                    writer.writerow(row)
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

            # If we've reached the batch size, write to the CSV file
            if len(batch_data["uuid"]) >= batch_size:
                write_batch(batch_data)
        # Write any remaining records
        if len(batch_data["uuid"]) > 0:
            write_batch(batch_data)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, DuckDBTargetParams)

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "DuckDBTargetParams":
        assert isinstance(target_params, DuckDBTargetParams)
        return DuckDBTargetParams(parquet_location=target_params.parquet_location)


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


class DuckDBCSVWriter(BaseDBWriter):
    Client = DuckDBClient
    Record = DuckDBDocumentRecord
    ClientParams = DuckDBClientParams
    TargetParams = DuckDBTargetParams
