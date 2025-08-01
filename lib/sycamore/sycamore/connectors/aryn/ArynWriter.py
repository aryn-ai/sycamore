import tempfile
from dataclasses import dataclass
from typing import Optional, Mapping, Any

import requests

from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.data import Document
from sycamore.decorators import experimental


@dataclass
class ArynWriterClientParams(BaseDBWriter.ClientParams):
    def __init__(self, aryn_url: str, api_key: str, **kwargs):
        self.aryn_url = aryn_url
        assert self.aryn_url is not None, "Aryn URL is required"
        self.api_key = api_key
        assert self.api_key is not None, "API key is required"
        self.kwargs = kwargs


@dataclass
class ArynWriterTargetParams(BaseDBWriter.TargetParams):
    def __init__(
        self,
        docset_id: Optional[str] = None,
        update_schema: bool = False,
        update_keys: Optional[list[str]] = None,
    ):
        self.docset_id = docset_id
        self.update_schema = update_schema
        self.update_keys = update_keys

    def compatible_with(self, other: "BaseDBWriter.TargetParams") -> bool:
        return True


class ArynWriterRecord(BaseDBWriter.Record):
    def __init__(self, doc: Document):
        self.doc = doc

    @classmethod
    def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "ArynWriterRecord":
        return cls(document)


class ArynWriterClient(BaseDBWriter.Client):
    def __init__(self, client_params: ArynWriterClientParams, **kwargs):
        self.aryn_url = client_params.aryn_url
        self.api_key = client_params.api_key
        self.kwargs = kwargs

    @classmethod
    def from_client_params(cls, params: "BaseDBWriter.ClientParams") -> "BaseDBWriter.Client":
        assert isinstance(params, ArynWriterClientParams)
        return cls(params)

    def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, ArynWriterTargetParams)
        docset_id = target_params.docset_id

        headers = {"Authorization": f"Bearer {self.api_key}"}
        update_schema = target_params.update_schema
        update_keys = target_params.update_keys
        sess = requests.Session()
        for record in records:
            assert isinstance(record, ArynWriterRecord)
            doc = record.doc
            print(doc)
            with tempfile.TemporaryFile(prefix="aryn-writer-", suffix=".ArynSDoc") as stream:
                params: dict[str, Any] = {"docset_id": docset_id, "update_schema": update_schema}
                if update_keys:
                    params["update_keys"] = update_keys
                    # Reduce payload size by removing elements if not updating schema only.
                    del doc.elements
                doc.web_serialize(stream)
                stream.seek(0)
                files: Mapping = {"doc": stream}
                sess.post(
                    url=f"{self.aryn_url}/docsets/write",
                    params=params,
                    files=files,
                    headers=headers,
                )
            # For each batch we'll update the Aryn schema with only the first doc of each batch
            update_schema = False
            update_keys = None

    def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        pass

    def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams"):
        pass


@experimental
class ArynWriter(BaseDBWriter):
    Client = ArynWriterClient
    Record = ArynWriterRecord
    ClientParams = ArynWriterClientParams
    TargetParams = ArynWriterTargetParams
