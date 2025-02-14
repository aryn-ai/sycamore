from dataclasses import dataclass
from typing import Optional, Mapping

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
    def __init__(self, docset_id: Optional[str] = None):
        self.docset_id = docset_id

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

        for record in records:
            assert isinstance(record, ArynWriterRecord)
            doc = record.doc
            files: Mapping = {"doc": doc.serialize()}
            requests.post(
                url=f"{self.aryn_url}/docsets/write", params={"docset_id": docset_id}, files=files, headers=headers
            )

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
