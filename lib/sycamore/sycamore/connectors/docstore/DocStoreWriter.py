import os
from dataclasses import dataclass
from typing import Optional

import requests

from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.data import Document


@dataclass
class DocStoreWriterClientParams(BaseDBWriter.ClientParams):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key if api_key is not None else os.getenv("ARYN_API_KEY")
        assert self.api_key is not None, "API key is required"
        self.kwargs = kwargs


@dataclass
class DocStoreWriterTargetParams(BaseDBWriter.TargetParams):
    def __init__(self, docset_id: Optional[str] = None):
        self.docset_id = docset_id

    def compatible_with(self, other: "BaseDBWriter.TargetParams") -> bool:
        return True


class DocStoreWriterRecord(BaseDBWriter.Record):
    def __init__(self, doc: Document):
        self.doc = doc

    @classmethod
    def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "DocStoreWriterRecord":
        return cls(document)


class DocStoreWriterClient(BaseDBWriter.Client):
    def __init__(self, client_params: DocStoreWriterClientParams, **kwargs):
        self.api_key = client_params.api_key
        self.kwargs = kwargs

    @classmethod
    def from_client_params(cls, params: "BaseDBWriter.ClientParams") -> "BaseDBWriter.Client":
        assert isinstance(params, DocStoreWriterClientParams)
        return cls(params)

    def create_docset(self, name: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        res = requests.post(url="http://localhost:8001/v1/docstore/docsets", data={"name": name}, headers=headers)
        return res.json()["docset_id"]

    def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, DocStoreWriterTargetParams)
        docset_id = target_params.docset_id
        name = ""
        if docset_id is None:
            docset_id = self.create_docset(name)

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        for record in records:
            assert isinstance(record, DocStoreWriterRecord)
            doc = record.doc
            res = requests.post(url=f"http://localhost:8001/v1/docstore/docsets/write",
                                params={"docset_id": docset_id},
                                data=doc.serialize(), headers=headers)


    def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        pass

    def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> "BaseDBWriter.TargetParams":
        pass


class DocStoreWriter(BaseDBWriter):
    Client = DocStoreWriterClient
    Record = DocStoreWriterRecord
    ClientParams = DocStoreWriterClientParams
    TargetParams = DocStoreWriterTargetParams
