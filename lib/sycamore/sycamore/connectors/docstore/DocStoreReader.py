import json
import os
from dataclasses import dataclass
from typing import Optional, Any

import requests
from requests import Response

from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data import Document, Element


@dataclass
class DocStoreClientParams(BaseDBReader.ClientParams):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key if api_key is not None else os.getenv("ARYN_API_KEY")
        assert self.api_key is not None, "API key is required"
        self.kwargs = kwargs


@dataclass
class DocStoreQueryParams(BaseDBReader.QueryParams):
    def __init__(self, docset_id: str):
        self.docset_id = docset_id


class DocStoreQueryResponse(BaseDBReader.QueryResponse):
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        docs = []
        for doc in self.docs:
            elements_dicts = doc.get("elements", [])
            elements = [Element(**element_dict) for element_dict in elements_dicts]
            _doc = Document(**doc)
            _doc.elements = elements
            docs.append(_doc)

        return docs

class DocStoreClient(BaseDBReader.Client):
    def __init__(self, client_params: DocStoreClientParams, **kwargs):
        self.api_key = client_params.api_key
        self.kwargs = kwargs

    def read_records(self, query_params: "BaseDBReader.QueryParams") -> "DocStoreQueryResponse":
        assert isinstance(query_params, DocStoreQueryParams)
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response: Response = requests.post(
            f"http://0.0.0.0:8001/v1/docstore/docsets/{query_params.docset_id}/read", stream=True,
            headers=headers)
        assert response.status_code == 200
        i = 1
        docs = []
        for chunk in response.iter_lines():
            # print(f"\n{chunk}\n")
            doc = json.loads(chunk)
            docs.append(doc)

        return DocStoreQueryResponse(docs)

    def check_target_presence(self, query_params: "BaseDBReader.QueryParams") -> bool:
        return True

    @classmethod
    def from_client_params(cls, params: "BaseDBReader.ClientParams") -> "DocStoreClient":
        assert isinstance(params, DocStoreClientParams)
        return cls(params)


class DocStoreReader(BaseDBReader):
    Client = DocStoreClient
    Record = DocStoreQueryResponse
    ClientParams = DocStoreClientParams
    QueryParams = DocStoreQueryParams
