import json
from dataclasses import dataclass
from typing import Any

import requests
from requests import Response

from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data import Document
from sycamore.data.element import create_element


@dataclass
class ArynClientParams(BaseDBReader.ClientParams):
    def __init__(self, aryn_url: str, api_key: str, **kwargs):
        self.aryn_url = aryn_url
        assert self.aryn_url is not None, "Aryn URL is required"
        self.api_key = api_key
        assert self.api_key is not None, "API key is required"
        self.kwargs = kwargs


@dataclass
class ArynQueryParams(BaseDBReader.QueryParams):
    def __init__(self, docset_id: str):
        self.docset_id = docset_id


class ArynQueryResponse(BaseDBReader.QueryResponse):
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs

    def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
        docs = []
        for doc in self.docs:
            elements = doc.get("elements", [])
            _doc = Document(**doc)
            _doc.data["elements"] = [create_element(**element) for element in elements]
            docs.append(_doc)

        return docs


class ArynClient(BaseDBReader.Client):
    def __init__(self, client_params: ArynClientParams, **kwargs):
        self.aryn_url = client_params.aryn_url
        self.api_key = client_params.api_key
        self.kwargs = kwargs

    def read_records(self, query_params: "BaseDBReader.QueryParams") -> "ArynQueryResponse":
        assert isinstance(query_params, ArynQueryParams)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response: Response = requests.post(
            f"{self.aryn_url}/docsets/{query_params.docset_id}/read", stream=True, headers=headers
        )
        assert response.status_code == 200
        docs = []
        print(f"Reading from docset: {query_params.docset_id}")
        for chunk in response.iter_lines():
            # print(f"\n{chunk}\n")
            doc = json.loads(chunk)
            docs.append(doc)

        return ArynQueryResponse(docs)

    def check_target_presence(self, query_params: "BaseDBReader.QueryParams") -> bool:
        return True

    @classmethod
    def from_client_params(cls, params: "BaseDBReader.ClientParams") -> "ArynClient":
        assert isinstance(params, ArynClientParams)
        return cls(params)


class ArynReader(BaseDBReader):
    Client = ArynClient
    Record = ArynQueryResponse
    ClientParams = ArynClientParams
    QueryParams = ArynQueryParams
