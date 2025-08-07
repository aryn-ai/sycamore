import io
import json
import logging
import struct
from dataclasses import dataclass
import random
from time import time
from typing import Any, TYPE_CHECKING, Optional

import httpx

from sycamore.connectors.aryn.client import ArynClient

from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data import Document
from sycamore.data.element import create_element
from sycamore.decorators import experimental

if TYPE_CHECKING:
    from ray.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DocFilter:
    doc_ids: Optional[list[str]] = None
    sample_ratio: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.doc_ids is not None and self.sample_ratio is not None:
            raise ValueError("Cannot specify both doc_ids and sample_ratio")
        if self.sample_ratio is not None and (self.sample_ratio < 0 or self.sample_ratio > 1):
            raise ValueError("sample_ratio must be between 0 and 1")

    def select(self, doc_list: list[str]) -> list[str]:
        if self.doc_ids is not None:
            return [doc_id for doc_id in doc_list if doc_id in self.doc_ids]
        elif self.sample_ratio is not None:
            if self.seed is not None:
                random.seed(self.seed)
            sample_size = max(1, int(len(doc_list) * self.sample_ratio))
            return random.sample(doc_list, sample_size)
        else:
            return doc_list


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
    def __init__(self, docset_id: str, doc_filter: Optional[DocFilter] = None):
        self.docset_id = docset_id
        self.doc_filter = doc_filter


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


class ArynReaderClient(BaseDBReader.Client):
    def __init__(self, client: ArynClient, client_params: ArynClientParams, **kwargs):
        self.aryn_url = client_params.aryn_url
        self.api_key = client_params.api_key
        self._client = client
        self.kwargs = kwargs

    def read_records(self, query_params: "BaseDBReader.QueryParams") -> "ArynQueryResponse":
        assert isinstance(query_params, ArynQueryParams)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        client = httpx.Client()
        with client.stream(
            "POST", f"{self.aryn_url}/docsets/{query_params.docset_id}/read", headers=headers
        ) as response:

            docs = []
            print(f"Reading from docset: {query_params.docset_id}")
            buffer = io.BytesIO()
            to_read = 0
            start_new_doc = True
            doc_size_buf = bytearray(4)
            idx = 0
            chunk_count = 0
            t0 = time()
            for chunk in response.iter_bytes():
                cur_pos = 0
                chunk_count += 1
                remaining = len(chunk)
                print(f"Chunk {chunk_count} size: {len(chunk)}")
                assert len(chunk) >= 4, f"Chunk too small: {len(chunk)} < 4"
                while cur_pos < len(chunk):
                    if start_new_doc:
                        doc_size_buf[idx:] = chunk[cur_pos : cur_pos + 4 - idx]
                        to_read = struct.unpack("!i", doc_size_buf)[0]
                        print(f"Reading doc of size: {to_read}")
                        doc_size_buf = bytearray(4)
                        idx = 0
                        cur_pos += 4
                        remaining = len(chunk) - cur_pos
                        start_new_doc = False
                    if to_read > remaining:
                        buffer.write(chunk[cur_pos:])
                        to_read -= remaining
                        print(f"Remaining to read: {to_read}")
                        # Read the next chunk
                        break
                    else:
                        print("Reading the rest of the doc from the chunk")
                        buffer.write(chunk[cur_pos : cur_pos + to_read])
                        docs.append(json.loads(buffer.getvalue().decode()))
                        buffer.flush()
                        buffer.seek(0)
                        cur_pos += to_read
                        to_read = 0
                        start_new_doc = True
                        if (cur_pos - len(chunk)) < 4:
                            idx = left_over = cur_pos - len(chunk)
                            doc_size_buf[:left_over] = chunk[cur_pos:]
                            # Need to get the rest of the next chunk
                            break

            t1 = time()
            print(f"Reading took: {t1 - t0} seconds")
        return ArynQueryResponse(docs)

    def check_target_presence(self, query_params: "BaseDBReader.QueryParams") -> bool:
        return True

    @classmethod
    def from_client_params(cls, params: "BaseDBReader.ClientParams") -> "ArynReaderClient":
        assert isinstance(params, ArynClientParams)
        client = ArynClient(params.aryn_url, params.api_key)
        return cls(client, params)


@experimental
class ArynReader(BaseDBReader):
    Client = ArynReaderClient
    Record = ArynQueryResponse
    ClientParams = ArynClientParams
    QueryParams = ArynQueryParams

    def __init__(
        self,
        client_params: ArynClientParams,
        query_params: ArynQueryParams,
        **kwargs,
    ):
        super().__init__(client_params=client_params, query_params=query_params, **kwargs)

    def _to_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        assert isinstance(self._client_params, ArynClientParams)
        assert isinstance(self._query_params, ArynQueryParams)

        client = self.Client.from_client_params(self._client_params)
        aryn_client = client._client

        doc_id = doc["doc_id"]
        doc = aryn_client.get_doc(self._query_params.docset_id, doc["doc_id"])
        elements = doc.get("elements", [])
        document = Document(**doc)
        document.doc_id = doc_id
        document.data["elements"] = []
        for json_element in elements:
            element = create_element(**json_element)
            element.data["doc_id"] = json_element["id"]
            document.data["elements"].append(element)
        return {"doc": Document.serialize(document)}

    def execute(self, **kwargs) -> "Dataset":

        assert isinstance(self._client_params, ArynClientParams)
        assert isinstance(self._query_params, ArynQueryParams)

        client = self.Client.from_client_params(self._client_params)
        aryn_client = client._client

        docs = aryn_client.list_docs(self._query_params.docset_id)
        if self._query_params.doc_filter is not None:
            docs = self._query_params.doc_filter.select(docs)
        logger.debug(f"Found {len(docs)} docs in docset: {self._query_params.docset_id}")

        from ray.data import from_items

        ds = from_items([{"doc_id": doc_id} for doc_id in docs])
        return ds.map(self._to_doc)
