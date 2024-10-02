from dataclasses import dataclass
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

from sycamore.data.document import Document
from sycamore.plan_nodes import Scan
from sycamore.utils.time_trace import TimeTrace

if TYPE_CHECKING:
    from ray.data import Dataset


class BaseDBReader(Scan):

    # Type param for the client
    class Client(ABC):
        @classmethod
        @abstractmethod
        def from_client_params(cls, params: "BaseDBReader.ClientParams") -> "BaseDBReader.Client":
            pass

        @abstractmethod
        def read_records(self, query_params: "BaseDBReader.QueryParams") -> "BaseDBReader.QueryResponse":
            pass

        @abstractmethod
        def check_target_presence(self, query_params: "BaseDBReader.QueryParams") -> bool:
            pass

    # Type param for the objects that are read from the db
    class QueryResponse(ABC):
        @abstractmethod
        def to_docs(self, query_params: "BaseDBReader.QueryParams") -> list[Document]:
            pass

    # Type param for the object used to estabilish the read target
    # e.g. opensearch/pinecone index, s3 bucket, weaviate collection
    # will also include the Query, and filters for the read
    @dataclass
    class QueryParams(ABC):
        pass

    # Type param for the object used to create a client
    @dataclass
    class ClientParams(ABC):
        pass

    def __init__(
        self,
        client_params: ClientParams,
        query_params: QueryParams,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client_params = client_params
        self._query_params = query_params

    def read_docs(self) -> list[Document]:
        client = self.Client.from_client_params(self._client_params)

        if not client.check_target_presence(self._query_params):
            raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")
        records = client.read_records(query_params=self._query_params)
        docs = records.to_docs(query_params=self._query_params)
        return docs

    def execute(self, **kwargs) -> "Dataset":
        from sycamore.utils.ray_utils import check_serializable

        check_serializable(self._client_params, self._query_params)
        from ray.data import from_items

        with TimeTrace("Reader"):
            return from_items(items=[{"doc": doc.serialize()} for doc in self.read_docs()])

    def local_source(self) -> list[Document]:
        return self.read_docs()

    def format(self):
        return "reader"
