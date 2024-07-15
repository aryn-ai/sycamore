from dataclasses import dataclass

from abc import ABC, abstractmethod
from typing import Optional
from ray.data import Dataset, from_items

from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Read
from sycamore.utils.ray_utils import check_serializable
from sycamore.utils.time_trace import TimeTrace


class BaseDBReader(Read):

    # Type param for the client
    class Client(ABC):
        @classmethod
        @abstractmethod
        def from_client_params(cls, params: "BaseDBReader.ClientParams") -> "BaseDBReader.Client":
            pass

        @abstractmethod
        def read_records(
            self, input_docs: list[Document], query_params: "BaseDBReader.QueryParams"
        ) -> list["BaseDBReader.Record"]:
            pass

        @abstractmethod
        def check_target_presence(self, query_params: "BaseDBReader.QueryParams") -> bool:
            pass

    # Type param for the objects that are read from the db
    class Record(ABC):
        @classmethod
        @abstractmethod
        def to_doc(cls, record: "BaseDBReader.Record", query_params: "BaseDBReader.QueryParams") -> list[Document]:
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
        plan: Optional[Node],
        client_params: ClientParams,
        query_params: QueryParams,
        **kwargs,
    ):
        super().__init__(plan, **kwargs)
        check_serializable(client_params, query_params, filter)
        self._client_params = client_params
        self._query_params = query_params

    def read_docs(self, input_docs: list[Document]) -> Dataset:
        client = self.Client.from_client_params(self._client_params)

        if not client.check_target_presence(self._query_params):
            raise ValueError("Target is not present\n" f"Parameters: {self._query_params}\n")
        records = client.read_records(input_docs=input_docs, query_params=self._query_params)
        docs = [self.Record.to_doc(r, self._query_params) for r in records]
        flat_docs = [item for sublist in docs for item in sublist]
        input_docs.extend(flat_docs)
        return from_items(items=[{"doc": doc.serialize()} for doc in input_docs])

    def execute(self, input_docs: list[Document] = [], **kwargs) -> Dataset:
        with TimeTrace("Reader"):
            return self.read_docs(input_docs=input_docs)
