from dataclasses import dataclass

from abc import ABC, abstractmethod
from typing import Callable

from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write
from sycamore.transforms.map import MapBatch
from sycamore.utils.time_trace import TimeTrace


class BaseDBWriter(MapBatch, Write):

    # Type param for the client
    class Client:
        @classmethod
        @abstractmethod
        def from_client_params(cls, params: "BaseDBWriter.ClientParams") -> "BaseDBWriter.Client":
            pass

        @abstractmethod
        def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
            pass

        @abstractmethod
        def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
            pass

        @abstractmethod
        def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> "BaseDBWriter.TargetParams":
            pass

        def close(self):
            pass

    # Type param for the objects to write to the db
    class Record(ABC):
        @classmethod
        @abstractmethod
        def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "BaseDBWriter.Record":
            pass

    # Type param for the object used to configure the write target
    # e.g. opensearch/pinecone index, s3 bucket, weaviate collection...
    @dataclass
    class TargetParams(ABC):
        def compatible_with(self, other: "BaseDBWriter.TargetParams") -> bool:
            return self == other

    # Type param for the object used to create a client
    @dataclass
    class ClientParams(ABC):
        pass

    def __init__(
        self,
        plan: Node,
        client_params: ClientParams,
        target_params: TargetParams,
        filter: Callable[[Document], bool] = lambda d: True,
        **kwargs,
    ):
        super().__init__(plan, f=self._write_docs_tt, **kwargs)

        self._filter = filter
        self._client_params = client_params
        self._target_params = target_params

    def write_docs(self, docs: list[Document]) -> list[Document]:
        try:
            client = self.Client.from_client_params(self._client_params)
            client.create_target_idempotent(self._target_params)
            created_target_params = client.get_existing_target_params(self._target_params)
            if not self._target_params.compatible_with(created_target_params):
                raise ValueError(
                    "Found mismatching target parameters in script and destination\n"
                    f"Script: {self._target_params}\n"
                    f"Destination: {created_target_params}\n"
                )
            records = [self.Record.from_doc(d, created_target_params) for d in docs if self._filter(d)]
            client.write_many_records(records, self._target_params)
        except Exception as e:
            raise ValueError(f"Error writing to target: {e}")
        finally:
            client.close()
        return docs

    def _write_docs_tt(self, docs: list[Document]) -> list[Document]:
        if self._name:
            with TimeTrace(self._name):
                return self.write_docs(docs)
        else:
            with TimeTrace("UnknownWriter"):
                return self.write_docs(docs)
