from dataclasses import dataclass

from abc import ABC, abstractmethod

from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write
from sycamore.transforms.map import MapBatch
from sycamore.utils.ray_utils import check_serializable


class BaseDBWriter(MapBatch, Write):

    # Type param for the client
    class Client(ABC):
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

    # Type param for the objects to write to the db
    class Record(ABC):
        @classmethod
        @abstractmethod
        def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "BaseDBWriter.Record":
            pass

    # Type param for the object used to configure a new index if necessary
    @dataclass
    class TargetParams(ABC):
        pass

    # Type param for the object used to create a client
    @dataclass
    class ClientParams(ABC):
        pass

    def __init__(self, plan: Node, client_params: ClientParams, target_params: TargetParams, **ray_remote_args):
        super().__init__(plan, f=self.write_docs, **ray_remote_args)
        check_serializable(client_params, target_params)
        self._client_params = client_params
        self._target_params = target_params

    def write_docs(self, docs: list[Document]) -> list[Document]:
        client = self.Client.from_client_params(self._client_params)
        client.create_target_idempotent(self._target_params)
        created_target_params = client.get_existing_target_params(self._target_params)
        if created_target_params != self._target_params:
            raise ValueError(
                "Found mismatching target parameters in script and destination\n"
                f"Script: {self._target_params}\n"
                f"Destination: {created_target_params}\n"
            )
        records = [self.Record.from_doc(d, created_target_params) for d in docs]
        client.write_many_records(records, self._target_params)
        return docs
