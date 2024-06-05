from dataclasses import dataclass
from typing import Any, Iterable, Optional

from abc import ABC, abstractmethod
import numpy as np

from ray.data import Dataset, Datasink
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from sycamore.data.document import Document, MetadataDocument
from sycamore.plan_nodes import Node, Write


class BaseDBWriter(Write):

    # Type param for the client
    class client_t(ABC):
        @abstractmethod
        @staticmethod
        def from_client_params(params: "BaseDBWriter.client_params_t") -> "BaseDBWriter.client_t":
            pass

        @abstractmethod
        def write_many_records(self, records: list["BaseDBWriter.record_t"]):
            pass

        @abstractmethod
        def create_index_if_missing(self, index_params: "BaseDBWriter.index_params_t"):
            pass

    # Type param for the objects to write to the db
    class record_t(ABC):

        @abstractmethod
        @staticmethod
        def from_doc(document: Document) -> "BaseDBWriter.record_t":
            pass

        @abstractmethod
        def serialize(self) -> bytes:
            pass

        @abstractmethod
        @staticmethod
        def deserialize(byteses: bytes) -> "BaseDBWriter.record_t":
            pass

    # Type param for the object used to configure a new index if necessary
    @dataclass
    class index_params_t(ABC):
        pass

    # Type param for the object used to create a client
    @dataclass
    class client_params_t(ABC):
        pass

    def __init__(
        self, plan: Node, client_params: client_params_t, index_params: Optional[index_params_t], **ray_remote_args
    ):
        super().__init__(plan, **ray_remote_args)
        self._client_params = client_params
        self._index_params = index_params

    def get_client(self, client_params: client_params_t) -> client_t:
        return self.client_t.from_client_params(client_params)

    def doc_to_record(self, doc: Document) -> record_t:
        return self.record_t.from_doc(doc)

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        record_dataset = self._ray_map_docs_to_records(input_dataset)
        client = self.get_client(self._client_params)
        datasink = BaseDBWriter.InnerDatasink(client, self._index_params, self)
        record_dataset.write_datasink(datasink, ray_remote_args=self.resource_args)
        return input_dataset

    def _ray_map_docs_to_records(self, dataset: Dataset) -> Dataset:

        def ray_callable(ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            all_docs = [Document.deserialize(s) for s in ray_input.get("docs", [])]
            docs = filter(lambda d: not isinstance(d, MetadataDocument), all_docs)
            records = [self.doc_to_record(d) for d in docs]
            return {"records": [r.serialize() for r in records]}

        return dataset.map_batches(ray_callable, **self.resource_args)

    class InnerDatasink(Datasink):

        def __init__(
            self,
            client: "BaseDBWriter.client_t",
            index_params: Optional["BaseDBWriter.index_params_t"],
            owner: "BaseDBWriter",
        ):
            self._client = client
            self._index_params = index_params
            self._owner = owner

        def on_write_start(self) -> None:
            if self._index_params:
                self._client.create_index_if_missing(self._index_params)

        def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
            builder = DelegatingBlockBuilder()
            for block in blocks:
                builder.add_block(block)
            master_block = builder.build()
            rows = BlockAccessor.for_block(master_block).to_arrow().to_pylist()
            records = [self._owner.record_t.deserialize(row["record"]) for row in rows]
            self._client.write_many_records(records)
