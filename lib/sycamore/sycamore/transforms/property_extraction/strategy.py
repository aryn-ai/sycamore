from abc import ABC, abstractmethod
from typing import Iterable

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import Schema


class StepThroughStrategy(ABC):
    @abstractmethod
    def step_through(self, document: Document) -> Iterable[list[Element]]:
        pass


class OneElementAtATime(StepThroughStrategy):
    def step_through(self, document: Document) -> Iterable[list[Element]]:
        for elt in document.elements:
            yield [elt]


class BatchElements(StepThroughStrategy):
    def __init__(self, batch_size: int = 10):
        self._batch_size = batch_size

    def step_through(self, document: Document) -> Iterable[list[Element]]:
        for i in range(0, len(document.elements), self._batch_size):
            batch = document.elements[i : i + self._batch_size]
            if batch:
                yield batch


class SchemaPartitionStrategy(ABC):
    @abstractmethod
    def partition_schema(self, schema: Schema) -> list[Schema]:
        pass


class NoSchemaSplitting(SchemaPartitionStrategy):
    def partition_schema(self, schema: Schema) -> list[Schema]:
        return [schema]
