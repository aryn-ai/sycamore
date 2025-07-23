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


class SchemaPartitionStrategy(ABC):
    @abstractmethod
    def partition_schema(self, schema: Schema) -> list[Schema]:
        pass


class NoSchemaSplitting(SchemaPartitionStrategy):
    def partition_schema(self, schema: Schema) -> list[Schema]:
        return [schema]
