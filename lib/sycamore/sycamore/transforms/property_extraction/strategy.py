from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional
from pydantic import BaseModel, ConfigDict

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import Schema
from sycamore.llms.prompts.prompts import RenderedPrompt


class StepThroughStrategy(ABC):
    @abstractmethod
    def step_through(self, document: Document) -> Iterable[list[Element]]:
        pass


class OneElementAtATime(StepThroughStrategy):
    def step_through(self, document: Document) -> Iterable[list[Element]]:
        for elt in document.elements:
            yield [elt]


class NPagesAtATime(StepThroughStrategy):
    def __init__(self, n: int = 1):
        self._n = n

    def step_through(self, document: Document) -> Iterable[list[Element]]:
        batch: list[Element] = []
        cutoff = document.elements[0].properties["page_number"] + self._n
        for elt in document.elements:
            pn = elt.properties["page_number"]
            if pn >= cutoff:
                yield batch
                cutoff = pn + self._n
                batch = [elt]
            else:
                batch.append(elt)
        if len(batch) > 0:
            yield batch


class SchemaPartitionStrategy(ABC):
    @abstractmethod
    def partition_schema(self, schema: Schema) -> list[Schema]:
        pass


class NoSchemaSplitting(SchemaPartitionStrategy):
    def partition_schema(self, schema: Schema) -> list[Schema]:
        return [schema]


class RichProperty(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    # TODO: Any -> DataType
    type: Any
    # TODO: Any -> Union[DataType.types]
    value: Any

    is_valid: bool = True

    attribution: list[int] = []
    # TODO: Any -> Union[DataType.types]
    invalid_guesses: list[Any] = []

    llm_prompt: Optional[RenderedPrompt] = None


class SchemaUpdateResult(BaseModel):
    out_schema: Schema
    out_fields: dict[str, RichProperty]
    completed: bool


class SchemaUpdateStrategy(ABC):
    @abstractmethod
    def update_schema(
        self, in_schema: Schema, new_fields: dict[str, RichProperty], existing_fields: dict[str, RichProperty] = dict()
    ) -> SchemaUpdateResult:
        pass


class TakeFirstTrimSchema(SchemaUpdateStrategy):
    def update_schema(
        self, in_schema: Schema, new_fields: dict[str, RichProperty], existing_fields: dict[str, RichProperty] = dict()
    ) -> SchemaUpdateResult:
        expected_fields = set(f.name for f in in_schema.fields)
        arrays = set(sf.name for sf in in_schema.fields if sf.field_type == "array")
        for k, v in new_fields.items():
            if v.value is not None and k in expected_fields:
                if k not in existing_fields:
                    existing_fields[k] = v
                elif k in arrays:
                    existing_fields[k].value.extend(v.value)
                    existing_fields[k].attribution.extend(v.attribution)

        missing_fields = expected_fields - set(existing_fields.keys())
        out_schema = Schema(fields=[sf for sf in in_schema.fields if sf.name in missing_fields | arrays])

        return SchemaUpdateResult(
            out_schema=out_schema,
            out_fields=existing_fields,
            completed=len(out_schema.fields) == 0 and len(arrays) == 0,
        )


default_stepthrough = OneElementAtATime()
default_schema_partition = NoSchemaSplitting()
default_schema_update = TakeFirstTrimSchema()
