from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional
from pydantic import BaseModel

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import ObjectProperty, ArrayProperty, NamedProperty, SchemaV2, DataType
from sycamore.transforms.property_extraction.types import RichProperty


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
    def partition_schema(self, schema: SchemaV2) -> list[SchemaV2]:
        pass


class NoSchemaSplitting(SchemaPartitionStrategy):
    def partition_schema(self, schema: SchemaV2) -> list[SchemaV2]:
        return [schema]


class SchemaUpdateResult(BaseModel):
    out_schema: SchemaV2
    out_fields: dict[str, RichProperty]
    completed: bool


class SchemaUpdateStrategy(ABC):
    @abstractmethod
    def update_schema(
        self,
        in_schema: SchemaV2,
        new_fields: dict[str, RichProperty],
        existing_fields: dict[str, RichProperty] = dict(),
    ) -> SchemaUpdateResult:
        pass


class TakeFirstTrimSchema(SchemaUpdateStrategy):

    def _get_field_or(self, fields: dict[str, RichProperty], name: str, default: Any) -> Any:
        ret = fields.get(name)
        if isinstance(ret, RichProperty):
            ret = ret.value
        if ret is None:
            ret = default
        return ret

    def _update_object(
        self,
        in_obj_spec: ObjectProperty | SchemaV2,
        new_fields: dict[str, RichProperty],
        existing_fields: dict[str, RichProperty],
    ) -> tuple[dict[str, RichProperty], Optional[ObjectProperty]]:
        updated_specs = []
        updated_values = {}
        for inner_prop in in_obj_spec.properties:
            name = inner_prop.name
            if inner_prop.type.type == DataType.OBJECT:
                existing_obj = self._get_field_or(existing_fields, name, {})
                new_obj = self._get_field_or(new_fields, name, {})

                updated_obj, updated_spec = self._update_object(inner_prop.type, new_obj, existing_obj)
                updated_obj_p = RichProperty(name=name, type=DataType.OBJECT, value=updated_obj)
                if updated_spec is not None:
                    updated_specs.append(NamedProperty(name=name, type=updated_spec))
                updated_values[name] = updated_obj_p
                continue

            if inner_prop.type.type == DataType.ARRAY:
                existing_arr = self._get_field_or(existing_fields, name, [])
                new_arr = self._get_field_or(new_fields, name, [])

                assert isinstance(inner_prop.type, ArrayProperty)
                updated_arr = self._update_array(inner_prop.type, new_arr, existing_arr)
                updated_values[name] = RichProperty(name=name, type=DataType.ARRAY, value=updated_arr)
                updated_specs.append(inner_prop)
                continue

            if (v := existing_fields.get(name)) is not None and v.value is not None:
                updated_values[name] = v
                continue

            if (v := new_fields.get(name)) is not None and v.value is not None:
                updated_values[name] = v
                continue

            updated_specs.append(inner_prop)

        if len(updated_specs) > 0:
            return existing_fields | updated_values, ObjectProperty(properties=updated_specs)
        return existing_fields | updated_values, None

    def _update_array(
        self, in_arr_spec: ArrayProperty, new_fields: list[RichProperty], existing_fields: list[RichProperty]
    ) -> list[RichProperty]:
        return existing_fields + new_fields

    def update_schema(
        self,
        in_schema: SchemaV2,
        new_fields: dict[str, RichProperty],
        existing_fields: dict[str, RichProperty] = dict(),
    ) -> SchemaUpdateResult:
        out_fields, out_schema_obj = self._update_object(in_schema, new_fields, existing_fields)
        if out_schema_obj is None:
            out_schema = SchemaV2(properties=[])
        else:
            out_schema = SchemaV2(properties=out_schema_obj.properties)
        return SchemaUpdateResult(
            out_schema=out_schema,
            out_fields=out_fields,
            completed=len(out_schema.properties) == 0,
        )


default_stepthrough = OneElementAtATime()
default_schema_partition = NoSchemaSplitting()
default_schema_update = TakeFirstTrimSchema()
