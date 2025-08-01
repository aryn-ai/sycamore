from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional, Self
from pydantic import BaseModel, ConfigDict, Field

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import ObjectProperty, ArrayProperty, NamedProperty, SchemaV2, DataType
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


class RichProperty(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str]
    # TODO: Any -> DataType
    type: Any
    # TODO: Any -> Union[DataType.types]
    value: Any

    is_valid: bool = True

    attribution: list[int] = Field(default=[], repr=False)
    # TODO: Any -> Union[DataType.types]
    invalid_guesses: list[Any] = []

    llm_prompt: Optional[RenderedPrompt] = None

    @staticmethod
    def from_prediction(prediction: Any, attributable_elements: list[Element], name: Optional[str] = None) -> Self:
        if isinstance(prediction, dict):
            v_dict: dict[str, RichProperty] = {}
            for k, v in prediction.items():
                v_dict[k] = RichProperty.from_prediction(v, attributable_elements, name=k)
            return RichProperty(
                name=name,
                type=DataType.OBJECT,
                value=v_dict,
                attribution=[e.element_index for e in attributable_elements if e.element_index is not None],
            )
        if isinstance(prediction, list):
            v_list: list[RichProperty] = []
            for x in prediction:
                v_list.append(RichProperty.from_prediction(x, attributable_elements))
            return RichProperty(
                name=name,
                type=DataType.ARRAY,
                value=v_list,
                attribution=[e.element_index for e in attributable_elements if e.element_index is not None],
            )
        return RichProperty(
            name=name,
            type=type(prediction).__name__,
            value=prediction,
            attribution=[e.element_index for e in attributable_elements if e.element_index is not None],
        )


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
                existing_obj = existing_fields.get(name)
                if isinstance(existing_obj, RichProperty):
                    existing_obj = existing_obj.value
                if existing_obj is None:
                    existing_obj = {}
                new_obj = new_fields.get(name)
                if isinstance(new_obj, RichProperty):
                    new_obj = new_obj.value
                if new_obj is None:
                    new_obj = {}

                updated_obj, updated_spec = self._update_object(inner_prop.type, new_obj, existing_obj)
                updated_obj = RichProperty(name=name, type=DataType.OBJECT, value=updated_obj)
                if updated_spec is not None:
                    updated_specs.append(NamedProperty(name=name, type=updated_spec))
                updated_values[name] = updated_obj
                continue

            if inner_prop.type.type == DataType.ARRAY:
                new_arr = new_fields.get(name)
                if isinstance(new_arr, RichProperty):
                    new_arr = new_arr.value
                if new_arr is None:
                    new_arr = []
                existing_arr = existing_fields.get(name)
                if isinstance(existing_arr, RichProperty):
                    existing_arr = existing_arr.value
                if existing_arr is None:
                    existing_arr = []

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
