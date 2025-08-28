from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional
from pydantic import BaseModel

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import ObjectProperty, ArrayProperty, NamedProperty, SchemaV2, DataType, ValidatorType
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.utils.zt import ZTLeaf, zip_traverse


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

    def _validate_prop(self, validators: list[ValidatorType], value: RichProperty):
        for validator in validators:
            valid, new_val = validator.validate_property(value.value)
            value.is_valid = valid
            value.value = new_val
            if not valid:
                break

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
                self._validate_prop(inner_prop.type.validators, updated_obj_p)
                updated_values[name] = updated_obj_p
                continue

            if inner_prop.type.type == DataType.ARRAY:
                existing_arr = self._get_field_or(existing_fields, name, [])
                new_arr = self._get_field_or(new_fields, name, [])

                assert isinstance(inner_prop.type, ArrayProperty)
                updated_arr = self._update_array(inner_prop.type, new_arr, existing_arr)
                updated_arr_p = RichProperty(name=name, type=DataType.ARRAY, value=updated_arr)
                self._validate_prop(inner_prop.type.validators, updated_arr_p)
                updated_values[name] = updated_arr_p
                updated_specs.append(inner_prop)
                continue

            if (v := existing_fields.get(name)) is not None and v.value is not None:
                if not v.is_valid and name in new_fields and new_fields[name] is not None:
                    nf = new_fields[name]
                    self._validate_prop(inner_prop.type.validators, nf)
                    if nf.is_valid:
                        updated_values[name] = nf
                        continue
                updated_values[name] = v
                continue

            if (v := new_fields.get(name)) is not None and v.value is not None:
                self._validate_prop(inner_prop.type.validators, v)
                updated_values[name] = v
                continue

            updated_specs.append(inner_prop)

        if len(updated_specs) > 0:
            return existing_fields | updated_values, ObjectProperty(properties=updated_specs)
        return existing_fields | updated_values, None

    def _update_array(
        self, in_arr_spec: ArrayProperty, new_fields: list[RichProperty], existing_fields: list[RichProperty]
    ) -> list[RichProperty]:
        inner_t = in_arr_spec.item_type
        updated_new_fields = []
        for nf in new_fields:
            if inner_t.type == DataType.OBJECT:
                obj, _ = self._update_object(inner_t, nf.value, {})
                rp = RichProperty(name=None, type=DataType.OBJECT, value=obj)
                self._validate_prop(inner_t.validators, rp)
                updated_new_fields.append(rp)
                continue
            if inner_t.type == DataType.ARRAY:
                arr = self._update_array(inner_t, nf.value, [])
                rp = RichProperty(name=None, type=DataType.ARRAY, value=arr)
                self._validate_prop(inner_t.validators, rp)
                updated_new_fields.append(rp)
                continue

            self._validate_prop(inner_t.validators, nf)
            updated_new_fields.append(nf)

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


class TakeFirstTrimSchemaZT(SchemaUpdateStrategy):
    def update_schema(
        self,
        in_schema: SchemaV2,
        new_fields: dict[str, RichProperty],
        existing_fields: dict[str, RichProperty] = dict(),
    ) -> SchemaUpdateResult:
        sch_obj = in_schema.as_object_property()
        nf_rp = RichProperty(type=DataType.OBJECT, value=new_fields, name=None)
        ef_rp = RichProperty(type=DataType.OBJECT, value=existing_fields, name=None)
        out_rp = RichProperty(type=DataType.OBJECT, value={}, name=None)
        out_sch = ObjectProperty(properties=[])

        for k, (prop, nf, ef, out, out_prop), (prop_p, nf_p, ef_p, out_p, out_prop_p) in zip_traverse(
            sch_obj, nf_rp, ef_rp, out_rp, out_sch, order="before", intersect_keys=False
        ):
            # Schema might have been trimmed, so keep existing fields (that may have
            # come from a previous extraction step)
            if prop is None:
                if ef is not None:
                    out_p.value[k] = ef
                continue

            # If field in existing fields (ef), take that
            # Else if field in new fields (nf), take that
            # If field is in both and it's an Array, concat them
            # If field is a leaf and exists in either, trim it out
            trim = False
            if ef is not None:
                if prop.get_type() not in (DataType.ARRAY, DataType.OBJECT):
                    trim = True
                if nf is not None and prop.get_type() is DataType.ARRAY:
                    out_p.value[k] = RichProperty(value=ef.value + nf.value, type=DataType.ARRAY, name=ef.name)
                    nf.value = []
                    ef.value = []
                elif not ef.is_valid and nf is not None and nf.is_valid:
                    out_p.value[k] = nf
                else:
                    out_p.value[k] = ef
            elif nf is not None:
                if prop.get_type() not in (DataType.ARRAY, DataType.OBJECT):
                    trim = True
                out_p.value[k] = nf

            # If this property should not be trimmed (was not found or is an array/object)
            # Add it to the parent property list if applicable. Array properties are added
            # with full item_type, which means that fields inside arrays of objects are not
            # trimmed
            if (
                not trim
                and isinstance(prop, NamedProperty)
                and prop_p.get_type() is DataType.OBJECT
                and not isinstance(out_prop_p, ZTLeaf)
            ):
                if prop.get_type() is DataType.OBJECT:
                    np = NamedProperty(name=prop.name, type=ObjectProperty(properties=[]))
                else:
                    np = prop
                proplist = (out_prop_p.type if isinstance(out_prop_p, NamedProperty) else out_prop_p).properties
                if not any(p.name == np.name for p in proplist):
                    proplist.append(np)

        # Finally, drop empty objects in the out schema. We couldn't do this in the
        # previous trimming operation bc we didn't know if the object would be empty
        # or not
        for k, (prop,), (prop_p,) in zip_traverse(out_sch, order="after"):
            if (
                prop.get_type() is DataType.OBJECT
                and prop_p.get_type() is DataType.OBJECT
                and len(prop.type.properties) == 0
            ):
                if isinstance(prop_p, NamedProperty):
                    prop_p.type.properties.remove(prop)
                else:
                    prop_p.properties.remove(prop)

        return SchemaUpdateResult(
            out_schema=SchemaV2(properties=out_sch.properties),
            out_fields=out_rp.value,
            completed=len(out_sch.properties) == 0,
        )
