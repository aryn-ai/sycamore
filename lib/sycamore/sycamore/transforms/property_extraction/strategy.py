from abc import ABC, abstractmethod
from typing import Iterable
from pydantic import BaseModel

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.schema import ObjectProperty, NamedProperty, SchemaV2, DataType
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.utils.zip_traverse import ZTLeaf, zip_traverse


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
                    trim = trim and ef.is_valid
            elif nf is not None:
                if prop.get_type() not in (DataType.ARRAY, DataType.OBJECT):
                    trim = True
                out_p.value[k] = nf
                trim = trim and nf.is_valid

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
                    opc = prop.unwrap().model_copy()
                    opc.properties = []
                    np = NamedProperty(name=prop.name, type=opc)
                else:
                    np = prop
                obj_p = out_prop_p.type if isinstance(out_prop_p, NamedProperty) else out_prop_p
                assert isinstance(obj_p, ObjectProperty)
                proplist = obj_p.properties
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
                obj_p = prop_p.type if isinstance(prop_p, NamedProperty) else prop_p
                assert isinstance(obj_p, ObjectProperty)
                obj_p.properties.remove(prop)

        return SchemaUpdateResult(
            out_schema=SchemaV2(properties=out_sch.properties),
            out_fields=out_rp.value,
            completed=len(out_sch.properties) == 0,
        )


default_stepthrough = OneElementAtATime()
default_schema_partition = NoSchemaSplitting()
default_schema_update = TakeFirstTrimSchema()
