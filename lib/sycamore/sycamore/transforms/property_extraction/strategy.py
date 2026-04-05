import logging
from abc import ABC, abstractmethod
from typing import Iterable, Any
from pydantic import BaseModel

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.datatype import DataType
from sycamore.schema import ObjectProperty, NamedProperty, SchemaV2
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
                if prop.get_type() not in (DataType.ARRAY, DataType.OBJECT, DataType.BOOL):
                    trim = True
                if nf is not None and prop.get_type() in (DataType.ARRAY, DataType.BOOL):
                    if prop.get_type() is DataType.ARRAY:
                        ef.value = [] if ef.value is None else ef.value
                        nf.value = [] if nf.value is None else nf.value
                        if prop.type.item_type.type in (DataType.ARRAY, DataType.OBJECT):
                            combined = ef.value + nf.value
                        else:
                            if isinstance(ef.value, list) and isinstance(nf.value, list) and ef.value is not None and nf.value is not None:
                                combined = self.dedup_rp_array(ef.value + nf.value)
                            else:
                                logging.warning(f"Unexpected values: {ef.value}, {nf.value}.  Can't combine these.")
                                combined = ef.value
                        out_p.value[k] = RichProperty(value=combined, type=DataType.ARRAY, name=ef.name)
                        nf.value = []
                        ef.value = []
                    elif prop.get_type() is DataType.BOOL:
                        if ef.value is not None:
                            if ef.value is True:
                                out_p.value[k] = ef  # Already true so keep it and trim.
                                trim = True
                            elif ef.value is False and nf.value is True:  # Flip to true from false, take the new value.
                                out_p.value[k] = nf
                                trim = True
                            else:
                                out_p.value[k] = ef  # Keep the old value until we flip.
                elif not ef.is_valid and nf is not None and nf.is_valid:
                    out_p.value[k] = nf
                else:
                    out_p.value[k] = ef
                    trim = trim and ef.is_valid
            elif nf is not None:
                if prop.get_type() not in (DataType.ARRAY, DataType.OBJECT, DataType.BOOL):
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
                    assert isinstance(opc, ObjectProperty), "Type narrowing, unreachable"
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

    def dedup_rp_array(self, rp_list: list[RichProperty]) -> list[RichProperty]:
        rp_map: dict[Any, RichProperty] = {}
        for rp in rp_list:
            if rp.value is None:
                continue
            if rp.value not in rp_map:
                rp_map[rp.value] = rp
            else:
                attribution = rp_map[rp.value].attribution
                if attribution is None or rp.attribution is None:
                    continue
                if attribution.element_indices is None:
                    attribution.element_indices = []
                if rp.attribution.element_indices is None:
                    rp.attribution.element_indices = []
                attribution.element_indices.extend(rp.attribution.element_indices)
                p = attribution.page
                if p is None:
                    p = []
                elif not isinstance(p, list):
                    p = [p]
                pp = rp.attribution.page
                if pp is None:
                    pp = []
                elif not isinstance(pp, list):
                    pp = [pp]
                p.extend(pp)
                pages = set(p)
                attribution.page = sorted(list(pages))
        return sorted([rp for rp in rp_map.values()], key=lambda rp: rp.value)


default_stepthrough = OneElementAtATime()
default_schema_partition = NoSchemaSplitting()
default_schema_update = TakeFirstTrimSchema()
