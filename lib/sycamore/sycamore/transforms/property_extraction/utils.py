from typing import Optional, Any
from pydantic import TypeAdapter
from sycamore.schema import (
    NamedProperty,
    DataType,
    PropertyType,
)
from sycamore.transforms.property_extraction.types import RichProperty


def create_named_property(prop_data: dict[str, Any], n_examples: Optional[int] = None) -> NamedProperty:
    name = prop_data["name"]

    if (declared_type := prop_data["type"]) not in DataType.values():
        prop_data["custom_type"] = declared_type
        prop_data["type"] = DataType.CUSTOM

    if n_examples is not None:
        prop_data["examples"] = list(set(prop_data.get("examples", [])))[:n_examples]

    prop_type: PropertyType = TypeAdapter(PropertyType).validate_python(prop_data)

    return NamedProperty(
        name=name,
        type=prop_type,
    )


def stitch_together_objects(ob1: RichProperty, ob2: RichProperty) -> RichProperty:
    if not isinstance(ob1, RichProperty):
        # Reachable by document.properties.entity_metadata not
        # containing pydantic rich properties (dicts instead)
        ob1 = RichProperty.validate_recursive(ob1)
    if not isinstance(ob2, RichProperty):
        ob2 = RichProperty.validate_recursive(ob2)
    if ob1.type == DataType.ARRAY and ob2.type == DataType.ARRAY:
        ret = ob1.model_copy()
        ret.value += ob2.value
        return ret

    if ob1.type == DataType.OBJECT and ob2.type == DataType.OBJECT:
        rd = {}
        for k in ob1.value.keys():
            if k in ob2.value:
                rd[k] = stitch_together_objects(ob1.value[k], ob2.value[k])
            else:
                rd[k] = ob1.value[k]
        for k in ob2.value.keys():
            if k not in rd:
                rd[k] = ob2.value[k]
        ret = ob1.model_copy()
        ret.value = rd
        return ret

    if ob1 == ob2:
        return ob1

    raise NotImplementedError(f"Cannot stitch together objects with types {ob1.type} and {ob2.type}")
