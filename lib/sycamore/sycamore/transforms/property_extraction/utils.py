from typing import Optional, Any
from pydantic import TypeAdapter
from sycamore.schema import (
    NamedProperty,
    DataType,
    PropertyType,
)


def create_named_property(prop_data: dict[str, Any], n_examples: Optional[int] = None) -> NamedProperty:
    name = prop_data["name"]

    if (declared_type := prop_data["type"]) not in DataType.values():
        prop_data["custom_type"] = declared_type
        prop_data["type"] = DataType.CUSTOM

    if n_examples is not None:
        prop_data["examples"] = list(set(prop_data.get("examples", [])))[:n_examples]

    prop_type = TypeAdapter(PropertyType).validate_python(prop_data)

    return NamedProperty(
        name=name,
        type=prop_type,
    )
