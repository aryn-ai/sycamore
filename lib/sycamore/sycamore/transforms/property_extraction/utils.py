from typing import Any
import json
from sycamore.schema import DataType
from sycamore.transforms.property_extraction.types import RichProperty


def dedup_examples(x: list[Any]) -> list[Any]:
    """
    Recursively deduplicate a list of items, ensuring that nested lists and dicts are also deduplicated.
    """

    def _recursively_sorted(obj):
        """Recursively sort lists and dicts for consistent hashing/serialization."""
        if isinstance(obj, dict):
            return {k: _recursively_sorted(obj[k]) for k in sorted(obj)}
        if isinstance(obj, list):
            # Sort lists of hashable items, otherwise sort by their JSON representation
            try:
                return sorted((_recursively_sorted(i) for i in obj), key=lambda x: json.dumps(x, sort_keys=True))
            except TypeError:
                return [_recursively_sorted(i) for i in obj]
        return obj

    ret_val = [json.loads(s) for s in {json.dumps(_recursively_sorted(d), sort_keys=True) for d in x}]
    return ret_val


def remove_keys_recursive(
    obj: Any, keys_to_remove: set[str] = {"required", "default", "extraction_instructions", "source", "validators"}
) -> Any:
    if isinstance(obj, dict):
        # Remove unwanted keys at this level
        return {k: remove_keys_recursive(v) for k, v in obj.items() if k not in keys_to_remove}
    elif isinstance(obj, list):
        # Recurse into lists
        return [remove_keys_recursive(item) for item in obj]
    else:
        # Base case: return the object as is
        return obj


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
