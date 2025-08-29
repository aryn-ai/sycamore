from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.transforms.property_extraction.utils import recursively_sorted

import json


class TestRichProperty:

    def test_load_dump_to_python(self):
        pred = {"a": {"b": "c", "d": ["e", "f", "g"], "h": {"i": "j"}}, "k": "l"}

        rp = RichProperty.from_prediction(pred)
        rp_py = rp.to_python()

        assert json.dumps(pred) == json.dumps(recursively_sorted(rp_py))

    def test_load_dump_pydantic(self):
        pred = {"a": {"b": "c", "d": ["e", "f", "g"], "h": {"i": "j"}}, "k": "l"}
        rp0 = RichProperty.from_prediction(pred)
        rp_dump = rp0.model_dump()
        rp1 = RichProperty.validate_recursive(rp_dump)
        assert rp0 == rp1
