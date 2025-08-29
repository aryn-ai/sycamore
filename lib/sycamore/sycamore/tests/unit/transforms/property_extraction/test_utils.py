from sycamore.schema import DataType
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.transforms.property_extraction.utils import stitch_together_objects


def test_stitch_together_objects():
    ob1 = RichProperty.from_prediction(
        prediction={
            "a": "aval",
            "b": {
                "b1": "b1val",
                "b3": "b3val",
            },
            "c": [
                "c1",
                "c2",
                "c3",
            ],
        },
    )
    ob2 = RichProperty.from_prediction(
        prediction={
            "b": {
                "b2": "b2val",
                "b4": "b4val",
            },
            "c": ["c3", "c4", "c5"],
            "d": "dval",
        },
    )

    ob3 = stitch_together_objects(ob1, ob2)
    assert ob3.type == DataType.OBJECT
    ob3d = ob3.value
    assert isinstance(ob3d, dict)
    assert ob3d["a"].value == "aval"
    assert ob3d["b"].value["b1"].value == "b1val"
    assert ob3d["b"].value["b2"].value == "b2val"
    assert ob3d["b"].value["b3"].value == "b3val"
    assert ob3d["b"].value["b4"].value == "b4val"
    assert [c.value for c in ob3d["c"].value] == ["c1", "c2", "c3", "c3", "c4", "c5"]
    assert ob3d["d"].value == "dval"
