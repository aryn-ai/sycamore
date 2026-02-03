from sycamore.datatype import DataType
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.transforms.property_extraction.utils import stitch_together_objects, remove_keys_recursive


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


def test_remove_keys_recursive():
    obj = {
        "properties": [
            {
                "name": "document_title",
                "type": {
                    "type": "string",
                    "required": False,
                    "description": "The title of the document",
                    "default": None,
                    "extraction_instructions": None,
                    "examples": ["NON-BINDING LETTER OF INTENT"],
                    "source": None,
                    "validators": [],
                },
            },
            {
                "name": "document_date",
                "type": {
                    "type": "date",
                    "required": False,
                    "description": "The date the document was issued",
                    "default": None,
                    "extraction_instructions": None,
                    "examples": ["2008-12-01"],
                    "source": None,
                    "validators": [],
                    "format": "YYYY-MM-DD",
                },
            },
            {
                "name": "parties",
                "type": {
                    "type": "array",
                    "required": False,
                    "description": "The parties involved in the agreement",
                    "default": None,
                    "extraction_instructions": None,
                    "examples": [
                        {"address": "3 Vaughan Ave Doncaster, XO DN1 2QE", "name": "Cynergi Holdings, Inc."},
                        {
                            "address": "2348 Lucerne Road, Suite 172 Mount-Royal, QC H3R 4J8",
                            "name": "Sports Supplement Acquisition Group, Inc.",
                        },
                    ],
                    "source": None,
                    "validators": [],
                    "item_type": {
                        "type": "object",
                        "required": False,
                        "description": "Information about a party involved in the agreement",
                        "default": None,
                        "extraction_instructions": None,
                        "examples": None,
                        "source": None,
                        "validators": [],
                        "properties": [
                            {
                                "name": "name",
                                "type": {
                                    "type": "string",
                                    "required": False,
                                    "description": "The name of the party",
                                    "default": None,
                                    "extraction_instructions": None,
                                    "examples": None,
                                    "source": None,
                                    "validators": [],
                                },
                            },
                            {
                                "name": "address",
                                "type": {
                                    "type": "string",
                                    "required": False,
                                    "description": "The address of the party",
                                    "default": None,
                                    "extraction_instructions": None,
                                    "examples": None,
                                    "source": None,
                                    "validators": [],
                                },
                            },
                        ],
                    },
                },
            },
        ]
    }

    cleaned = remove_keys_recursive(obj)
    assert cleaned == {
        "properties": [
            {
                "name": "document_title",
                "type": {
                    "type": "string",
                    "description": "The title of the document",
                    "examples": ["NON-BINDING LETTER OF INTENT"],
                },
            },
            {
                "name": "document_date",
                "type": {
                    "type": "date",
                    "description": "The date the document was issued",
                    "examples": ["2008-12-01"],
                    "format": "YYYY-MM-DD",
                },
            },
            {
                "name": "parties",
                "type": {
                    "type": "array",
                    "description": "The parties involved in the agreement",
                    "examples": [
                        {"address": "3 Vaughan Ave Doncaster, XO DN1 2QE", "name": "Cynergi Holdings, Inc."},
                        {
                            "address": "2348 Lucerne Road, Suite 172 Mount-Royal, QC H3R 4J8",
                            "name": "Sports Supplement Acquisition Group, Inc.",
                        },
                    ],
                    "item_type": {
                        "type": "object",
                        "description": "Information about a party involved in the agreement",
                        "examples": None,
                        "properties": [
                            {
                                "name": "name",
                                "type": {"type": "string", "description": "The name of the party", "examples": None},
                            },
                            {
                                "name": "address",
                                "type": {"type": "string", "description": "The address of the party", "examples": None},
                            },
                        ],
                    },
                },
            },
        ]
    }
