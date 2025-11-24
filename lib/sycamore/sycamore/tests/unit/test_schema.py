import json
from sycamore.schema import SchemaV2, make_property, make_named_property, NamedProperty, RegexValidator

single_property_dict_old = {
    "name": "state",
    "field_type": "string",
    "description": "Two-letter state code",
    "examples": ["NC"],
}

single_property_schema_old = {"fields": [single_property_dict_old]}

single_property_dict_old_property = {
    "name": "state",
    "property_type": "string",
    "description": "Two-letter state code",
    "examples": ["NC"],
}

single_property_schema_old_properties = {"properties": [single_property_dict_old_property]}


single_property_dict_new = {
    "name": "state",
    "type": {
        "type": "string",
        "description": "Two-letter state code",
        "examples": ["NC"],
    },
}

single_property_schema_new = {"properties": [single_property_dict_new]}

custom_type_dict_old = {
    "name": "email_address",
    "field_type": "email",
    "description": "Email address of the user",
}

custom_type_schema_old = {"fields": [custom_type_dict_old]}

custom_type_dict_new = {
    "name": "email_address",
    "type": {
        "type": "custom",
        "custom_type": "email",
        "description": "Email address of the user",
    },
}

custom_type_schema_new = {"properties": [custom_type_dict_new]}


def test_read_old_schema():
    schema = SchemaV2.model_validate(single_property_schema_old)
    assert len(schema.properties) == 1
    assert schema.properties[0].name == "state"
    assert schema.properties[0].type.type == "string"
    assert schema.properties[0].type.description == "Two-letter state code"
    assert schema.properties[0].type.examples == ["NC"]


def test_read_old_schema_properties():
    schema = SchemaV2.model_validate(single_property_schema_old_properties)
    assert len(schema.properties) == 1
    assert schema.properties[0].name == "state"
    assert schema.properties[0].type.type == "string"
    assert schema.properties[0].type.description == "Two-letter state code"
    assert schema.properties[0].type.examples == ["NC"]


def test_read_new_schema():
    schema = SchemaV2.model_validate(single_property_schema_new)
    assert len(schema.properties) == 1
    assert schema.properties[0].name == "state"
    assert schema.properties[0].type.type == "string"
    assert schema.properties[0].type.description == "Two-letter state code"
    assert schema.properties[0].type.examples == ["NC"]


def test_read_old_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_old)
    assert len(schema.properties) == 1
    assert schema.properties[0].name == "email_address"
    assert schema.properties[0].type.type == "custom"
    assert schema.properties[0].type.custom_type == "email"
    assert schema.properties[0].type.description == "Email address of the user"


def test_read_new_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_new)
    assert len(schema.properties) == 1
    assert schema.properties[0].name == "email_address"
    assert schema.properties[0].type.type == "custom"
    assert schema.properties[0].type.custom_type == "email"
    assert schema.properties[0].type.description == "Email address of the user"


nested_schema_dict = {
    "properties": [
        make_named_property(name="user_id", type="string", description="User ID"),
        make_named_property(
            name="location",
            type="object",
            properties=[
                make_named_property(name="city", type="string", description="City name"),
                make_named_property(name="state", type="string", description="State name"),
                make_named_property(name="country", type="string", description="Country name"),
                make_named_property(
                    name="years_resident",
                    type="object",
                    properties=[
                        make_named_property(name="start", type="int", description="Year residency started"),
                        make_named_property(name="end", type="int", description="Year residency ended"),
                    ],
                ),
            ],
        ),
        make_named_property(
            name="settings", type="array", item_type=make_property(type="string"), description="User settings"
        ),
        make_named_property(
            name="billing_status",
            type="choice",
            choices=["paid", "free", "trial"],
            description="Billing status of the user",
        ),
    ]
}

nested_schema = SchemaV2.model_validate(nested_schema_dict)


def test_flatten():

    expected = [
        ("user_id", "string"),
        ("location.city", "string"),
        ("location.state", "string"),
        ("location.country", "string"),
        ("location.years_resident.start", "int"),
        ("location.years_resident.end", "int"),
        ("billing_status", "choice"),
    ]

    flat_schema = nested_schema.flatten()
    flat_properties = [(prop.name, prop.type.type) for prop in flat_schema.properties]
    assert flat_properties == expected, "Flattened schema properties should match expected structure"


expected_flattened = """{
  "properties": [
    {
      "name": "user_id",
      "type": "string",
      "description": "User ID"
    },
    {
      "name": "location.city",
      "type": "string",
      "description": "City name"
    },
    {
      "name": "location.state",
      "type": "string",
      "description": "State name"
    },
    {
      "name": "location.country",
      "type": "string",
      "description": "Country name"
    },
    {
      "name": "location.years_resident.start",
      "type": "int",
      "description": "Year residency started"
    },
    {
      "name": "location.years_resident.end",
      "type": "int",
      "description": "Year residency ended"
    },
    {
      "name": "billing_status",
      "type": "choice",
      "description": "Billing status of the user",
      "choices": [
        "paid",
        "free",
        "trial"
      ]
    }
  ]
}"""


def test_render_flattened():
    flat_str = nested_schema.render_flattened()
    assert flat_str == expected_flattened, "Flattened schema string should match expected output"


compatible_schema = SchemaV2.model_validate(
    {
        "properties": [
            make_named_property(
                name="state", type="string", description="Two-letter state code", default="CA", examples=["NC"]
            ),
            make_named_property(
                name="city", type="string", description="City", examples=["San Francisco", "Los Angeles"]
            ),
        ]
    }
)


non_compatible_schema_type = SchemaV2.model_validate(
    {
        "properties": [
            make_named_property(
                name="state", type="string", description="Two-letter state code", default="CA", examples=["NC"]
            ),
            make_named_property(
                name="cities", type="array", item_type=make_property(type="string"), description="Cities"
            ),
        ]
    }
)


non_compatible_schema_param = SchemaV2.model_validate(
    {
        "properties": [
            make_named_property(
                name="state",
                type="string",
                description="Two-letter state code",
                extraction_instructions="May be present in an address.",
                default="CA",
                examples=["NC"],
            ),
            make_named_property(
                name="city", type="string", description="City", examples=["San Francisco", "Los Angeles"]
            ),
        ]
    }
)


def test_serialize_backwards_compat():
    res = compatible_schema.model_dump()
    assert len(res["properties"]) == 2

    assert res["properties"][0]["name"] == "state"
    assert res["properties"][0]["property_type"] == "string"
    assert res["properties"][0]["description"] == "Two-letter state code"
    assert res["properties"][0]["default"] == "CA"
    assert res["properties"][0]["examples"] == ["NC"]


def test_serialize_not_backwards_compat_type():
    res = non_compatible_schema_type.model_dump()
    assert len(res["properties"]) == 2
    assert set(res["properties"][0].keys()) == {"name", "type"}
    assert res["properties"][0]["name"] == "state"
    assert res["properties"][0]["type"]["type"] == "string"
    assert res["properties"][1]["type"]["type"] == "array"


def test_serialize_not_backwards_compat_param():
    res = non_compatible_schema_param.model_dump()
    assert len(res["properties"]) == 2
    assert set(res["properties"][0].keys()) == {"name", "type"}
    assert res["properties"][0]["name"] == "state"
    assert res["properties"][0]["type"]["type"] == "string"
    assert res["properties"][0]["type"]["extraction_instructions"] == "May be present in an address."


def test_serialize_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_new)
    res = schema.model_dump()
    res["properties"][0]["name"] == "email_address"
    res["properties"][0]["property_type"] == "email"
    res["properties"][0]["description"] == "Email address of the user"


def test_type_alias():
    str_prop = make_property(type="str")
    assert str_prop.type == "string", "Type alias 'str' should be converted to 'string'"

    bool_prop = make_property(type="boolean")
    assert bool_prop.type == "bool", "Type alias 'boolean' should be converted to 'bool'"

    obj_prop = make_property(type="struct", properties=[make_named_property(name="field", type="str")])
    assert obj_prop.type == "object", "Type alias 'struct' should be converted to 'object'"
    assert len(obj_prop.properties) == 1
    assert obj_prop.properties[0].name == "field"
    assert obj_prop.properties[0].type.type == "string"


def test_ziptraverse():
    from sycamore.utils.zip_traverse import zip_traverse

    traverseme_schema_dict = {
        "properties": [
            make_named_property(
                name="location",
                type="object",
                properties=[
                    make_named_property(name="city", type="string", description="City name"),
                    make_named_property(name="state", type="string", description="State name"),
                    make_named_property(name="country", type="string", description="Country name"),
                    make_named_property(
                        name="years_resident",
                        type="object",
                        properties=[
                            make_named_property(name="start", type="int", description="Year residency started"),
                            make_named_property(name="end", type="int", description="Year residency ended"),
                        ],
                    ),
                ],
            ),
            make_named_property(
                name="settings",
                type="array",
                item_type=make_property(
                    type="object",
                    properties=[
                        make_named_property(name="retention", type="bool", description="Is data retention enabled"),
                        make_named_property(name="student", type="bool", description="Is a student account"),
                    ],
                ),
                description="User settings",
            ),
        ]
    }
    sch = SchemaV2.model_validate(traverseme_schema_dict)
    for k, (v,), (p,) in zip_traverse(sch.as_object_property(), order="before"):
        if p is sch:
            assert k in ("location", "settings")
            assert v.name in ("location", "settings")
            continue
        if not isinstance(v, NamedProperty):
            assert p.name == "settings"
            continue
        if v.name in ("retention", "student"):
            assert p.type == "object"
            continue
        if v.name in ("city", "state", "country", "years_resident"):
            assert p.name == "location"
            continue
        if v.name in ("start", "end"):
            assert p.name == "years_resident"
            continue


def test_validator_json_serialize():
    r = RegexValidator(regex=r"[0-9]{3}")
    res, _ = r.validate_property("123")
    assert res

    js = json.dumps(r.model_dump())
    r2 = RegexValidator.model_validate_json(js)

    assert r.regex == r2.regex
    assert r.allowable_types == r2.allowable_types


# In order to facilitate compatibility, if deserialization fails, we fallback
# to the old schema format. In the past, this could cause weird error messages
# if you passed in a malformed schema, as you would always get an exception
# message from the old schema. This test verifies that we now raise the
# exception from the new schema. This could still be confusing if you have a
# malformed old schema, but given that we eventually expect people to use the
# new schema, this changes seems directionally correct.
def test_exception_fallback_message():
    bad_schema = {
        "properties": [
            {
                "name": "state",
                "type": {
                    "type": "strng",  # misspelled string
                    "description": "Two-letter state code",
                    "examples": ["NC"],
                },
            }
        ]
    }

    try:
        _ = SchemaV2.model_validate(bad_schema)
    except Exception as e:
        assert "strng" in str(e), "Exception should mention the invalid type 'strng'"
    else:
        # The old schema deserialization error is for missing 'field_type'.
        assert False, "Expected exception was not raised for invalid schema"
