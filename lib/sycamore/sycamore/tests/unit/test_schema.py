from sycamore.schema import SchemaV2, make_property, make_named_property

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
    assert (
        schema.model_dump(exclude_unset=True, exclude_none=True) == single_property_schema_new
    ), "Old schema should match the new schema format"


def test_read_old_schema_properties():
    schema = SchemaV2.model_validate(single_property_schema_old_properties)
    assert (
        schema.model_dump(exclude_unset=True, exclude_none=True) == single_property_schema_new
    ), "Old schema with properties should match the new schema format"


def test_read_new_schema():
    schema = SchemaV2.model_validate(single_property_schema_new)
    assert schema.model_dump(exclude_unset=True) == single_property_schema_new, "Schemas should match"


def test_read_old_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_old)
    assert schema.model_dump(exclude_unset=True, exclude_none=True) == custom_type_schema_new, "Schemas should match"


def test_read_new_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_new)
    assert schema.model_dump(exclude_unset=True, exclude_none=True) == custom_type_schema_new, "Schemas should match"


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
