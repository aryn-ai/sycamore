from sycamore.schema import SchemaV2

single_property_dict_old = {
    "name": "state",
    "field_type": "string",
    "description": "Two-letter state code",
    "examples": ["NC"],
}

single_property_schema_old = {"fields": [single_property_dict_old]}

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
        schema.model_dump(exclude_unset=True) == single_property_schema_new
    ), "Old schema should match the new schema format"


def test_read_new_schema():
    schema = SchemaV2.model_validate(single_property_schema_new)
    assert schema.model_dump(exclude_unset=True) == single_property_schema_new, "Schemas should match"


def test_read_old_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_old)
    assert schema.model_dump(exclude_unset=True, exclude_none=True) == custom_type_schema_new, "Schemas should match"


def test_read_new_custom_type():
    schema = SchemaV2.model_validate(custom_type_schema_new)
    assert schema.model_dump(exclude_unset=True, exclude_none=True) == custom_type_schema_new, "Schemas should match"
