from sycamore.schema import SchemaV2, make_property, make_named_property, BooleanExpValidator

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

    obj_prop = make_property(type="struct", properties=[make_named_property(name="field", type="str")])
    assert obj_prop.type == "object", "Type alias 'struct' should be converted to 'object'"
    assert len(obj_prop.properties) == 1
    assert obj_prop.properties[0].name == "field"
    assert obj_prop.properties[0].type.type == "string"


# BooleanExpValidator Tests


def test_boolean_validator_basic_comparisons():
    """Test basic comparison operators."""
    # Test > operator
    validator = BooleanExpValidator(expression="> 0")
    assert validator.validate_property(5)[0] is True
    assert validator.validate_property(0)[0] is False
    assert validator.validate_property(-1)[0] is False

    # Test >= operator
    validator = BooleanExpValidator(expression=">= 0")
    assert validator.validate_property(5)[0] is True
    assert validator.validate_property(0)[0] is True
    assert validator.validate_property(-1)[0] is False

    # Test < operator
    validator = BooleanExpValidator(expression="< 100")
    assert validator.validate_property(50)[0] is True
    assert validator.validate_property(100)[0] is False
    assert validator.validate_property(150)[0] is False

    # Test <= operator
    validator = BooleanExpValidator(expression="<= 100")
    assert validator.validate_property(50)[0] is True
    assert validator.validate_property(100)[0] is True
    assert validator.validate_property(150)[0] is False

    # Test == operator
    validator = BooleanExpValidator(expression="== 42")
    assert validator.validate_property(42)[0] is True
    assert validator.validate_property(43)[0] is False

    # Test != operator
    validator = BooleanExpValidator(expression="!= 0")
    assert validator.validate_property(1)[0] is True
    assert validator.validate_property(0)[0] is False


def test_boolean_validator_range_comparisons():
    """Test range comparisons with and operator."""
    # Test year range validation
    validator = BooleanExpValidator(expression="> 1800 and <= 2025")
    assert validator.validate_property(1995)[0] is True
    assert validator.validate_property(2008)[0] is True
    assert validator.validate_property(1800)[0] is False  # Not > 1800
    assert validator.validate_property(2025)[0] is True
    assert validator.validate_property(2026)[0] is False  # Not <= 2025
    assert validator.validate_property(1750)[0] is False  # Not > 1800

    # Test percentage range validation
    validator = BooleanExpValidator(expression=">= 0 and <= 100")
    assert validator.validate_property(92.5)[0] is True
    assert validator.validate_property(0)[0] is True
    assert validator.validate_property(100)[0] is True
    assert validator.validate_property(-1)[0] is False
    assert validator.validate_property(101)[0] is False


def test_boolean_validator_length_function():
    """Test len() function calls."""
    # Test len() > 0 for non-empty strings
    validator = BooleanExpValidator(expression="len() > 0")
    assert validator.validate_property("The Metropolitan Tower")[0] is True
    assert validator.validate_property("")[0] is False
    assert validator.validate_property("A")[0] is True
    assert validator.validate_property(None)[0] is False

    # Test len() with comparison
    validator = BooleanExpValidator(expression="len() >= 5")
    assert validator.validate_property("Hello")[0] is True
    assert validator.validate_property("Hi")[0] is False
    assert validator.validate_property("World!")[0] is True


def test_boolean_validator_null_checks():
    """Test 'is null' checks."""
    # Test simple null check
    validator = BooleanExpValidator(expression="is null")
    assert validator.validate_property(None)[0] is True
    assert validator.validate_property("not null")[0] is False
    assert validator.validate_property(0)[0] is False

    # Test positive or null
    validator = BooleanExpValidator(expression="> 0 or is null")
    assert validator.validate_property(150)[0] is True
    assert validator.validate_property(0)[0] is False
    assert validator.validate_property(-1)[0] is False
    assert validator.validate_property(None)[0] is True
    assert validator.validate_property(0.1)[0] is True

    # Test non-negative or null
    validator = BooleanExpValidator(expression=">= 0 or is null")
    assert validator.validate_property(17500000.0)[0] is True
    assert validator.validate_property(0)[0] is True
    assert validator.validate_property(-1)[0] is False
    assert validator.validate_property(None)[0] is True

    # Test len() or null
    validator = BooleanExpValidator(expression="len() > 0 or is null")
    assert validator.validate_property("ABC Real Estate Partners")[0] is True
    assert validator.validate_property("")[0] is False
    assert validator.validate_property(None)[0] is True
    assert validator.validate_property("A")[0] is True


def test_boolean_validator_field_references():
    """Test field references in expressions."""
    # Test cross-field validation
    validator = BooleanExpValidator(expression="> year_built and <= 2025 or is null")

    # Valid cases
    assert validator.validate_property(2018, {"year_built": 1995})[0] is True
    assert validator.validate_property(2015, {"year_built": 2008})[0] is True  # 2015 > 2008 and <= 2025
    assert validator.validate_property(None, {"year_built": 1995})[0] is True  # is null

    # Invalid cases
    assert validator.validate_property(1990, {"year_built": 1995})[0] is False  # 1990 not > 1995
    assert validator.validate_property(2026, {"year_built": 1995})[0] is False  # 2026 not <= 2025
    assert validator.validate_property(2000, {})[0] is False  # no year_built in context


def test_boolean_validator_logical_operators():
    """Test logical operators (and, or, not)."""
    # Test 'and' operator
    validator = BooleanExpValidator(expression="> 10 and < 20")
    assert validator.validate_property(15)[0] is True
    assert validator.validate_property(5)[0] is False
    assert validator.validate_property(25)[0] is False

    # Test 'or' operator
    validator = BooleanExpValidator(expression="< 10 or > 20")
    assert validator.validate_property(5)[0] is True
    assert validator.validate_property(25)[0] is True
    assert validator.validate_property(15)[0] is False

    # Test 'not' operator
    validator = BooleanExpValidator(expression="not == 0")
    assert validator.validate_property(1)[0] is True
    assert validator.validate_property(0)[0] is False


def test_boolean_validator_constraint_string():
    """Test constraint string generation."""
    validator = BooleanExpValidator(expression="> 0 and <= 100")
    expected = "must satisfy the boolean expression: `> 0 and <= 100`"
    assert validator.constraint_string() == expected


def test_boolean_validator_error_handling():
    """Test error handling for invalid expressions."""
    # Test with malformed expression
    validator = BooleanExpValidator(expression="invalid expression ++")
    is_valid, processed_value = validator.validate_property(42)
    assert is_valid is False  # Should fail gracefully
    assert processed_value == 42  # Value should be returned unchanged

    # Test with division by zero-like scenario
    validator = BooleanExpValidator(expression="> nonexistent_field")
    is_valid, processed_value = validator.validate_property(42, {})
    assert is_valid is False  # Should fail when field doesn't exist
    assert processed_value == 42


def test_boolean_validator_allowable_types():
    """Test that validator has correct allowable types."""
    from sycamore.schema import DataType

    validator = BooleanExpValidator(expression="> 0")
    expected_types = {DataType.STRING, DataType.FLOAT, DataType.INT, DataType.BOOL}
    assert validator.allowable_types == expected_types


def test_boolean_validator_with_property():
    from sycamore.schema import FloatProperty

    # Create a property with boolean expression validator
    prop_dict = {"type": "float", "validators": [{"type": "boolean_exp", "expression": "> 0"}]}

    prop = FloatProperty.model_validate(prop_dict)
    assert len(prop.validators) == 1
    assert prop.validators[0].expression == "> 0"

    # Test validation works
    validator = prop.validators[0]
    assert validator.validate_property(5.0)[0] is True
    assert validator.validate_property(-1.0)[0] is False


def test_boolean_validator_real_estate_examples():
    """Test with actual real estate schema examples."""
    # Property name validation
    validator = BooleanExpValidator(expression="len() > 0")
    assert validator.validate_property("The Metropolitan Tower")[0] is True
    assert validator.validate_property("")[0] is False

    # Square footage validation
    validator = BooleanExpValidator(expression="> 0")
    assert validator.validate_property(125000.0)[0] is True
    assert validator.validate_property(-1)[0] is False

    # Units validation with null allowed
    validator = BooleanExpValidator(expression="> 0 or is null")
    assert validator.validate_property(150)[0] is True
    assert validator.validate_property(None)[0] is True
    assert validator.validate_property(0)[0] is False

    # Year built validation
    validator = BooleanExpValidator(expression="> 1800 and <= 2025")
    assert validator.validate_property(1995)[0] is True
    assert validator.validate_property(1750)[0] is False
    assert validator.validate_property(2026)[0] is False

    # Occupancy rate validation
    validator = BooleanExpValidator(expression=">= 0 and <= 100")
    assert validator.validate_property(92.5)[0] is True
    assert validator.validate_property(-1)[0] is False
    assert validator.validate_property(101)[0] is False
