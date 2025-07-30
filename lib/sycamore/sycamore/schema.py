from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypeAlias
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    ValidationError,
)


class SchemaField(BaseModel):
    """Represents a field in a DocSet schema."""

    name: str
    """The name of the field."""

    field_type: str
    """The type of the field."""

    default: Optional[Any] = None
    """The default value for the field."""

    description: Optional[str] = None
    """A natural language description of the field."""

    examples: Optional[list[Any]] = None
    """A list of example values for the field."""


class Schema(BaseModel):
    """Represents the schema of a DocSet."""

    fields: list[SchemaField]
    """A list of fields belong to this schema."""


class DataType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    CUSTOM = "custom"
    CHOICE = "choice"
    OBJECT = "object"

    @classmethod
    def values(cls):
        return set(map(lambda c: c.value, cls))


class PropertyValidator(BaseModel):
    """Represents a validator for a field in a DocSet schema."""

    pass


class SourceSpec(BaseModel):
    pass


class Property(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    # Not clear how to declare the type for this without explicitly
    # enumerating all the values of DataType.
    type: Literal[str]  # type: ignore
    """Type type of the property."""

    required: bool = False
    """Whether the property is required."""

    description: Optional[str] = None
    """A brief description of the property."""

    extraction_instructions: Optional[str] = None
    """Additional instructions (prompts) to use when extracting the property."""

    examples: Optional[list[Any]] = None
    """Example values for this property."""

    source: Optional[SourceSpec] = None
    """Where to look for the field in the document.
    
    Defaults to the entire document. 
    """

    validators: list[PropertyValidator] = []
    """Validators to apply to this property."""


class BoolProperty(Property):
    type: Literal[DataType.BOOL] = DataType.BOOL


class IntProperty(Property):
    type: Literal[DataType.INT] = DataType.INT


class FloatProperty(Property):
    type: Literal[DataType.FLOAT] = DataType.FLOAT


class StringProperty(Property):
    type: Literal[DataType.STRING] = DataType.STRING


class DateProperty(Property):
    type: Literal[DataType.DATE] = DataType.DATE

    format: str = "YYYY-MM-DD"


class DateTimeProperty(Property):
    type: Literal[DataType.DATETIME] = DataType.DATETIME

    # TODO: Do we care about microseconds or timezones here? Hoping to use datetime.isoformat().
    format: str = "YYYY-MM-DDTHH:MM:SS"


class ArrayProperty(Property):
    type: Literal[DataType.ARRAY] = DataType.ARRAY

    item_type: "PropertyType"


class ChoiceProperty(Property):
    type: Literal[DataType.CHOICE] = DataType.CHOICE

    choices: list[Any]


class CustomProperty(Property):
    type: Literal[DataType.CUSTOM] = DataType.CUSTOM

    custom_type: str


class NamedProperty(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """The name of the property."""

    type: "PropertyType"
    """The type of the property."""


class ObjectProperty(Property):
    type: Literal[DataType.OBJECT] = DataType.OBJECT
    properties: list[NamedProperty]


PropertyType: TypeAlias = Annotated[
    (
        BoolProperty
        | IntProperty
        | FloatProperty
        | StringProperty
        | DateProperty
        | DateTimeProperty
        | ArrayProperty
        | ChoiceProperty
        | CustomProperty
        | ObjectProperty
    ),
    Field(discriminator="type"),
]


def _convert_to_named_property(schema_prop: SchemaField) -> NamedProperty:
    """Convert a SchemaProperty to a NamedProperty."""

    prop_type_dict = {
        "type": schema_prop.field_type,
        "default": schema_prop.default,
        "description": schema_prop.description,
        "examples": schema_prop.examples,
    }

    if (declared_type := prop_type_dict["type"]) not in DataType.values():
        prop_type_dict["custom_type"] = declared_type
        prop_type_dict["type"] = DataType.CUSTOM

    prop_type: PropertyType = TypeAdapter(PropertyType).validate_python(prop_type_dict)

    return NamedProperty(
        name=schema_prop.name,
        type=prop_type,
    )


def _validate_new_schema(v: Any, handler: ValidatorFunctionWrapHandler) -> NamedProperty:
    try:
        return handler(v)
    except ValidationError:
        # Attempt to validate as a SchemaProperty and convert to NamedProperty
        schema_prop = SchemaField.model_validate(v)
        return _convert_to_named_property(schema_prop)


# @experimental
class SchemaV2(BaseModel):
    """Represents the schema of a DocSet."""

    properties: list[Annotated[NamedProperty, WrapValidator(_validate_new_schema)]] = Field(
        description="A list of properties belong to this schema.",
        validation_alias=AliasChoices("properties", "fields"),
    )

    @property
    def fields(self) -> list[NamedProperty]:
        """Alias for properties."""
        return self.properties
