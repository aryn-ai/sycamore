from abc import ABC, abstractmethod
import datetime
from enum import Enum
import re
import json
import logging
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
    model_serializer,
    model_validator,
    SerializerFunctionWrapHandler,
)


logger = logging.getLogger(__name__)


class SchemaField(BaseModel):
    """Represents a field in a DocSet schema."""

    name: str
    """The name of the field."""

    field_type: str = Field(validation_alias=AliasChoices("field_type", "property_type"))
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

    @classmethod
    def from_python(cls, python_val: Any) -> "DataType":
        return cls.from_python_type(type(python_val))

    @classmethod
    def from_python_type(cls, python_type: type) -> "DataType":
        """Convert a Python type to a DataType."""
        if python_type is bool:
            return cls.BOOL
        elif python_type is int:
            return cls.INT
        elif python_type is float:
            return cls.FLOAT
        elif python_type is str:
            return cls.STRING
        elif python_type is datetime.date:
            return cls.DATE
        elif python_type is datetime.datetime:
            return cls.DATETIME
        elif issubclass(python_type, list):
            return cls.ARRAY
        elif issubclass(python_type, dict):
            return cls.OBJECT
        else:
            logger.warning(f"Unsupported Python type: {python_type}. Defaulting to string.")
            return cls.STRING

    @classmethod
    def _missing_(cls, value: object) -> "DataType":
        """Handle missing values by returning a default DataType."""

        if isinstance(value, cls):
            return value
        elif not isinstance(value, str):
            raise ValueError(f"Invalid DataType value: {value}. Expected a string.")
        v = value.lower()

        # Handle common type names that are not in the enum
        if v in {"str", "text"}:
            return cls.STRING
        elif v in {"integer"}:
            return cls.INT
        elif v in {"list"}:
            return cls.ARRAY
        elif v in {"struct"}:
            return cls.OBJECT
        else:
            for member in cls:
                if member.value == v:
                    return member

        raise ValueError(f"Invalid DataType value: {value}. Valid values are: {', '.join(cls.values())}")


class PropertyValidator(BaseModel, ABC):
    """Represents a validator for a field in a DocSet schema."""

    type: Literal[str]  # type: ignore
    allowable_types: set[DataType]

    n_retries: int = Field(default=0, ge=0)

    @abstractmethod
    def constraint_string(self) -> str:
        pass

    @abstractmethod
    def validate_property(self, propval: Any) -> tuple[bool, Any]:
        pass


class RegexValidator(PropertyValidator):
    """Validates a field in a DocSet schema by comparing against a regex"""

    type: Literal["regex"] = "regex"
    allowable_types: set[DataType] = {DataType.STRING}

    regex: str
    _compiled_regex: Optional[re.Pattern] = None

    @model_validator(mode="after")
    def compile_regex(self) -> "RegexValidator":
        self._compiled_regex = re.compile(self.regex)
        return self

    def constraint_string(self) -> str:
        return f"must match the regex: `{self.regex}`"

    def validate_property(self, propval: Any) -> tuple[bool, Any]:
        import re

        if not isinstance(propval, str):
            return False, propval
        if self._compiled_regex is None:
            self._compiled_regex = re.compile(self.regex)
        s = re.fullmatch(self._compiled_regex, propval)
        return (s is not None), propval


ValidatorType: TypeAlias = Annotated[
    (RegexValidator),
    Field(discriminator="type"),
]


class SourceSpec(BaseModel):
    pass


class Property(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    # Not clear how to declare the type for this without explicitly
    # enumerating all the values of DataType.
    type: Literal[str]  # type: ignore
    """The type of the property."""

    required: bool = False
    """Whether the property is required."""

    description: Optional[str] = None
    """A brief description of the property."""

    default: Optional[Any] = None
    """The default value for the property."""

    extraction_instructions: Optional[str] = None
    """Additional instructions (prompts) to use when extracting the property."""

    examples: Optional[list[Any]] = None
    """Example values for this property."""

    source: Optional[SourceSpec] = None
    """Where to look for the field in the document.

    Defaults to the entire document.
    """

    validators: list[ValidatorType] = []
    """Validators to apply to this property."""

    @model_validator(mode="after")
    def check_validator_types(self) -> "Property":
        for v in self.validators:
            if self.type not in v.allowable_types:
                raise ValueError(f"{v.type} is not a valid validator for {self.type} property")
        return self


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

    # The default value here is to improve backward compatibility with
    # existing schemas. In a few places the old schema had type "array"
    # without a clear child type. This defaults to string to avoid breaking at
    # schema deserialization, though a more specific type should always be
    # used if available.
    item_type: "PropertyType" = StringProperty()


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


def _validate_with_type_alias(v: Any, handler: ValidatorFunctionWrapHandler) -> "PropertyType":
    if isinstance(v, dict) and "type" in v:
        v["type"] = DataType(v["type"])
    return handler(v)


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
    WrapValidator(_validate_with_type_alias),
]


def make_property(**kwargs) -> PropertyType:
    return TypeAdapter(PropertyType).validate_python(kwargs)


def make_named_property(name: str, **kwargs) -> NamedProperty:
    """Create a NamedProperty with the given name and property type."""
    return NamedProperty(name=name, type=make_property(**kwargs))


def _convert_to_named_property(schema_prop: SchemaField) -> NamedProperty:
    """Convert a SchemaProperty to a NamedProperty."""

    prop_type_dict = {
        "default": schema_prop.default,
        "description": schema_prop.description,
        "examples": schema_prop.examples,
    }

    if (declared_type := schema_prop.field_type) not in DataType.values():
        prop_type_dict["custom_type"] = declared_type
        prop_type_dict["type"] = DataType.CUSTOM
    else:
        prop_type_dict["type"] = DataType(schema_prop.field_type)

    return NamedProperty(
        name=schema_prop.name,
        type=make_property(**prop_type_dict),
    )


def _validate_new_schema(v: Any, handler: ValidatorFunctionWrapHandler) -> NamedProperty:
    try:
        return handler(v)
    except ValidationError as e:
        if any("valid validator" in ed["msg"] for ed in e.errors()):
            raise
        # Attempt to validate as a SchemaProperty and convert to NamedProperty
        schema_prop = SchemaField.model_validate(v)
        return _convert_to_named_property(schema_prop)


# @experimental
class SchemaV2(BaseModel):
    """Represents the schema of a DocSet."""

    properties: list[Annotated[NamedProperty, WrapValidator(_validate_new_schema)]] = Field(
        description="A list of properties belonging to this schema.",
        validation_alias=AliasChoices("properties", "fields"),
    )

    @property
    def fields(self) -> list[NamedProperty]:
        """Alias for properties."""
        return self.properties

    def flatten(self) -> "SchemaV2":
        """Flatten the schema by removing nested properties."""

        def flatten_object(prefix: str, props: list[NamedProperty], out_props: list[NamedProperty]) -> None:
            """Flatten an ObjectProperty into its properties."""
            for p in props:
                if p.type.type == DataType.ARRAY:
                    continue
                elif p.type.type == DataType.OBJECT:
                    # Flatten nested object properties
                    new_prefix = f"{prefix}.{p.name}" if prefix else p.name
                    flatten_object(new_prefix, p.type.properties, out_props)
                else:
                    new_p = p.model_copy(deep=True)
                    if len(prefix) > 0:
                        new_p.name = f"{prefix}.{p.name}"
                    out_props.append(new_p)

        flattened_properties: list[NamedProperty] = []
        flatten_object("", self.properties, flattened_properties)
        return SchemaV2(properties=flattened_properties)

    def render_flattened(self) -> str:
        flattened = self.flatten()
        props = [
            {"name": p.name, **p.type.model_dump(exclude_unset=True, exclude_none=True)} for p in flattened.properties
        ]
        return json.dumps({"properties": props}, indent=2)

    @model_serializer(mode="wrap")
    def serialize_backwards_compatible(self, nxt: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize the schema in a backwards-compatible format.

        This is designed to handle scenarios where client code may still
        expect the old schema format. If the schema does not use any new
        functionality, it will serialize to the same format as the previous
        version. This adds some overhead to both serialization and
        deserialization, but simplifies the upgrade path.

        """
        fields = []
        for p in self.properties:
            if p.type.type in {DataType.OBJECT, DataType.ARRAY, DataType.CHOICE}:
                return nxt(self)

            set_fields = p.type.model_fields_set
            if not set_fields.issubset({"type", "custom_type", "default", "description", "examples"}):
                return nxt(self)

            fields.append(
                {
                    "name": p.name,
                    "property_type": p.type.custom_type if p.type.type == DataType.CUSTOM else p.type.type.value,
                    "default": p.type.default,
                    "description": p.type.description,
                    "examples": p.type.examples,
                }
            )

        return {"properties": fields}
