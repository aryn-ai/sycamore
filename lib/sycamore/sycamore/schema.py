from typing import Optional, Any

from pydantic import BaseModel


class SchemaField(BaseModel):
    """Represents a field in a DocSet schema."""

    name: str

    field_type: str
    """The type of the field."""

    default: Optional[Any] = None

    description: Optional[str] = None
    """A natural language description of the field."""

    examples: Optional[list[Any]] = None
    """A list of example values for the field."""


class Schema(BaseModel):
    """Represents the schema of a DocSet."""

    fields: list[SchemaField]
    """A list of fields belong to this schema."""
