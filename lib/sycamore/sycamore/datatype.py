import datetime
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


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
        elif v in {"boolean"}:
            return cls.BOOL
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
