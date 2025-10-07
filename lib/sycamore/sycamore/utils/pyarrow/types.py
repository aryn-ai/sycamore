from dateutil.parser import isoparser
from typing import Any, TYPE_CHECKING

from sycamore.data import Document
from sycamore.schema import DataType, make_property, NamedProperty, Property, SchemaV2

if TYPE_CHECKING:
    import pyarrow as pa


def named_property_to_pyarrow(np: NamedProperty) -> "pa.Field":
    import pyarrow as pa

    return pa.field(name=np.name, type=property_to_pyarrow(np.type), nullable=True)


def property_to_pyarrow(property: Property) -> "pa.DataType":
    import pyarrow as pa

    match property.type:
        case DataType.BOOL:
            return pa.bool_()
        case DataType.INT:
            return pa.int64()
        case DataType.FLOAT:
            return pa.float64()
        case DataType.STRING:
            return pa.string()
        # TODO: There are lots of pyarrow date/time types. Not sure which is best here.
        case DataType.DATE:
            return pa.date32()
        case DataType.DATETIME:
            return pa.date64()
        case DataType.ARRAY:
            return pa.list_(property_to_pyarrow(property.item_type))
        case DataType.OBJECT:
            return pa.struct(fields=[named_property_to_pyarrow(np) for np in property.properties])
        case DataType.CHOICE:
            # TODO: Currently this takes the type of the first element in the list of choices.
            dt = DataType.from_python(property.choices[0])
            return property_to_pyarrow(make_property(type=dt))
        case DataType.CUSTOM:
            raise ValueError("Custom types not supported in pyarrow conversion")
        case _:
            raise ValueError(f"Unknown type {property.type}")


def schema_to_pyarrow(schema: SchemaV2) -> "pa.schema":
    import pyarrow as pa

    dt = property_to_pyarrow(schema.as_object_property())
    return pa.schema(fields=list(dt))


date_parser = isoparser()


def _build_array(data: list[Any], arrow_type: "pa.DataType") -> "pa.Array":
    """Recursively builds a PyArrow Array from a list of Python objects based on the
    provided Arrow data type.
    """
    import pyarrow as pa

    # This uses dateutil.isoparser to parse dates so that they support reduced
    # precision dates like YYYY and YYYY-MM. Theoretically this could/should
    # be handled at the extraction layer.
    if pa.types.is_date32(arrow_type):
        return pa.array([date_parser.parse_isodate(d) if isinstance(d, str) else d for d in data], type=arrow_type)
    elif pa.types.is_date64(arrow_type):
        return pa.array([date_parser.isoparse(d) if isinstance(d, str) else d for d in data], type=arrow_type)

    # Other scalar types
    if not pa.types.is_nested(arrow_type):
        return pa.array(data, type=arrow_type)

    if pa.types.is_struct(arrow_type):
        child_arrays = []
        for field in arrow_type:
            field_data = [(d.get(field.name) if isinstance(d, dict) else None) for d in data]
            child_array = _build_array(field_data, field.type)
            child_arrays.append(child_array)

        # A struct is considered null if the input element was not a dictionary.
        null_mask = pa.array([not isinstance(d, dict) for d in data], type=pa.bool_())
        res = pa.StructArray.from_arrays(child_arrays, type=arrow_type, mask=null_mask)

        return res

    if pa.types.is_list(arrow_type):
        offsets = [0]
        flattened_data = []
        validity = []

        for sublist in data:
            is_valid_list = isinstance(sublist, list)
            validity.append(is_valid_list)
            if is_valid_list:
                flattened_data.extend(sublist)
            # The offset increases by the length of the sublist. For null entries,
            # the length is 0, so the offset doesn't change.
            offsets.append(len(flattened_data))

        # Recursively build the array for the flattened child data.
        child_array = _build_array(flattened_data, arrow_type.value_type)

        null_mask = pa.array([not v for v in validity], type=pa.bool_())
        return pa.ListArray.from_arrays(offsets, child_array, mask=null_mask)

    raise TypeError(f"Unsupported Arrow type for conversion: {arrow_type}")


def docs_to_pyarrow(docs: list[Document], schema: "pa.Schema | SchemaV2", property_root: str = "entity") -> "pa.Table":
    """Converts the properties from a list of Documents to a PyArrow Table
    based on the specified schema.

    Args:
        docs: The list of documents to convert.
        schema: The Scyamore or PyArrow Schema defining the structure of the
           target table.
        property_root: The subtree of the properties dict to export. Defaults
           to "entity"

    Returns:
        A PyArrow Table containing the data from the list of dictionaries.
    """
    import pyarrow as pa

    columns = []

    if isinstance(schema, SchemaV2):
        pa_schema = schema_to_pyarrow(schema)
    else:
        pa_schema = schema

    for field in schema:
        column_data = [doc.properties.get(property_root, {}).get(field.name) for doc in docs]
        column_array = _build_array(column_data, field.type)
        columns.append(column_array)

    return pa.Table.from_arrays(columns, schema=pa_schema)
