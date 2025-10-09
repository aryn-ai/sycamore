from datetime import date, datetime
from dateutil.parser import isoparser
from typing import Any, TYPE_CHECKING

from sycamore.data import Document
from sycamore.schema import DataType, NamedProperty, Property, SchemaV2

if TYPE_CHECKING:
    import pyarrow as pa

def named_property_to_pyarrow(np: NamedProperty) -> "pa.Field":
    import pyarrow as pa
    #print(f"nullable={not np.type.required}")
    #return pa.field(name=np.name, type=property_to_pyarrow(np.type), nullable=not np.type.required)
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
            # TODO: This is not quite right. We should be making a union type.
            return pa.string()
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
    """
    Recursively builds a PyArrow Array from a list of Python objects based on the
    provided Arrow data type.

    Args:
        data: A list of Python objects (scalars, dicts, lists, or None).
        arrow_type: The target PyArrow DataType for the array.

    Returns:
        A PyArrow Array corresponding to the input data and type.

    Raises:
        TypeError: If an unsupported Arrow type is encountered.
    """
    import pyarrow as pa

    #print(f"In _build_array {data=} {arrow_type=}")

    # TODO: Hack around date formats. Need a story around this. 
    if pa.types.is_date32(arrow_type):
        return pa.array([date_parser.parse_isodate(d) if isinstance(d, str) else d for d in data], type=arrow_type)
    elif pa.types.is_date64(arrow_type):
        return pa.array([date_parser.isoparse(d) if isinstance(d, str) else d for d in data], type=arrow_type)

    if not pa.types.is_nested(arrow_type):
        return pa.array(data, type=arrow_type)
    # Recursive Case: Handle Structs.
    if pa.types.is_struct(arrow_type):
        child_arrays = []
        # For each field in the struct, extract its data and build its array.
        for field in arrow_type:
            # Extract data for the current field from the list of dicts.
            # Handles cases where the parent dict is None or the key is missing.
            field_data = [(d.get(field.name) if isinstance(d, dict) else None) for d in data]
            # Recursively call _build_array for the child field.
            child_array = _build_array(field_data, field.type)
            child_arrays.append(child_array)

        # A struct is considered null if the input element was not a dictionary.
        validity_mask = pa.array([isinstance(d, dict) for d in data], type=pa.bool_())
        #print(f"In struct case {validity_mask=}")

        #res = pa.StructArray.from_arrays(child_arrays, fields=list(arrow_type), mask=validity_mask)
        res = pa.StructArray.from_arrays(child_arrays, fields=list(arrow_type))
        #print(f"{res=} {list(arrow_type)=}")
        
        return res

    # Recursive Case: Handle Lists.
    if pa.types.is_list(arrow_type):
        offsets = [0]
        flattened_data = []
        validity = []

        # Iterate through the list of lists to build offsets and flatten the data.
        for sublist in data:
            is_valid_list = isinstance(sublist, list)
            validity.append(is_valid_list)
            if is_valid_list:
                flattened_data.extend(sublist)
            # The offset increases by the length of the sublist. For null entries,
            # the length is 0, so the offset doesn't change.
            offsets.append(len(flattened_data))

        #print(f"In list case: {offsets=} {flattened_data=} {validity=} {arrow_type.value_type=}")
            
        # Recursively build the array for the flattened child data.
        child_array = _build_array(flattened_data, arrow_type.value_type)

        #print(f"After recurse {child_array=}")
        
        validity_mask = pa.array(validity, type=pa.bool_())


        #print(f"returning {offsets=} {child_array=} {validity_mask=}")
        #return pa.ListArray.from_arrays(offsets, child_array, mask=validity_mask)
        return pa.ListArray.from_arrays(offsets, child_array)

    raise TypeError(f"Unsupported Arrow type for conversion: {arrow_type}")


def docs_to_pyarrow(docs: list[Document], schema: "pa.Schema | SchemaV2") -> "pa.Table":
    """
    Converts a list of Python dictionaries to a PyArrow Table according to a
    specified schema.

    This function handles arbitrarily nested data containing structs and lists,
    and correctly processes missing keys or null values.

    Args:
        list_of_dicts: The list of dictionaries to convert.
        schema: The PyArrow Schema defining the structure of the target table.

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
        # Get python value for this field. 
        column_data = [doc.properties.get("entity", {}).get(field.name) for doc in docs]
        
        # Compute possibly nested pyarrow Array
        column_array = _build_array(column_data, field.type)

        columns.append(column_array)

    #print(f"{columns=}")
    
    return pa.Table.from_arrays(columns, schema=pa_schema)


def docs_to_pivoted_pyarrow(docs: list[Document], schema: SchemaV2) -> dict[str, "pa.Table"]:
    # TODO: Make this more efficient. Right now it requires multiple passes over the data.
    array_properties = {}
    array_docs = {}
    new_properties = []
    
    for prop in schema.properties:
        if prop.type.type == DataType.ARRAY and prop.type.type.item_type.type == DataType.OBJECT:
            # This is a candidate for pivoting.
            # TODO: Might be an issue here with duplicate names? Need to decide on a semantics for that. 
            array_properties[prop.name] = prop
        else:
            new_properties.append(prop)
            
    for d in docs:
        entity = d.properties.get("entity", {})
        for p in array_properties.keys():
            if p in entity:

                array_docs.setdefault(p, []).append()


                del entity[p]

            

# --- Example Usage ---
if __name__ == '__main__':
    # Define a complex, nested schema.
    schema = pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("user", pa.struct([
            pa.field("name", pa.string()),
            pa.field("age", pa.int32())
        ])),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("metrics", pa.list_(pa.struct([
            pa.field("key", pa.string()),
            pa.field("value", pa.float64())
        ])))
    ])

    # Create sample data with missing fields and null values.
    data = [
        {
            "id": 1,
            "user": {"name": "Alice", "age": 30},
            "tags": ["A", "B"],
            "metrics": [{"key": "m1", "value": 1.1}, {"key": "m2", "value": 2.2}]
        },
        {
            "id": 2,
            # "user" struct is completely missing.
            "tags": None,  # This list is null.
            "metrics": []  # This list is empty but not null.
        },
        {
            "id": 3,
            "user": {"name": "Charlie"},  # "age" is missing inside the struct.
            "tags": ["C"],
            # "metrics" list is missing.
        }
    ]

    # Convert the data to a PyArrow Table.
    try:
        arrow_table = convert_to_pyarrow(data, schema)
        
        print("--- Successfully created PyArrow Table ---")
        print(arrow_table)
        
        print("\n--- Table Schema ---")
        print(arrow_table.schema)

        print("\n--- Column 'user' (StructArray) ---")
        print(arrow_table.column('user'))

        print("\n--- Column 'metrics' (ListArray of StructArrays) ---")
        print(arrow_table.column('metrics'))

    except Exception as e:
        print(f"An error occurred: {e}")

