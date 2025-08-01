import logging
from sycamore.data.document import Document
from sycamore.schema import SchemaV2
from sycamore.transforms.property_extraction.utils import create_named_property

_logger = logging.getLogger(__name__)


def intersection_of_fields(docs: list[Document]) -> Document:

    fake_doc = Document()  # to store the schema

    if not docs:
        _logger.warning("Empty list of documents provided, returning empty schema.")
        fake_doc.properties["_schema"] = SchemaV2(properties=[])
        return fake_doc

    if len(docs) == 1:
        fake_doc.properties["_schema"] = docs[0].properties.get("_schema", SchemaV2(properties=[]))
        return fake_doc

    # Merge fields from all results by taking an intersection of field names from each item in results
    merged_fields = {}
    field_names = []
    for doc in docs:
        temp = set()
        for property in doc.properties.get("_schema", SchemaV2(properties=[])).properties:
            name = property.name
            temp.add(name)
            if name not in merged_fields:
                merged_fields[name] = {
                    "name": name,
                    "type": property.type.type.value,
                    "description": property.type.description,
                    "examples": property.type.examples,
                }
            else:
                if property.type.type.value != merged_fields[name]["type"]:
                    continue
                merged_fields[name]["examples"].extend(property.type.examples)
        if temp:
            field_names.append(temp)

    if not field_names:
        _logger.warning("No fields found in any document, returning empty schema.")
        fake_doc.properties["_schema"] = SchemaV2(properties=[])
        return fake_doc

    common_field_names = set.intersection(*field_names)
    if not common_field_names:
        _logger.warning("No common fields found across documents, returning empty schema.")
        fake_doc.properties["_schema"] = SchemaV2(properties=[])
        return fake_doc

    schema = SchemaV2(
        properties=[create_named_property(merged_fields[name], n_examples=5) for name in common_field_names]
    )
    fake_doc.properties["_schema"] = schema

    return fake_doc
