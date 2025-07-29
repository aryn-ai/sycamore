import logging
from sycamore.data.document import Document
from sycamore.schema import Schema, SchemaField

_logger = logging.getLogger(__name__)


def intersection_of_fields(docs: list[Document]) -> Document:

    fake_doc = Document()  # to store the schema

    if not docs:
        _logger.warning("Empty list of documents provided, returning empty schema.")
        fake_doc.properties["_schema"] = Schema(fields=[])
        return fake_doc

    if len(docs) == 1:
        fake_doc.properties["_schema"] = docs[0].properties.get("_schema", Schema(fields=[]))
        return fake_doc

    # Merge fields from all results by taking an intersection of field names from each item in results
    merged_fields = {}
    field_names = []
    for doc in docs:
        temp = []
        for field in doc.properties.get("_schema", Schema(fields=[])).fields:
            name = field.name
            temp.append(name)
            if name not in merged_fields:
                merged_fields[name] = {
                    "type": field.field_type,
                    "description": field.description,
                    "examples": field.examples,
                }
            else:
                if field.field_type != merged_fields[name]["type"]:
                    continue
                merged_fields[name]["examples"].extend(field["examples"])
        if temp:
            field_names.append(temp)
    common_field_names = set.intersection(*map(set, field_names))

    schema = Schema(
        fields=[
            SchemaField(
                name=name,
                field_type=merged_fields[name]["type"],
                description=merged_fields[name]["description"],
                examples=list(set(merged_fields[name]["examples"]))[:5],
            )
            for name in common_field_names
        ]
    )
    fake_doc.properties["_schema"] = schema

    return fake_doc
