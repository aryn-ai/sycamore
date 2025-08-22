from typing import Callable
from collections import Counter
import logging
from sycamore.data.document import Document
from sycamore.schema import SchemaV2
from sycamore.transforms.property_extraction.utils import dedup_examples

_logger = logging.getLogger(__name__)


def _process_schema_fields(docs: list[Document], combine_fields_fn: Callable[[list[set[str]]], set[str]]) -> Document:
    """
    Common logic for processing schema fields from documents.

    Args:
        docs: List of documents with schemas to process
        combine_fields_fn: Function that determines how to combine field sets from different documents
                         (e.g., intersection or union)

    Returns:
        A document with the processed schema
    """

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
                merged_fields[name] = property
            else:
                if property.type.type != merged_fields[name].type.type:
                    continue  # Skip if types don't match
                if property.type.examples:
                    if merged_fields[name].type.examples is None:
                        merged_fields[name].type.examples = []
                    merged_fields[name].type.examples.extend(property.type.examples)
        if temp:
            field_names.append(temp)

    if not field_names:
        _logger.warning("No fields found in any document, returning empty schema.")
        fake_doc.properties["_schema"] = SchemaV2(properties=[])
        return fake_doc

    # Apply the combining function to determine which fields to keep
    common_field_names = combine_fields_fn(field_names)
    if not common_field_names:
        _logger.warning("No common fields found across documents, returning empty schema.")
        fake_doc.properties["_schema"] = SchemaV2(properties=[])
        return fake_doc

    schema = SchemaV2(properties=[merged_fields[name] for name in common_field_names])
    for prop in schema.properties:
        examples = dedup_examples(prop.type.examples or [])[:5]  # Limit to 5 examples
        prop.type.examples = examples if examples else None
    fake_doc.properties["_schema"] = schema

    return fake_doc


def intersection_of_fields(docs: list[Document]) -> Document:
    """
    Creates an intersection of schema fields from all documents.
    Only fields present in all documents are included.

    Args:
        docs: List of documents with _schema properties to merge

    Returns:
        A document with a merged schema containing only common fields
    """
    return _process_schema_fields(docs, lambda fields: set.intersection(*fields) if fields else set())


def union_of_fields(docs: list[Document]) -> Document:
    """
    Creates a union of schema fields from all documents.
    All unique fields from any document are included.

    Args:
        docs: List of documents with _schema properties to merge

    Returns:
        A document with a merged schema containing all unique fields
    """
    return _process_schema_fields(docs, lambda fields: set.union(*fields) if fields else set())


def make_freq_filter_fn(min_occurence_ratio: float = 0.5) -> Callable[[list[Document]], Document]:
    """
    Creates a frequency-based field filtering function.

    Args:
        min_occurence_ratio: Minimum ratio of documents that must contain a field (0.0-1.0)

    Returns:
        A function that filters fields based on their occurrence frequency
    """

    if not (0.0 <= min_occurence_ratio <= 1.0):
        raise ValueError("min_occurence_ratio must be between 0.0 and 1.0")

    def frequency_filtered_fields(docs: list[Document]) -> Document:
        """
        Creates a schema with fields filtered by occurrence frequency.
        Includes fields present in at least 50% of documents.

        This is a middle ground between intersection (all documents) and
        union (any document).

        Args:
            docs: List of documents with _schema properties to merge

        Returns:
            A document with a merged schema containing fields that appear in at least
            len(docs) * min_occurence_rate documents (default 50%).
        """

        def filter_by_frequency(fields: list[set[str]]) -> set[str]:
            if not fields:
                return set()

            # Calculate field frequencies
            field_count = Counter()  # type: ignore
            for field_set in fields:
                field_count.update(field_set)

            min_docs = max(1, int(len(fields) * min_occurence_ratio))

            # Include fields that meet the threshold
            return {field for field, count in field_count.items() if count >= min_docs}

        return _process_schema_fields(docs, filter_by_frequency)

    return frequency_filtered_fields
