import copy
import re
from typing import Callable

from sycamore.data import Document
from sycamore.functions.tokenizer import Tokenizer
from sycamore.transforms.extract_entity import EntityExtractor


def document_threshold_llm_filter(
    doc: Document,
    field: str,
    entity_extractor: EntityExtractor,
    threshold: int,
    keep_none: bool,
) -> bool:
    if doc.field_to_value(field) is None:
        return keep_none
    doc = entity_extractor.extract_entity(doc)
    # todo: move data extraction and validation to entity extractor
    try:
        return int(re.findall(r"\d+", doc.properties[entity_extractor.property()])[0]) >= threshold
    except IndexError:
        return False


def tokenized_threshold_llm_filter(
    doc: Document,
    field: str,
    entity_extractor: EntityExtractor,
    threshold: int,
    keep_none: bool,
    element_sorter: Callable[[Document], None],
    max_tokens: int,
    tokenizer: Tokenizer,
) -> bool:
    element_sorter(doc)
    evaluated_elements = 0

    new_field = entity_extractor.property()
    ind = 0
    while ind < len(doc.elements):
        combined_text = ""
        window_indices = set()
        current_tokens = 0
        for element in doc.elements[ind:]:
            txt = element.field_to_value(field)
            if not txt:
                ind += 1
                window_indices.add(element.element_index)
                continue
            element_tokens = len(tokenizer.tokenize(txt))
            if current_tokens + element_tokens > max_tokens and current_tokens != 0:
                break
            if "type" in element:
                combined_text += f"Element type: {element['type']}\n"
            if "page_number" in element["properties"]:
                combined_text += f"Page_number: {element['properties']['page_number']}\n"
            if "_element_index" in element["properties"]:
                combined_text += f"Element_index: {element['properties']['_element_index']}\n"
            combined_text += f"Text: {txt}\n"
            window_indices.add(element.element_index)
            current_tokens += element_tokens
            ind += 1
        dummy_element = copy.deepcopy(element)
        dummy_element[field] = combined_text
        e_doc = Document(dummy_element.data)
        e_doc = entity_extractor.extract_entity(e_doc)
        try:
            score = int(re.findall(r"\d+", e_doc.properties[new_field])[0])
        except IndexError:
            score = 0
        for i in range(0, len(window_indices)):
            doc.elements[ind - i - 1]["properties"][f"{new_field}"] = score
            doc.elements[ind - i - 1]["properties"][f"{new_field}_source_element_index"] = window_indices
        doc_source_field_name = f"{new_field}_source_element_index"
        if score >= doc.get(doc_source_field_name, 0):
            doc.properties[f"{new_field}"] = score
            doc.properties[f"{new_field}_source_element_index"] = window_indices
        if score >= threshold:
            return True
        evaluated_elements += 1

    if evaluated_elements == 0:  # no elements found for property
        return keep_none
    return False


def untokenized_threshold_llm_filter(
    doc: Document,
    field: str,
    entity_extractor: EntityExtractor,
    threshold: int,
    keep_none: bool,
    element_sorter: Callable[[Document], None],
) -> bool:
    element_sorter(doc)
    evaluated_elements = 0

    new_field = entity_extractor.property()

    for element in doc.elements:
        e_doc = Document(element.data)
        if e_doc.field_to_value(field) is None:
            continue
        e_doc = entity_extractor.extract_entity(e_doc)
        element.properties[new_field] = e_doc.properties[new_field]
        # todo: move data extraction and validation to entity extractor
        try:
            score = int(re.findall(r"\d+", element.properties[new_field])[0])
        except IndexError:
            continue
        # storing the element_index of the element that provides the highest match score for a document.
        doc_source_field_name = f"{new_field}_source_element_index"
        if score >= doc.get(doc_source_field_name, 0):
            doc.properties[f"{new_field}"] = score
            doc.properties[f"{new_field}_source_element_index"] = element.element_index
        if score >= threshold:
            return True
        evaluated_elements += 1

    if evaluated_elements == 0:  # no elements found for property
        return keep_none
    return False
