import copy
import re
from typing import Callable, Optional

from sycamore.data import Document
from sycamore.functions.tokenizer import Tokenizer, CharacterTokenizer
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.base_llm import LLMMap
from sycamore.transforms.basics import Filter
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.extract_entity import EntityExtractor, OpenAIEntityExtractor
from sycamore.transforms.similarity import SimilarityScorer
from sycamore.utils.llm_utils import merge_elements


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
        ind, combined_text, window_indices = merge_elements(ind, doc.elements, field, tokenizer, max_tokens)
        dummy_element = copy.deepcopy(doc.elements[0])
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


def plan_llm_filter_as_llm_map(
    child: Node,
    llm: LLM,
    new_field: str,
    prompt: SycamorePrompt,
    field: str,
    threshold: int = 3,
    keep_none: bool = False,
    use_elements: bool = False,
    similarity_query: Optional[str] = None,
    similarity_scorer: Optional[SimilarityScorer] = None,
    max_tokens: int = 512,
    tokenizer: Optional[Tokenizer] = None,
    **kwargs,
) -> Node:
    if tokenizer is None:
        # create single-element batches
        tokenizer = CharacterTokenizer()
        max_tokens = 1
    if keep_none:
        prompt = prompt.fork(no_field_behavior="empty")
    else:
        prompt = prompt.fork(no_field_behavior="crash")
    entity_extractor = OpenAIEntityExtractor(
        entity_name=new_field,
        llm=llm,
        use_elements=use_elements,
        prompt=prompt,
        field=field,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        similarity_query=similarity_query,
        similarity_scorer=similarity_scorer,
    )
    # entity_extractor.as_llm_map returns a CompositeTransform
    # consisting of: (optionally a similarity scorer), a preprocess
    # map to figure out the batches if a tokenizer is around (+ sorting),
    # the llm map itself, and a postprocess map to assign the source_indices
    # that got the property. asserts are for mypy type coercion.
    comptransform = entity_extractor.as_llm_map(child)
    assert isinstance(comptransform, CompositeTransform)
    llm_map = comptransform.nodes[-2]
    assert isinstance(llm_map, LLMMap)

    def llmmap_validate_elements(doc: Document) -> bool:
        if keep_none and len(doc.elements) == 0:
            return True
        try:
            score = int(re.findall(r"\d+", doc.properties.get(new_field, ""))[-1])
        except IndexError:
            return False
        return score >= threshold

    def llmmap_validate_documents(doc: Document) -> bool:
        if keep_none:
            return True
        try:
            _ = int(re.findall(r"\d+", doc.properties.get(new_field, ""))[-1])
            return True
        except IndexError:
            return False

    if use_elements:
        llm_map._validate = llmmap_validate_elements
    else:
        llm_map._validate = llmmap_validate_documents

    def filter_fn(doc: Document) -> bool:
        try:
            score = int(re.findall(r"\d+", doc.properties.get(new_field, ""))[-1])
            doc.properties[new_field] = score
        except IndexError:
            return keep_none
        return score >= threshold

    filter = Filter(child=comptransform.nodes[-1], f=filter_fn)
    new_comptransform = CompositeTransform(child, [])
    new_comptransform.nodes = comptransform.nodes + [filter]
    return new_comptransform
