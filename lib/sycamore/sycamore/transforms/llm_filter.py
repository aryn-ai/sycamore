import copy
import re
from typing import Callable, Optional

from sycamore.data import Document, Element
from sycamore.functions.tokenizer import Tokenizer
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import ElementListIterPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.transforms.base_llm import LLMMap
from sycamore.transforms.basics import Filter
from sycamore.transforms.extract_entity import EntityExtractor
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
    prompt: ElementListIterPrompt,
    threshold: int = 3,
    keep_none: bool = False,
    use_elements: bool = False,
    similarity_query: Optional[str] = None,
    similarity_scorer: Optional[SimilarityScorer] = None,
    max_tokens: int = 512,
    tokenizer: Optional[Tokenizer] = None,
    **kwargs,
) -> Node:

    source_idx_key = f"{new_field}_source_element_indices"
    iteration_var_name = f"{new_field}_i"
    if not use_elements:

        def eb(elts: list[Element]) -> list[list[Element]]:
            source_indices = {e.element_index for e in elts}
            for e in elts:
                e.properties[source_idx_key] = source_indices
            return [elts]

    elif tokenizer is None:

        def eb(elts: list[Element]) -> list[list[Element]]:
            for e in elts:
                e.properties[source_idx_key] = {e.element_index}
            return [[e] for e in elts]

    else:

        def eb(elts: list[Element]) -> list[list[Element]]:
            curr_tks = 0
            curr_batch: list[Element] = []
            batches = []
            source_indices = set()
            assert tokenizer is not None, "Cannot batch elements based on token counts because tokenizer is None"
            for e in elts:
                eltl = prompt.element_list_constructor([e])
                tks = len(tokenizer.tokenize(eltl))
                if tks + curr_tks > max_tokens:
                    batches.append(curr_batch)
                    curr_tks = tks
                    curr_batch = [e]
                    source_indices = {e.element_index}
                    e.properties[source_idx_key] = source_indices
                else:
                    e.properties[source_idx_key] = source_indices
                    source_indices.add(e.element_index)
                    curr_batch.append(e)
                    curr_tks += tks
            batches.append(curr_batch)
            return batches

    prompt = prompt.set(element_batcher=eb, iteration_var_name=iteration_var_name)  # type: ignore

    def llmmap_validate(doc: Document) -> bool:
        if keep_none and len(doc.elements) == 0:
            return True
        try:
            score = int(re.findall(r"\d+", doc.properties.get(new_field, ""))[0])
        except IndexError:
            return False
        return score >= threshold

    def postprocess(doc: Document) -> Document:
        last_eclub: set[int] = set()
        club_idx = 0
        target_club_idx = doc.properties[iteration_var_name]
        for e in doc.elements:
            if len(last_eclub) > 0 and e.properties[source_idx_key] != last_eclub:
                club_idx += 1
            last_eclub = e.properties[source_idx_key]
            if club_idx == target_club_idx:
                doc.properties[source_idx_key] = last_eclub
                break
        return doc

    def filter_fn(doc: Document) -> bool:
        try:
            score = int(re.findall(r"\d+", doc.properties.get(new_field, ""))[0])
            doc.properties[new_field] = score
        except IndexError:
            return keep_none
        return score >= threshold

    llm_map = LLMMap(
        child=child,
        prompt=prompt,
        output_field=new_field,
        llm=llm,
        iteration_var=iteration_var_name,
        validate=llmmap_validate,
        **kwargs,
    )
    pp_map = Map(child=llm_map, f=postprocess)
    filter = Filter(child=pp_map, f=filter_fn)

    return filter
