from typing import TYPE_CHECKING

from sycamore.functions.tokenizer import Tokenizer
from sycamore.data import Element

if TYPE_CHECKING:
    from sycamore.docset import DocSet


def merge_elements(
    index: int,
    elements: list[Element],
    field: str,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> tuple[int, str, set]:
    """
    Processes document elements to generate combined text adhering to token limit.

    Args:
        index: index of element to start on
        elements (list): List of document elements.
        field (str): The field to extract values from elements.
        tokenizer: Tokenizer to split text into tokens.
        max_tokens (int): Maximum number of tokens allowed.

    Returns:
        tuple[int, str, set]:
            Updated index after processing,
            Combined text of processed elements,
            Set of indices of processed elements.

    """
    combined_text = ""
    window_indices = set()
    current_tokens = 0
    for element in elements[index:]:
        txt = element.field_to_value(field)
        if not txt:
            index += 1
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
        index += 1
    return index, combined_text, window_indices


def _init_total_usage() -> dict[str, int | float]:
    return {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "wall_latency": 0.0,
    }


def compute_llm_usage(ds: "DocSet") -> dict[str, dict[str, int | float]]:
    """
    Compute the llm usage from the most recent materialize node

    Args:
        ds: The docset to get usage from. If there are llm calls after the last materialize
            that usage will be ignored

    Returns:
        dict[str, dict[str, int | float]]:
            A mapping from model to completion_tokens, prompt_tokens, total_tokens, and wall_latency

    """
    from sycamore.materialize import Materialize
    from sycamore.plan_nodes import NodeTraverseOrder

    m = ds.plan.get_plan_nodes(Materialize, order=NodeTraverseOrder.BEFORE)[0]
    mds = m.load_metadata()
    llm_mds = [m for m in mds if "model" in m.metadata]

    total_usage_per_model = {}

    for m in llm_mds:
        model = m.metadata["model"]
        if model not in total_usage_per_model:
            total_usage_per_model[model] = _init_total_usage()
        total_usage_per_model[model]["completion_tokens"] += m.metadata["usage"]["completion_tokens"]
        total_usage_per_model[model]["prompt_tokens"] += m.metadata["usage"]["prompt_tokens"]
        total_usage_per_model[model]["total_tokens"] += m.metadata["usage"]["total_tokens"]
        total_usage_per_model[model]["wall_latency"] += m.metadata["wall_latency"].total_seconds()
    return total_usage_per_model
