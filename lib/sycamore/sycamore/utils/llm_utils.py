from sycamore.data import Element
from sycamore.functions.tokenizer import Tokenizer


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
