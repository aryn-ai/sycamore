import math
from typing import Any, List, Union, Optional

from sycamore import DocSet
from sycamore.context import context_params, Context
from sycamore.data import MetadataDocument
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataMessagesPrompt,
)

BASE_PROPS = [
    "filename",
    "filetype",
    "page_number",
    "page_numbers",
    "links",
    "element_id",
    "parent_id",
    "_schema",
    "_schema_class",
    "entity",
]

NUM_DOCS_GENERATE = 60
NUM_TEXT_CHARS_GENERATE = 2500


def math_operation(val1: int, val2: int, operator: str) -> Union[int, float]:
    """
    Basic arithmetic operations on integers.

    Args:
        val1: First integer in operation.
        val2: Second integer in operation.
        operator: Type of operation; "add", "subtract", "divide", or "multiply"

    Returns:
        An integer or floating point number.
    """
    if operator == "add":
        return val1 + val2
    elif operator == "subtract":
        return val1 - val2
    elif operator == "divide":
        try:
            return val1 / val2
        except ZeroDivisionError:
            return math.nan
    elif operator == "multiply":
        return val1 * val2
    else:
        raise ValueError("Invalid math operator " + operator)


@context_params
def summarize_data(
    llm: LLM,
    question: str,
    result_description: str,
    result_data: List[Any],
    use_elements: bool = False,
    num_elements: int = 5,
    context: Optional[Context] = None,
    **kwargs,
) -> str:
    """
    Provides an English response to a question given relevant information.

    Args:
        llm: LLM to use for summarization.
        question: Question to answer.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        use_elements: use text contents from document.elements instead of document.text_representation.
        num_elements: number of elements whose text to use from each document.
        context: Optional Context object to get default parameters from.
        **kwargs

    Returns:
        Conversational response to question.
    """
    text = _get_text_for_summarize_data(
        result_description=result_description,
        result_data=result_data,
        use_elements=use_elements,
        num_elements=num_elements,
        **kwargs,
    )
    messages = SummarizeDataMessagesPrompt(question=question, text=text).as_messages()
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    # LLM response
    return completion


def _get_text_for_summarize_data(
    result_description: str, result_data: List[Any], use_elements: bool, num_elements: int, **kwargs
) -> str:
    text = f"Data description: {result_description}\n"

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        # consolidates relevant properties to give to LLM
        if isinstance(result, DocSet):
            for i, doc in enumerate(result.take(NUM_DOCS_GENERATE, **kwargs)):
                if isinstance(doc, MetadataDocument):
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                doc_text = f"Document {i}:\n"
                for k, v in props_dict.items():
                    doc_text += f"{k}: {v}\n"

                doc_text_representation = ""
                if not use_elements:
                    if doc.text_representation is not None:
                        doc_text_representation += doc.text_representation[:NUM_TEXT_CHARS_GENERATE]
                else:
                    for element in doc.elements[:num_elements]:
                        # Greedy fill doc level text length
                        if len(doc_text_representation) >= NUM_TEXT_CHARS_GENERATE:
                            break
                        doc_text_representation += (element.text_representation or "") + "\n"
                doc_text += f"Text contents:\n{doc_text_representation}\n"

                text += doc_text + "\n"
        else:
            text += str(result_data) + "\n"

    return text