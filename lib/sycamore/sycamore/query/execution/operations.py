import math
from typing import Any, List, Union, Optional

import structlog

from sycamore import DocSet
from sycamore.context import context_params, Context
from sycamore.data import MetadataDocument
from sycamore.functions import CharacterTokenizer, Tokenizer
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataMessagesPrompt,
)
from sycamore.transforms.summarize import (
    NUM_TEXT_CHARS_GENERATE,
    DocumentSummarizer,
    collapse,
    QuestionAnsweringSummarizer,
    BASE_PROPS,
)

log = structlog.get_logger(__name__)


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
    max_tokens: int = 120 * 1000,
    tokenizer: Tokenizer = CharacterTokenizer(),
    context: Optional[Context] = None,
    **kwargs,
) -> str:
    """
    Provides an English response to a question given relevant information. Uses a default maximum for 120k characters,
    that should loosely translate to 30k tokens (1 token ~= 4 chars).

    Args:
        llm: LLM to use for summarization.
        question: Question to answer.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        use_elements: Use text contents from document.elements instead of document.text_representation.
        num_elements: Number of elements whose text to use from each document.
        max_tokens: Maximum number of tokens allowed in the summary to send to the LLM.
        tokenizer: Tokenizer to use for counting against max_tokens.
        context: Optional Context object to get default parameters from.
        **kwargs: Additional keyword arguments.

    Returns:
        Conversational response to question.
    """
    text = _get_text_for_summarize_data(
        result_description=result_description,
        result_data=result_data,
        use_elements=use_elements,
        num_elements=num_elements,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        **kwargs,
    )
    messages = SummarizeDataMessagesPrompt(question=question, text=text).as_messages()
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    # LLM response
    return completion


def _get_text_for_summarize_data(
    result_description: str,
    result_data: List[Any],
    use_elements: bool,
    num_elements: int,
    max_tokens: Optional[int] = None,
    tokenizer: Optional[Tokenizer] = None,
    **kwargs,
) -> str:
    text = f"Data description: {result_description}\n"
    if (max_tokens is not None and tokenizer is None) or (max_tokens is None and tokenizer is not None):
        raise ValueError("Both max_tokens and tokenizer must be provided together.")

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        # consolidates relevant properties to give to LLM
        if isinstance(result, DocSet):
            done = False
            # For query result caching in the executor, we need to consume the documents
            # so that the materialized data is complete, even if they are not all included
            # in the input prompt to the LLM.
            for di, doc in enumerate(result.take_all()):
                if isinstance(doc, MetadataDocument):
                    continue
                if done:
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                doc_text = f"Document {di}:\n"
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

                if tokenizer is not None and max_tokens is not None:  # for mypy
                    total_token_count = len(tokenizer.tokenize(text + doc_text))
                    if total_token_count > max_tokens:
                        log.warn(
                            "Unable to add all text from to the LLM summary request due to token limit."
                            f" Sending text from {di + 1} docs."
                        )
                        done = True
                        continue
                text += doc_text + "\n"
        else:
            text += str(result) + "\n"

    return text


@context_params
def summarize_map_reduce(
    llm: LLM,
    question: str,
    result_description: str,
    result_data: List[Any],
    use_elements: bool = False,
    num_elements: int = 5,
    max_tokens: int = 10 * 1000,
    tokenizer: Tokenizer = CharacterTokenizer(),
) -> str:
    """ """
    text = f"Data description: {result_description}\n"
    for i, result in enumerate(result_data):
        if isinstance(result, DocSet):
            docs = (
                result.filter(lambda d: isinstance(d, MetadataDocument) is False)
                .summarize(
                    summarizer=DocumentSummarizer(llm, question)
                )  # document-level summarization can be parallelized (per DocSet)
                .take_all()
            )
            for doc in docs:
                text += doc.properties["summary"] + "\n"

        else:
            text += str(result) + "\n"

    final_summary = collapse(text, max_tokens, tokenizer, QuestionAnsweringSummarizer(llm, question))
    return final_summary
