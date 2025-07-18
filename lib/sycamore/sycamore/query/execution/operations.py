import math
from typing import Any, List, Union, Optional

import structlog

from sycamore import DocSet
from sycamore.context import context_params, Context
from sycamore.data import Document
from sycamore.functions.tokenizer import OpenAITokenizer
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataMessagesPrompt,
)
from sycamore.transforms.summarize import (
    MultiStepDocumentSummarizer,
    OneStepDocumentSummarizer,
    Summarizer,
    SummaryDocument,
    EtCetera,
)

log = structlog.get_logger(__name__)

# making this takes 0.4s; see if we can delay
_DEFAULT_TOKENIZER = OpenAITokenizer("gpt-4o", max_tokens=128_000)
# multistep
_MULTISTEP_SUMMARIZE: tuple[type[Summarizer], dict[str, Any]] = (
    MultiStepDocumentSummarizer,
    {
        "fields": [EtCetera],
        "tokenizer": _DEFAULT_TOKENIZER,
        "llm_mode": LLMMode.ASYNC,
    },
)
# onestep
_ONESTEP_SUMMARIZE: tuple[type[Summarizer], dict[str, Any]] = (
    OneStepDocumentSummarizer,
    {
        "fields": [EtCetera],
        "tokenizer": _DEFAULT_TOKENIZER,
    },
)

DEFAULT_SUMMARIZE = _ONESTEP_SUMMARIZE


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
    question: Optional[str],
    data_description: str,
    input_data: List[Any],
    context: Optional[Context] = None,
    docset_summarizer: Optional[Summarizer] = None,
    **kwargs,
) -> str:
    """
    Provides an English response to a question given relevant information. Uses a default maximum for 120k characters,
    that should loosely translate to 30k tokens (1 token ~= 4 chars).

    Args:
        llm: LLM to use for summarization.
        question: Question to answer.
        data_description: Description of each of the inputs in input_data.
        input_data: List of inputs.
        context: Optional Context object to get default parameters from.
        docset_summarizer: Summarizer class to use to summarize the docset.
            Default is `DEFAULT_DOCSET_SUMMARIZER`
        summarizer_kwargs: keyword arguments to pass to the docset summarizer constructor. e.g.
            `tokenizer`, `token_limit`, and `element_batch_size`
        **kwargs: Additional keyword arguments.

    Returns:
        Conversational response to question.
    """
    if docset_summarizer is None:
        sum_class, ctor_kwargs = DEFAULT_SUMMARIZE
        docset_summarizer = sum_class(llm=llm, question=question, **ctor_kwargs)  # type: ignore

    if all(isinstance(d, DocSet) for d in input_data):
        docset_summaries = summarize_data_docsets(
            llm,
            question,
            input_data,
            docset_summarizer=docset_summarizer,
            data_description=data_description,
        )
        return "\n".join(docset_summaries)

    # LuNA pipelines can return list of integers, strings, or floats, depending on the pipeline.
    # While this should eventually be fixed, we handle it here by summarizing the information
    # differently in that case.
    # TODO: Jinjify.
    assert not any(
        isinstance(r, DocSet) for r in input_data
    ), f"Received heterogeneous input data (docsets and scalars) to summarize data: {input_data}"
    text = f"Data description: {data_description}\n"
    for i, d in enumerate(input_data):
        text += f"Input {i + 1}: {str(d)}\n"

    messages = SummarizeDataMessagesPrompt(question=question or "", text=text).as_messages()
    prompt = RenderedPrompt(messages=[RenderedMessage(role=m["role"], content=m["content"]) for m in messages])
    completion = llm.generate(prompt=prompt)
    return completion


def sum_to_text(d: Document) -> Document:
    if "summary" in d.properties:
        d.text_representation = d.properties.pop("summary")
    return d


def summarize_data_docsets(
    llm: LLM,
    question: Optional[str],
    input_data: List[DocSet],
    docset_summarizer: Summarizer,
    data_description: Optional[str] = None,
) -> list[str]:
    single_docs = [SummaryDocument(sub_docs=ds.take_all()) for ds in input_data]
    agged_ds = input_data[0].context.read.document(single_docs).summarize(docset_summarizer)
    texts = [d.properties["summary"] for d in agged_ds.take_all()]
    return texts
