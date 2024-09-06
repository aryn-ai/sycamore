import json
import math
from typing import Any, List, Union, Optional

from sycamore import DocSet
from sycamore.context import context_params, Context
from sycamore.data import MetadataDocument
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    SimplePrompt,
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
NUM_DOCS_PREVIEW = 10
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
    context: Optional[Context] = None,
    **kwargs,
) -> str:
    """
    Provides an English response to a question given relevant information.

    Args:
        client: LLM client.
        question: Question to answer.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        **kwargs

    Returns:
        Conversational response to question.
    """
    text = f"Description: {result_description}\n"

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        # consolidates relevant properties to give to LLM
        if isinstance(result, DocSet):
            for doc in result.take(NUM_DOCS_GENERATE, **kwargs):
                if isinstance(doc, MetadataDocument):
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                props_dict["text_representation"] = (
                    doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
                )

                text += json.dumps(props_dict, indent=2) + "\n"

        else:
            text += str(result_data) + "\n"

    messages = SummarizeDataMessagesPrompt(question=question, text=text).as_messages()
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    # LLM response
    return completion


class GenerateTablePrompt(SimplePrompt):
    def __init__(self, table_definition: str, text: str):
        super().__init__()

        self.system = "You generate JSON objects representing a table of database entries."

        self.user = (
            f"""Generate a JSON object representing a table that meets the following criteria: {table_definition}.
               The response should be in JSON format. It should contain a list of dictionaries,
               where each dictionary represents a row in the table. For example:
            """
            + """
            [
                { "column1": "value1", "column2": "value2" },
                { "column1": "value3", "column2": "value4" },
            ]
            """
            + f"""
            Here is the data to generate the table from:
            {text}
            """
        )


def generate_table(llm: LLM, table_definition: str, result_description: str, result_data: List[Any], **kwargs) -> str:
    """
    Generates a JSON table based on the input data provided.

    Args:
        client: LLM client.
        table_definition: Definition of the table to generate.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        **kwargs

    Returns:
        JSON list of dicts representing a table.
    """
    text = f"Description: {result_description}\n"

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        # consolidates relevant properties to give to LLM
        if isinstance(result, DocSet):
            for doc in result.take(NUM_DOCS_GENERATE, **kwargs):
                if isinstance(doc, MetadataDocument):
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                props_dict["text_representation"] = (
                    doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
                )

                text += json.dumps(props_dict, indent=2) + "\n"

        else:
            text += str(result_data) + "\n"

    messages = GenerateTablePrompt(table_definition=table_definition, text=text).as_messages()
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    # LLM response
    return completion


def generate_preview(result_description: str, result_data: List[Any], **kwargs) -> str:
    """
    Generates a JSON object representing a preview of a set of Documents.

    Args:
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        **kwargs

    Returns:
        JSON list of dicts representing a set of previews.
    """
    previews = {}

    for result in result_data:
        if isinstance(result, DocSet):
            for doc in result.take(NUM_DOCS_PREVIEW, **kwargs):
                if isinstance(doc, MetadataDocument):
                    continue

                # Remove duplicate parent docs.
                doc_id = doc.parent_id or doc.doc_id
                if doc_id in previews:
                    continue

                preview = {
                    "path": doc.properties.get("path"),
                    "title": doc.properties.get("title"),
                    "description": doc.properties.get("description"),
                    "snippet": (
                        doc.text_representation[:NUM_TEXT_CHARS_GENERATE]
                        if doc.text_representation is not None
                        else None
                    ),
                }
                previews[doc_id] = preview

    return json.dumps(list(previews.values()), indent=2)
