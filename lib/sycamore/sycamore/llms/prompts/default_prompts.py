import logging
from abc import ABC
from typing import Any, Optional, Type
import textwrap

from sycamore.llms.prompts.prompts import (
    ElementListPrompt,
    ElementPrompt,
    StaticPrompt,
    JinjaPrompt,
    JinjaElementPrompt,
)
from sycamore.llms.prompts.jinja_fragments import (
    J_DYNAMIC_DOC_TEXT,
    J_FIELD_VALUE_MACRO,
    J_FORMAT_SCHEMA_MACRO,
    J_SET_ENTITY,
    J_SET_SCHEMA,
    J_ELEMENT_BATCHED_LIST,
    J_ELEMENT_LIST_CAPPED,
)

logger = logging.getLogger(__name__)


class _SimplePrompt(ABC):
    system: Optional[str] = None
    user: Optional[str] = None
    var_name: str = "answer"

    def as_messages(self, prompt_kwargs: Optional[dict[str, Any]] = None) -> list[dict]:
        messages = []
        if self.system is not None:
            system = self.system
            if prompt_kwargs is not None:
                system = self.system.format(**prompt_kwargs)
            messages.append({"role": "system", "content": system})
        if self.user is not None:
            user = self.user
            if prompt_kwargs is not None:
                user = self.user.format(**prompt_kwargs)
            messages.append({"role": "user", "content": user})
        return messages

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash((self.system, self.user, self.var_name))


SimplePrompt = _SimplePrompt


class _EntityExtractorZeroShotGuidancePrompt(_SimplePrompt):
    system = "You are a helpful entity extractor"
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {entity}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


EntityExtractorZeroShotJinjaPrompt = JinjaPrompt(
    system="You are a helpful entity extractor",
    user="""You are given a few text elements of a document. The {{ entity }} of the document is in these few text elements.
    Using this context, FIND, COPY, and RETURN the {{ entity }}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {% for elt in doc.elements[:num_elements] %} ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
    {% endfor %}""",
    field="text_representation",
    num_elements=35,
)


class _EntityExtractorFewShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful entity extractor."
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements. Here are
    some example groups of text elements where the {entity} has been identified.
    {examples}
    Using the context from the document and the provided examples, FIND, COPY, and RETURN the {entity}. Only return the {entity} as part
    of your answer. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


EntityExtractorFewShotJinjaPrompt = JinjaPrompt(
    system="You are a helpful entity extractor",
    user="""You are given a few text elements of a document. The {{ entity }} of the document is in these few text elements. Here are
    some example groups of text elements where the {{ entity }} has been identified.
    {{ examples }}
    Using the context from the document and the provided examples, FIND, COPY, and RETURN the {{ entity }}. Only return the {{ entity }} as part
    of your answer. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {% for elt in doc.elements[:num_elements] %} ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
    {% endfor %}""",
    field="text_representation",
    num_elements=35,
)

MetadataExtractorJinjaPrompt = JinjaPrompt(
    system="""You are a helpful property extractor.
        You generate JSON objects according to a schema
        to represent unstructured text data""",
    user="""You are given a series of elements from a document and each element contains a page number.
        Your task is to extract the {{ entity_name }} from the document and also record the page number where the property is found.
        The {{ entity_name }} follows the schema {{ schema }}.
        The schema includes some description and type hints to help you
        find them in the document. Make sure to not use comma for number. Do not output these hints.
        Return all the properties. If a property is not present in the document return null.
        Output ONLY JSON conforming to this schema, and nothing else, pass it as json between '```json {JSON}```.

        Text:
        {% for elt in doc.elements %}
        Page Number {{ elt['properties']['page_number'] }}: {{ elt.text_representation }}
        {% endfor %}

        Make sure to return the output tuple with square brackets.

        """,
)

SummarizeImagesJinjaPrompt = JinjaElementPrompt(
    user=textwrap.dedent(
        """
        You are given an image from a PDF document along with with some snippets of text preceding
        and following the image on the page. Based on this context, please decide whether the image is a
        graph or not. An image is a graph if it is a bar chart or a line graph. If the image is a graph,
        please summarize the axes, including their units, and provide a summary of the results in no more
        than 5 sentences.

        Return the results in the following JSON schema:

        {
            "is_graph": true,
            "x-axis": string,
            "y-axis": string,
            "summary": string
        }

        If the image is not a graph, please summarize the contents of the image in no more than five sentences
        in the following JSON format:

        {
            "is_graph": false,
            "summary": string
        }

        In all cases return only JSON and check your work.

        {% if include_context -%}
            {%- set posns = namespace(pos=-1) -%}
            {%- for e in doc.elements -%}
                {%- if e is sameas elt -%}
                    {%- set posns.pos = loop.index0 -%}
                    {% break %}
                {%- endif -%}
            {%- endfor -%}
            {%- if posns.pos > 0 -%}
                {%- set pe = doc.elements[posns.pos - 1] -%}
                {%- if pe.type in ["Section-header", "Caption", "Text"] -%}
        The text preceding the image is: {{ pe.text_representation }}
                {%- endif -%}
            {%- endif %}
            {% if posns.pos != -1 and posns.pos < doc.elements|count - 1 -%}
                {%- set fe = doc.elements[posns.pos + 1] -%}
                {%- if fe.type in ["Caption", "Text"] -%}
        The text following the image is: {{ fe.text_representation }}
                {%- endif -%}
            {%- endif -%}
        {%- endif -%}
        """
    ),
    include_image=True,
)


class _TextSummarizerGuidancePrompt(SimplePrompt):
    system = "You are a helpful text summarizer."
    user = """Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up answer. Only return the summary as part of your answer.
    {query}
    """
    var_name = "summary"


TextSummarizerGuidancePrompt = ElementPrompt(
    system="You are a helpful text summarizer.",
    user="""Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up your answer. Only return the summary as part of your answer
    {elt_text}
    """,
)

TextSummarizerJinjaPrompt = JinjaElementPrompt(
    system="You are a helpful text summarizer.",
    user=textwrap.dedent(
        """\
    Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up your answer. Only return the summary as part of your answer.

    {{ elt.text_representation }}
    """
    ),
)


SchemaZeroShotJinjaPrompt = JinjaPrompt(
    system="You are a helpful entity extractor. You only return JSON Schema.",
    user=textwrap.dedent(
        """\
        You are given a few text elements of a document. Extract JSON Schema representing
        one entity of class {{ entity }} from the document. Using this context, FIND, FORMAT, and
        RETURN the JSON-LD Schema. Return a flat schema, without nested properties. Return at most
        {{ max_num_properties }} properties. Only return JSON Schema as part of your answer.
        {% if prompt_formatter is defined %}{{ prompt_formatter(doc.elements[:num_elements]) }}{% else %}"""
    )
    + J_ELEMENT_LIST_CAPPED
    + "{% endif %}",
)


class _TaskIdentifierZeroShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful task identifier. You return a string containing no whitespace."
    user = """You are given a dictionary where the keys are task IDs and the values are descriptions of tasks.
    Using this context, FIND and RETURN only the task ID that best matches the given question.
    Only return the task ID as a string. Do not return any additional information.
    {task_descriptions}
    Question: {question}
    """


TaskIdentifierZeroShotGuidancePrompt = StaticPrompt(
    system="You are a helpful task identifier. You return a string containing no whitespace.",
    user="""You are given a dictionary where the keys are task IDs and the values are descriptions of tasks.
    Using this context, FIND and RETURN only the task ID that best matches the given question.
    Only return the task ID as a string. Do not return any additional information.
    {task_descriptions}
    Question: {question}
    """,
)


class GraphEntityExtractorPrompt(SimplePrompt):
    user = """
    -Instructions-
    You are a information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract entities from the text input that match the entity schemas provided. Each entity
    and property extracted should directly reference part of the text input provided.
    """


class GraphRelationshipExtractorPrompt(SimplePrompt):
    user = """
    -Goal-
    You are a helpful information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract relationships that map between entities that have already been extracted from this text.

    """


class _ExtractTablePropertiesPrompt(SimplePrompt):
    user = """
        You are given a text string represented as a CSV (comma-separated values) and an image of a table.

        Instructions:
            Check if the table contains key-value pairs. A key-value pair table is a table where data is structured as key-value pairs. Generally, the first column contains the key and the second column contains the value. However, key-value pairs can also appear in other formats.
            If there is a one-to-one mapping between two cells, even if the relationship is not direct, they should be considered key-value pairs.
            If the table is a key-value pair table, return its key-value pairs as a JSON object.
            If the table is not a key-value pair table, return False.
            Use camelCase for the key names in the JSON object.
            Parse the CSV table, check the image, and return a flattened JSON object representing the key-value pairs from the table. The extracted key-value pairs should be formatted as a JSON object.
            Do not return nested objects; keep the dictionary only one level deep. The only valid value types are numbers, strings, None, and lists.
            A table can have multiple or all null values for a key. In such cases, return a JSON object with the specified key set to null for all rows in the table.
            For fields where the values are in standard measurement units like miles, nautical miles, knots, or Celsius, include the unit in the key name and only set the numeric value as the value:

                "Wind Speed: 9 knots" should become "windSpeedInKnots": 9
                "Temperature: 3°C" should become "temperatureInC": 3
                Ensure that key names are enclosed in double quotes.

            Return only the JSON object between ``` if the table is a key-value pair table; otherwise, return False.

        example of a key-value pair table:
            |---------------------------------|------------------|
            | header 1                        | header 2         |
            |---------------------------------|------------------|
            | NEW FIRE ALARM SYSTEMS          | $272 TWO HOURS   |
            | NEW SPRINKLER SYSTEMS           | $408 THREE HOURS |
            | NEW GASEOUS SUPPRESSION SYSTEMS | $272 TWO HOURS   |
            |---------------------------------|------------------|

            return ```{"NEW FIRE ALARM SYSTEMS": "$272 TWO HOURS", "NEW SPRINKLER SYSTEMS": "$408 THREE HOURS", "NEW GASEOUS SUPPRESSION SYSTEMS": "$272 TWO HOURS"}```

        example of a table which is not key-value pair table:
            |---------------------------------|------------------|------------------|
            | header 1                        | header 2         | header 3         |
            |---------------------------------|------------------|------------------|
            | NEW FIRE ALARM SYSTEMS          | $272 TWO HOURS   | $2752 ONE HOUR   |
            | NEW SPRINKLER SYSTEMS           | $408 THREE HOURS | $128 FIVE HOURS  |
            | NEW GASEOUS SUPPRESSION SYSTEMS | $272 TWO HOURS   | $652 TEN HOURS   |
            |---------------------------------|------------------|------------------|

            return False

        example of a key value table containing null values
            |---------------------------------|---------------------|
            | header 1 :                      | header 2: 'value2'  |
            | header 3 :                      | header 4 :          |
            | header 5 :                      | header 6:           |
            |---------------------------------|---------------------|

            return ```{"header1": null, "header2": "value2", "header3": null, "header4": null, "header5": null, "header6": null}```

            """


ExtractTablePropertiesPrompt = ElementPrompt(
    user="""
        You are given a text string represented as a CSV (comma-separated values) and an image of a table.

        Instructions:
            Check if the table contains key-value pairs. A key-value pair table is a table where data is structured as key-value pairs. Generally, the first column contains the key and the second column contains the value. However, key-value pairs can also appear in other formats.
            If there is a one-to-one mapping between two cells, even if the relationship is not direct, they should be considered key-value pairs.
            If the table is a key-value pair table, return its key-value pairs as a JSON object.
            If the table is not a key-value pair table, return False.
            Use camelCase for the key names in the JSON object.
            Parse the CSV table, check the image, and return a flattened JSON object representing the key-value pairs from the table. The extracted key-value pairs should be formatted as a JSON object.
            Do not return nested objects; keep the dictionary only one level deep. The only valid value types are numbers, strings, None, and lists.
            A table can have multiple or all null values for a key. In such cases, return a JSON object with the specified key set to null for all rows in the table.
            For fields where the values are in standard measurement units like miles, nautical miles, knots, or Celsius, include the unit in the key name and only set the numeric value as the value:

                "Wind Speed: 9 knots" should become "windSpeedInKnots": 9
                "Temperature: 3°C" should become "temperatureInC": 3
                Ensure that key names are enclosed in double quotes.

            Return only the JSON object between ``` if the table is a key-value pair table; otherwise, return False.

        example of a key-value pair table:
            |---------------------------------|------------------|
            | header 1                        | header 2         |
            |---------------------------------|------------------|
            | NEW FIRE ALARM SYSTEMS          | $272 TWO HOURS   |
            | NEW SPRINKLER SYSTEMS           | $408 THREE HOURS |
            | NEW GASEOUS SUPPRESSION SYSTEMS | $272 TWO HOURS   |
            |---------------------------------|------------------|

            return ```{"NEW FIRE ALARM SYSTEMS": "$272 TWO HOURS", "NEW SPRINKLER SYSTEMS": "$408 THREE HOURS", "NEW GASEOUS SUPPRESSION SYSTEMS": "$272 TWO HOURS"}```

        example of a table which is not key-value pair table:
            |---------------------------------|------------------|------------------|
            | header 1                        | header 2         | header 3         |
            |---------------------------------|------------------|------------------|
            | NEW FIRE ALARM SYSTEMS          | $272 TWO HOURS   | $2752 ONE HOUR   |
            | NEW SPRINKLER SYSTEMS           | $408 THREE HOURS | $128 FIVE HOURS  |
            | NEW GASEOUS SUPPRESSION SYSTEMS | $272 TWO HOURS   | $652 TEN HOURS   |
            |---------------------------------|------------------|------------------|

            return False

        example of a key value table containing null values
            |---------------------------------|---------------------|
            | header 1 :                      | header 2: 'value2'  |
            | header 3 :                      | header 4 :          |
            | header 5 :                      | header 6:           |
            |---------------------------------|---------------------|

            return ```{"header1": null, "header2": "value2", "header3": null, "header4": null, "header5": null, "header6": null}```

        CSV:
            {elt_text}
            """,
    include_element_image=True,
)


PropertiesZeroShotGuidancePrompt = ElementListPrompt(
    system="You are a helpful property extractor. You only return JSON.",
    user=textwrap.dedent(
        """\
    You are given a few text elements of a document. Extract JSON representing one entity of
    class {entity} from the document. The class only has properties {properties}. Using
    this context, FIND, FORMAT, and RETURN the JSON representing one {entity}.
    Only return JSON as part of your answer. If no entity is in the text, return "None".
    {text}
    """
    ),
)


PropertiesZeroShotJinjaPrompt = JinjaPrompt(
    system="You are a helpful property extractor. You only return JSON.",
    user=J_SET_SCHEMA
    + J_SET_ENTITY
    + textwrap.dedent(
        """\
    You are given some text of a document. Extract JSON representing one entity of
    class {{ entity }} from the document. The class only has properties {{ schema }}. Using
    this context, FIND, FORMAT, and RETURN the JSON representing one {{ entity }}.
    Only return JSON as part of your answer. If no entity is in the text, return "None".

    Document:
    """
    )
    + J_DYNAMIC_DOC_TEXT,
)

PropertiesFromSchemaJinjaPrompt = JinjaPrompt(
    system=(
        "You are a helpful property extractor. You have to return your response as a JSON that"
        "can be parsed with json.loads(<response>) in Python. Do not return any other text."
    ),
    user=(
        J_FORMAT_SCHEMA_MACRO
        + """\
Extract values for the following fields:
{{ format_schema(schema) }}

Document text:"""
        + J_DYNAMIC_DOC_TEXT
        + """

Don't return extra information.
If you cannot find a value for a requested property, use the provided default or the value 'None'.
Return your answers as a valid json dictionary that will be parsed in python with json.loads(<response>).
"""
    ),
)


class EntityExtractorMessagesPrompt(SimplePrompt):
    def __init__(self, question: str, field: str, format: Optional[str], discrete: bool = False):
        super().__init__()
        self.system = (
            "You are a helpful entity extractor that creates a new field in a "
            "database based on the value of an existing field. "
        )

        if discrete:
            self.user = f"""Below, you will be given a database field value and a question.
            Your job is to provide the answer to the question based on the value provided.
            Your response should ONLY contain the answer to the question. If you are not able
            to extract the new field given the information, respond with "None". The type
            of your response should be "{format}".
            Field value: {field}\n
            Answer the question "{question}":"""
        else:
            self.user = (
                f"Include as much relevant detail as "
                "possible that is related to/could help answer this question. Respond in "
                "sentences, not just a single word or phrase."
                f"Question: {question} Use this existing related database field "
                f'"{field}" to answer the question: '
            )


# TODO: Need to separate the condition when use_elements is True but the field is a property that does not require chunking.
#       Could save up on time by not scheduling llm calls.
LlmFilterMessagesJinjaPrompt = JinjaPrompt(
    system="You are a helpful classifier that filters database entries based on questions.",
    user=(
        J_FIELD_VALUE_MACRO
        + textwrap.dedent(
            """\
        Given an entry and a yes or no question, you will answer the question relating
        to the entry. You only respond with 0, 1, 2, 3, 4, or 5 based on your confidence
        level. 0 is a confident 'no' and 5 is a confident 'yes'.
        Question: {{ filter_question }}
        Entry: {% if not use_elements -%}
        Field Name: {{ field }}; Field Value: {{ field_value(doc, field, no_field_behavior) }}
        {% else %}"""
        )
        + J_ELEMENT_BATCHED_LIST
        + "{% endif %}"
        + textwrap.dedent(
            """\
            The response should be a value from [0,1,2,3,4,5]. 0 is a confident 'no' and 5 is a confident 'yes'."""
        )
    ),
)


class SummarizeDataMessagesPrompt(SimplePrompt):
    def __init__(self, question: str, text: str):
        super().__init__()

        self.system = (
            "You are a helpful conversational English response generator for queries regarding database entries."
        )

        self.user = (
            "The following question and answer are in regards to database entries. "
            "Respond ONLY with a conversational English response WITH JUSTIFICATION to the question "
            f'"{question}" given the answer "{text}". Include as much detail/evidence as possible.'
        )


class LlmClusterEntityFormGroupsMessagesPrompt(SimplePrompt):
    def __init__(self, field: str, instruction: str, text: str):
        super().__init__()

        self.user = (
            f"You are given a list of values corresponding to the database field '{field}'. Categorize the "
            f"occurrences of '{field}' and create relevant non-overlapping groups. Return ONLY JSON with "
            f"the various categorized groups of '{field}' based on the following instructions '{instruction}'. "
            'Return your answer in the following JSON format and check your work: {{"groups": [string]}}. '
            'For example, if the instruction is "Form groups of different types of food" '
            'and the values are "banana, milk, yogurt, chocolate, oranges", you would return something like '
            "{{\"groups\": ['fruit', 'dairy', 'dessert', 'other']}}. Form groups to encompass as many entries "
            "as possible and don't create multiple groups with the same meaning. Here is the list values "
            f'values corresponding to "{field}": "{text}".'
        )


class LlmClusterEntityAssignGroupsMessagesPrompt(SimplePrompt):
    def __init__(self, field: str, groups: list[str]):
        super().__init__()

        self.user = (
            f"Categorize the database entry you are given corresponding to '{field}' into one of the "
            f'following groups: "{groups}". Perform your best work to assign the group. Return '
            f"ONLY the string corresponding to the selected group. Here is the database entry you will use: "
        )


_deprecated_prompts: dict[str, Type[SimplePrompt]] = {
    "ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT": _EntityExtractorZeroShotGuidancePrompt,
    "ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT": _EntityExtractorFewShotGuidancePrompt,
    "ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT": _EntityExtractorFewShotGuidancePrompt,
    "ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT": _EntityExtractorFewShotGuidancePrompt,
    "TEXT_SUMMARIZER_GUIDANCE_PROMPT": _TextSummarizerGuidancePrompt,
    "TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT": _TextSummarizerGuidancePrompt,
}


def _deprecated_prompt(name: str) -> SimplePrompt:
    cls = _deprecated_prompts[name]
    logger.warning(f"The prompt {name} is deprecated. Switch to {cls.__name__}()")
    return cls()


def __getattr__(name: str):
    if name in _deprecated_prompts:
        return _deprecated_prompt(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
