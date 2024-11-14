import logging
from abc import ABC
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)


class SimplePrompt(ABC):
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


class EntityExtractorZeroShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful entity extractor"
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {entity}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


class EntityExtractorFewShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful entity extractor."
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements. Here are
    some example groups of text elements where the {entity} has been identified.
    {examples}
    Using the context from the document and the provided examples, FIND, COPY, and RETURN the {entity}. Only return the {entity} as part
    of your answer. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


class TextSummarizerGuidancePrompt(SimplePrompt):
    system = "You are a helpful text summarizer."
    user = """Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up answer. Only return the summary as part of your answer.
    {query}
    """
    var_name = "summary"


class SchemaZeroShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful entity extractor. You only return JSON Schema."
    user = """You are given a few text elements of a document. Extract JSON Schema representing one entity of
    class {entity} from the document. Using this context, FIND, FORMAT, and RETURN the JSON-LD Schema.
    Return a flat schema, without nested properties. Return at most {max_num_properties} properties.
    Only return JSON Schema as part of your answer.
    {query}
    """


class PropertiesZeroShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful property extractor. You only return JSON."
    user = """You are given a few text elements of a document. Extract JSON representing one entity of
    class {entity} from the document. The class only has properties {properties}. Using
    this context, FIND, FORMAT, and RETURN the JSON representing one {entity}.
    Only return JSON as part of your answer. If no entity is in the text, return "None".
    {query}
    """


class TaskIdentifierZeroShotGuidancePrompt(SimplePrompt):
    system = "You are a helpful task identifier. You return a string containing no whitespace."
    user = """You are given a dictionary where the keys are task IDs and the values are descriptions of tasks.
    Using this context, FIND and RETURN only the task ID that best matches the given question.
    Only return the task ID as a string. Do not return any additional information.
    {task_descriptions}
    Question: {question}
    """


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


class ExtractTablePropertiesPrompt(SimplePrompt):
    user = """
        You are given a text string represented as a CSV (comma-separated values) and an image of a table.

        Instructions:
            Check if the table contains key-value pairs. A key-value pair table is a table where data is structured as key-value pairs. Generally, the first column contains the key and the second column contains the value. However, key-value pairs can also appear in other formats.
            If there is a one-to-one mapping between two cells, even if the relationship is not direct, they should be considered key-value pairs.
            If the table is a key-value pair table, return its key-value pairs as a JSON object.
            If the table is not a key-value pair table, return False.
            Parse the table, check the image, and return a flattened JSON object representing the key-value pairs from the table. The extracted key-value pairs should be formatted as a JSON object.
            Do not return nested objects; keep the dictionary only one level deep. The only valid value types are numbers, strings, None, and lists.
            Use camelCase for the key names.
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
            """


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


class LlmFilterMessagesPrompt(SimplePrompt):
    def __init__(self, filter_question: str):
        super().__init__()

        self.system = "You are a helpful classifier that generously filters database entries based on questions."

        self.user = (
            "Given an entry and a question, you will answer the question relating "
            "to the entry. You only respond with 0, 1, 2, 3, 4, or 5 based on your "
            "confidence level. 0 is the most negative answer and 5 is the most positive "
            f"answer. Question: {filter_question}; Entry: "
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
    "ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT": EntityExtractorZeroShotGuidancePrompt,
    "ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT": EntityExtractorFewShotGuidancePrompt,
    "ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT": EntityExtractorFewShotGuidancePrompt,
    "ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT": EntityExtractorFewShotGuidancePrompt,
    "TEXT_SUMMARIZER_GUIDANCE_PROMPT": TextSummarizerGuidancePrompt,
    "TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT": TextSummarizerGuidancePrompt,
    "SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT": SchemaZeroShotGuidancePrompt,
    "SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT": SchemaZeroShotGuidancePrompt,
    "PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT": PropertiesZeroShotGuidancePrompt,
    "PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT_CHAT": PropertiesZeroShotGuidancePrompt,
}


def _deprecated_prompt(name: str) -> SimplePrompt:
    cls = _deprecated_prompts[name]
    logger.warn(f"The prompt {name} is deprecated. Switch to {cls.__name__}()")
    return cls()


def __getattr__(name: str):
    if name in _deprecated_prompts:
        return _deprecated_prompt(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
