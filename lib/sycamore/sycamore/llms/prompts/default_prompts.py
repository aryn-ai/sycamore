from abc import ABC, abstractmethod
import logging
from typing import Optional, Type

from guidance.models import Chat, Instruct, Model as GuidanceModel
from guidance import gen, user, system, assistant, instruction

logger = logging.getLogger(__name__)


class GuidancePrompt(ABC):
    @abstractmethod
    def execute(self, model: GuidanceModel, **kwargs) -> str:
        pass

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False


class SimpleGuidancePrompt(GuidancePrompt):
    system: str = "You are a helpful assistant."
    user: str
    var_name: str = "answer"

    def execute(self, model: GuidanceModel, **kwargs) -> str:
        if isinstance(model, Chat):
            return self._execute_chat(model, **kwargs)
        elif isinstance(model, Instruct):
            return self._execute_instruct(model, **kwargs)
        else:
            return self._execute_completion(model, **kwargs)

    def _execute_chat(self, model, **kwargs) -> str:
        with system():
            lm = model + self.system.format(**kwargs)

        with user():
            lm += self.user.format(**kwargs)

        with assistant():
            lm += gen(name=self.var_name)

        return lm[self.var_name]

    def _execute_instruct(self, model, **kwargs) -> str:
        with instruction():
            lm = model + self.user.format(**kwargs)
        lm += gen(name=self.var_name)
        return lm[self.var_name]

    def _execute_completion(self, model, **kwargs) -> str:
        lm = model + self.user.format(**kwargs) + gen(name=self.var_name)
        return lm[self.var_name]

    def __hash__(self):
        return hash((self.system, self.user, self.var_name))


class EntityExtractorZeroShotGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful entity extractor"
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {entity}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


class EntityExtractorFewShotGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful entity extractor."
    # ruff: noqa: E501
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements. Here are
    some example groups of text elements where the {entity} has been identified.
    {examples}
    Using the context from the document and the provided examples, FIND, COPY, and RETURN the {entity}. Only return the {entity} as part
    of your answer. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {query}
    """


class TextSummarizerGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful text summarizer."
    user = """Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up answer. Only return the summary as part of your answer.
    {query}
    """
    var_name = "summary"


class SchemaZeroShotGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful entity extractor. You only return JSON Schema."
    user = """You are given a few text elements of a document. Extract JSON Schema representing one entity of
    class {entity} from the document. Using this context, FIND, FORMAT, and RETURN the JSON-LD Schema.
    Return a flat schema, without nested properties. Return at most {max_num_properties} properties.
    Only return JSON Schema as part of your answer.
    {query}
    """


class PropertiesZeroShotGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful property extractor. You only return JSON."
    user = """You are given a few text elements of a document. Extract JSON representing one entity of
    class {entity} from the document. The class only has properties {properties}. Using
    this context, FIND, FORMAT, and RETURN the JSON representing one {entity}.
    Only return JSON as part of your answer. If no entity is in the text, return "None".
    {query}
    """


class OpenAIMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class OpenAIMessagesPromptBase:
    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str) -> None:
        message = OpenAIMessage(role, content)
        self.messages.append(message)

    def get_messages_dict(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]


class EntityExtractorMessagesPrompt(OpenAIMessagesPromptBase):
    def __init__(self, question: str, field: str, format: Optional[str], discrete: bool = False):
        super().__init__()

        self.add_message(
            "system",
            (
                "You are a helpful entity extractor that creates a new field in a "
                "database from your response to a question on an existing field. "
            ),
        )

        if discrete:
            self.add_message(
                "user",
                (
                    f"The format of your response should be {format}. "
                    "Use standard convention to determine the style of your response. Do not include any abbreviations. "
                    "The following sentence should be valid: The answer to the "
                    'question based on the existing field is "your response". Your response should ONLY '
                    "contain the answer. If you are not able to extract the new field given the "
                    "information, respond with None. "
                    f"Question: {question} Use the value of the database field "
                    f'"{field}" to answer the question: '
                ),
            )
        else:
            self.add_message(
                "user",
                (
                    f"Include as much relevant detail as "
                    "possible that is related to/could help answer this question. Respond in "
                    "sentences, not just a single word or phrase."
                    f"Question: {question} Use this existing related database field "
                    f'"{field}" to answer the question: '
                ),
            )


class LlmFilterMessagesPrompt(OpenAIMessagesPromptBase):
    def __init__(self, filter_question: str):
        super().__init__()

        self.add_message(
            "system",
            ("You are a helpful classifier that generously filters database entries based on questions."),
        )

        self.add_message(
            "user",
            (
                "Given an entry and a question, you will answer the question relating "
                "to the entry. You only respond with 0, 1, 2, 3, 4, or 5 based on your "
                "confidence level. 0 is the most negative answer and 5 is the most positive "
                f"answer. Question: {filter_question}; Entry: "
            ),
        )


class SummarizeDataMessagesPrompt(OpenAIMessagesPromptBase):
    def __init__(self, question: str, text: str):
        super().__init__()

        self.add_message(
            "system",
            ("You are a helpful conversational English response generator for queries " "regarding database entries."),
        )

        self.add_message(
            "user",
            (
                "The following question and answer are in regards to database entries. "
                "Respond ONLY with a conversational English response WITH JUSTIFICATION to the question "
                f'"{question}" given the answer "{text}". Include as much detail/evidence as possible.'
            ),
        )


class LlmClusterEntityFormGroupsMessagesPrompt(OpenAIMessagesPromptBase):
    def __init__(self, field: str, instruction: str, text: str):
        super().__init__()

        self.add_message(
            "user",
            (
                f"You are given a list of values corresponding to the database field '{field}'. Categorize the "
                f"occurrences of '{field}' and create relevant non-overlapping groups. Return ONLY JSON with "
                f"the various categorized groups of '{field}' based on the following instructions '{instruction}'. "
                'Return your answer in the following JSON format and check your work: {{"groups": [string]}}. '
                'For example, if the instruction is "Form groups of different types of food" '
                'and the values are "banana, milk, yogurt, chocolate, oranges", you would return something like '
                "{{\"groups\": ['fruit', 'dairy', 'dessert', 'other']}}. Form groups to encompass as many entries "
                "as possible and don't create multiple groups with the same meaning. Here is the list values "
                f'values corresponding to "{field}": "{text}".'
            ),
        )


class LlmClusterEntityAssignGroupsMessagesPrompt(OpenAIMessagesPromptBase):
    def __init__(self, field: str, groups: list[str]):
        super().__init__()

        self.add_message(
            "user",
            (
                f"Categorize the database entry you are given corresponding to '{field}' into one of the "
                f'following groups: "{groups}". Perform your best work to assign the group. Return '
                f"ONLY the string corresponding to the selected group. Here is the database entry you will use: "
            ),
        )


_deprecated_prompts: dict[str, Type[GuidancePrompt]] = {
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


def _deprecated_prompt(name: str) -> GuidancePrompt:
    cls = _deprecated_prompts[name]
    logger.warn(f"The prompt {name} is deprecated. Switch to {cls.__name__}()")
    return cls()


def __getattr__(name: str):
    if name in _deprecated_prompts:
        return _deprecated_prompt(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
