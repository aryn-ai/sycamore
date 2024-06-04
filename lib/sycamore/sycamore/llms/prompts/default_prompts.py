from abc import ABC, abstractmethod
import logging
from typing import Type

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
    user = """You are given a few text elements of a document. The {entity} of the document is in these few text elements. Using
    this context, FIND,COPY, and RETURN the {entity}. Only return the {entity} as part of your answer. DO NOT
    REPHRASE OR MAKE UP AN ANSWER.
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
