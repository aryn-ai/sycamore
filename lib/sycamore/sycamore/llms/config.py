from dataclasses import dataclass
from enum import Enum


@dataclass
class OpenAIModel:
    name: str
    is_chat: bool = False


class OpenAIModels(Enum):
    TEXT_DAVINCI = OpenAIModel(name="text-davinci-003", is_chat=True)
    GPT_3_5_TURBO = OpenAIModel(name="gpt-3.5-turbo", is_chat=True)
    GPT_4_TURBO = OpenAIModel(name="gpt-4-turbo", is_chat=True)
    GPT_4O = OpenAIModel(name="gpt-4o", is_chat=True)
    GPT_4O_STRUCTURED = OpenAIModel(
        name="gpt-4o-2024-08-06", is_chat=True
    )  # remove after october 2nd, gpt-4o will point to this model then
    GPT_4O_MINI = OpenAIModel(name="gpt-4o-mini", is_chat=True)
    GPT_3_5_TURBO_INSTRUCT = OpenAIModel(name="gpt-3.5-turbo-instruct", is_chat=False)
    GPT_4_1 = OpenAIModel(name="gpt-4.1", is_chat=True)
    GPT_4_1_MINI = OpenAIModel(name="gpt-4.1-mini", is_chat=True)
    GPT_4_1_NANO = OpenAIModel(name="gpt-4.1-nano", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None
