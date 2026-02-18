from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMMode(Enum):
    SYNC = 1
    ASYNC = 2
    BATCH = 3


class LLMModel:
    name: str
    is_chat: bool


@dataclass
class AnthropicModel(LLMModel):
    name: str
    is_chat: bool = False


class AnthropicModels(Enum):
    """Represents available Claude models."""

    CLAUDE_4_5_SONNET = AnthropicModel(name="claude-sonnet-4-5-20250929", is_chat=True)
    CLAUDE_4_5_OPUS = AnthropicModel(name="claude-opus-4-5-20251101", is_chat=True)
    CLAUDE_4_5_HAIKU = AnthropicModel(name="claude-haiku-4-5-20251001", is_chat=True)
    CLAUDE_4_1_OPUS = AnthropicModel(name="claude-opus-4-1-20250805", is_chat=True)
    CLAUDE_4_OPUS = AnthropicModel(name="claude-opus-4-20250514", is_chat=True)
    CLAUDE_4_SONNET = AnthropicModel(name="claude-sonnet-4-20250514", is_chat=True)
    CLAUDE_3_7_SONNET = AnthropicModel(name="claude-3-7-sonnet-latest", is_chat=True)
    CLAUDE_3_5_HAIKU = AnthropicModel(name="claude-3-5-haiku-latest", is_chat=True)
    CLAUDE_3_HAIKU = AnthropicModel(name="claude-3-haiku-20240307", is_chat=True)

    @classmethod
    def from_name(cls, name: str) -> Optional["AnthropicModels"]:
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


@dataclass
class BedrockModel(LLMModel):
    name: str
    is_chat: bool = False


def bedrock_derived(model: AnthropicModels) -> BedrockModel:
    return BedrockModel(name=f"us.anthropic.{model.value.name}-v1:0", is_chat=model.value.is_chat)


def old_bedrock_derived(model: AnthropicModels) -> BedrockModel:
    return BedrockModel(name=f"anthropic.{model.value.name}-v1:0", is_chat=model.value.is_chat)


class BedrockModels(Enum):
    """Represents available Bedrock models."""

    # Note that the models available on a given Bedrock account may vary.
    CLAUDE_4_5_SONNET = bedrock_derived(AnthropicModels.CLAUDE_4_5_SONNET)
    CLAUDE_4_5_HAIKU = bedrock_derived(AnthropicModels.CLAUDE_4_5_HAIKU)
    CLAUDE_4_1_OPUS = bedrock_derived(AnthropicModels.CLAUDE_4_1_OPUS)
    CLAUDE_4_OPUS = bedrock_derived(AnthropicModels.CLAUDE_4_OPUS)
    CLAUDE_4_SONNET = bedrock_derived(AnthropicModels.CLAUDE_4_SONNET)
    CLAUDE_3_7_SONNET = bedrock_derived(AnthropicModels.CLAUDE_3_7_SONNET)
    CLAUDE_3_5_HAIKU = bedrock_derived(AnthropicModels.CLAUDE_3_5_HAIKU)
    CLAUDE_3_HAIKU = old_bedrock_derived(AnthropicModels.CLAUDE_3_HAIKU)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


@dataclass
class GeminiModel(LLMModel):
    name: str
    is_chat: bool = False


class GeminiModels(Enum):
    """Represents available Gemini models. More info: https://googleapis.github.io/python-genai/"""

    GEMINI_3_PRO_PREVIEW = GeminiModel(name="gemini-3-pro-preview", is_chat=True)
    GEMINI_3_FLASH_PREVIEW = GeminiModel(name="gemini-3-flash-preview", is_chat=True)

    # Note that the models available on a given Gemini account may vary.
    GEMINI_FLASH_LATEST = GeminiModel(name="gemini-flash-latest", is_chat=True)  # latest including preview
    GEMINI_2_5_FLASH = GeminiModel(name="gemini-2.5-flash", is_chat=True)  # stable
    # This should be deprecated in favor of LATEST
    GEMINI_2_5_FLASH_PREVIEW = GEMINI_2_5_FLASH  # Alias for the preview model

    GEMINI_2_5_PRO = GeminiModel(name="gemini-2.5-pro", is_chat=True)
    GEMINI_2_5_PRO_PREVIEW = GEMINI_2_5_PRO  # Alias for the preview model

    GEMINI_FLASH_LITE_LATEST = GeminiModel(name="gemini-flash-lite-latest", is_chat=True)  # latest including preview
    GEMINI_2_5_FLASH_LITE = GeminiModel(name="gemini-2.5-flash-lite", is_chat=True)  # stable
    # This should be deprecated in favor of LATEST
    GEMINI_2_5_FLASH_LITE_PREVIEW = GEMINI_2_5_FLASH_LITE  # Alias for the preview model

    GEMINI_2_FLASH = GeminiModel(name="gemini-2.0-flash", is_chat=True)
    GEMINI_2_FLASH_LITE = GeminiModel(name="gemini-2.0-flash-lite", is_chat=True)
    GEMINI_2_FLASH_THINKING = GeminiModel(name="gemini-2.0-flash-thinking-exp", is_chat=True)
    GEMINI_2_PRO = GeminiModel(name="gemini-2.0-pro-exp-02-05", is_chat=True)
    GEMINI_1_5_PRO = GeminiModel(name="gemini-1.5-pro", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


@dataclass
class OpenAIModel(LLMModel):
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

    O4_MINI = OpenAIModel(name="o4-mini", is_chat=True)
    O3 = OpenAIModel(name="o3", is_chat=True)

    GPT_5 = OpenAIModel(name="gpt-5", is_chat=True)
    GPT_5_MINI = OpenAIModel(name="gpt-5-mini", is_chat=True)
    GPT_5_NANO = OpenAIModel(name="gpt-5-nano", is_chat=True)

    GPT_5_1 = OpenAIModel(name="gpt-5.1", is_chat=True)
    GPT_5_2 = OpenAIModel(name="gpt-5.2", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


class ChainedModel(LLMModel):

    def __init__(self, chain: list[LLMModel]):
        self.chain = chain
        self.is_chat = True  # This is not used anywhere.
