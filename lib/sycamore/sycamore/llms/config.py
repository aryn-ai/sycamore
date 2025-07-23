from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnthropicModels(Enum):
    """Represents available Claude models."""

    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-latest"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    @classmethod
    def from_name(cls, name: str) -> Optional["AnthropicModels"]:
        for m in iter(cls):
            if m.value == name:
                return m
        return None


@dataclass
class BedrockModel:
    name: str
    is_chat: bool = False


class BedrockModels(Enum):
    """Represents available Bedrock models."""

    # Note that the models available on a given Bedrock account may vary.
    CLAUDE_3_HAIKU = BedrockModel(name="anthropic.claude-3-haiku-20240307-v1:0", is_chat=True)
    CLAUDE_3_SONNET = BedrockModel(name="anthropic.claude-3-sonnet-20240229-v1:0", is_chat=True)
    CLAUDE_3_OPUS = BedrockModel(name="anthropic.claude-3-opus-20240229-v1:0", is_chat=True)
    CLAUDE_3_5_SONNET = BedrockModel(name="anthropic.claude-3-5-sonnet-20241022-v2:0", is_chat=True)
    CLAUDE_3_7_SONNET = BedrockModel(name="anthropic.claude-3-7-sonnet-20250219-v1:0", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


@dataclass
class GeminiModel:
    name: str
    is_chat: bool = False


class GeminiModels(Enum):
    """Represents available Gemini models. More info: https://googleapis.github.io/python-genai/"""

    # Note that the models available on a given Gemini account may vary.
    GEMINI_2_5_FLASH = GeminiModel(name="gemini-2.5-flash", is_chat=True)
    GEMINI_2_5_FLASH_PREVIEW = GEMINI_2_5_FLASH  # Alias for the preview model

    GEMINI_2_5_PRO = GeminiModel(name="gemini-2.5-pro", is_chat=True)
    GEMINI_2_5_PRO_PREVIEW = GEMINI_2_5_PRO  # Alias for the preview model

    GEMINI_2_5_FLASH_LITE = GeminiModel(name="gemini-2.5-flash-lite", is_chat=True)
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

    O4_MINI = OpenAIModel(name="o4-mini", is_chat=True)
    O3 = OpenAIModel(name="o3", is_chat=True)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None
