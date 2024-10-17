from unittest.mock import patch

from sycamore.llms import OpenAI, OpenAIModels, Bedrock, BedrockModels, get_llm, MODELS
from sycamore.llms.prompts import EntityExtractorFewShotGuidancePrompt, EntityExtractorZeroShotGuidancePrompt


def test_openai_davinci_fallback():
    llm = OpenAI(model_name=OpenAIModels.TEXT_DAVINCI.value)
    assert llm._model_name == OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value.name


def test_deprecated_prompt_fallback():
    from sycamore.llms.prompts.default_prompts import ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT

    assert isinstance(ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT, EntityExtractorZeroShotGuidancePrompt)

    from sycamore.llms.prompts import ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT

    assert isinstance(ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT, EntityExtractorFewShotGuidancePrompt)


def test_model_list():
    assert "openai." + OpenAIModels.TEXT_DAVINCI.value.name in MODELS
    assert "bedrock." + BedrockModels.CLAUDE_3_5_SONNET.value.name in MODELS


@patch("boto3.client")
def test_get_llm(mock_boto3_client):
    assert isinstance(get_llm("openai." + OpenAIModels.TEXT_DAVINCI.value.name)(), OpenAI)
    assert isinstance(get_llm("bedrock." + BedrockModels.CLAUDE_3_5_SONNET.value.name)(), Bedrock)
