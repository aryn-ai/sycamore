from pathlib import Path
from unittest.mock import patch

from sycamore.llms import OpenAI, OpenAIModels, Bedrock, BedrockModels, get_llm, MODELS
from sycamore.llms.llms import FakeLLM
from sycamore.llms.prompts import EntityExtractorFewShotGuidancePrompt, EntityExtractorZeroShotGuidancePrompt
from sycamore.utils.cache import DiskCache


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


class TestCache:
    def test_nocache(self, tmp_path):
        llm = FakeLLM()
        llm._llm_cache_set({}, None, "abc")
        assert llm._llm_cache_get({}, None) is None

    def test_use_caching(self, tmp_path: Path):
        llm = FakeLLM()
        assert llm._use_caching(None) is False

        llm = FakeLLM(cache=DiskCache(str(tmp_path)))
        assert llm._use_caching(None)
        assert llm._use_caching({})
        assert llm._use_caching({"temperature": 0})
        assert not llm._use_caching({"temperature": 1})

    def test_cache(self, tmp_path: Path):
        llm = FakeLLM(cache=DiskCache(str(tmp_path)))

        def doit(prompt_kwargs, llm_kwargs, result, overwrite=False, already_set=False):
            if overwrite:
                assert llm._llm_cache_get(prompt_kwargs, llm_kwargs) is not None
                llm._llm_cache_set(prompt_kwargs, llm_kwargs, result)
            elif not already_set:
                assert llm._llm_cache_get(prompt_kwargs, llm_kwargs) is None
                llm._llm_cache_set(prompt_kwargs, llm_kwargs, result)

            assert llm._llm_cache_get(prompt_kwargs, llm_kwargs) == result

        doit({}, None, "abc")
        doit({}, None, "abc2", overwrite=True)
        doit({}, {}, "def")
        doit({"prompt": "foff"}, {}, {"ghi": "jkl"})
        doit({}, {"magic": True}, [1, 2, 3])
        doit({}, None, "abc2", already_set=True)
