from pathlib import Path
from unittest.mock import patch

from sycamore.llms import OpenAI, OpenAIModels, Bedrock, BedrockModels, get_llm, MODELS
from sycamore.llms.llms import FakeLLM
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import DiskCache
import datetime
from sycamore.utils.thread_local import ThreadLocalAccess


def test_get_metadata():
    llm = FakeLLM()
    wall_latency = datetime.timedelta(seconds=1)
    metadata = llm.get_metadata({"prompt": "Hello", "temperature": 0.7}, "Test output", wall_latency, 10, 5)
    assert metadata["model"] == llm._model_name
    assert metadata["usage"] == {
        "completion_tokens": 10,
        "prompt_tokens": 5,
        "total_tokens": 15,
    }
    assert metadata["prompt"] == "Hello"
    assert metadata["output"] == "Test output"
    assert metadata["temperature"] == 0.7
    assert metadata["wall_latency"] == wall_latency


@patch("sycamore.llms.llms.add_metadata")
def test_add_llm_metadata(mock_add_metadata):
    llm = FakeLLM()
    with patch.object(ThreadLocalAccess, "present", return_value=True):
        llm.add_llm_metadata({}, "Test output", datetime.timedelta(seconds=0.5), 1, 2)
        mock_add_metadata.assert_called_once()

    # If TLS not present, add_metadata should not be called
    mock_add_metadata.reset_mock()
    with patch.object(ThreadLocalAccess, "present", return_value=False):
        llm.add_llm_metadata({}, "Test output", datetime.timedelta(seconds=0.5), 1, 2)
        mock_add_metadata.assert_not_called()


def test_openai_davinci_fallback():
    llm = OpenAI(model_name=OpenAIModels.TEXT_DAVINCI.value)
    assert llm._model_name == OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value.name


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
        llm._llm_cache_set(RenderedPrompt(messages=[]), None, "abc")
        assert llm._llm_cache_get(RenderedPrompt(messages=[]), None) is None

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

        def doit(prompt, llm_kwargs, result, overwrite=False, already_set=False):
            if overwrite:
                assert llm._llm_cache_get(prompt, llm_kwargs) is not None
                llm._llm_cache_set(prompt, llm_kwargs, result)
            elif not already_set:
                assert llm._llm_cache_get(prompt, llm_kwargs) is None
                llm._llm_cache_set(prompt, llm_kwargs, result)

            assert llm._llm_cache_get(prompt, llm_kwargs) == result

        doit(RenderedPrompt(messages=[]), None, "abc")
        doit(RenderedPrompt(messages=[]), None, "abc2", overwrite=True)
        doit(RenderedPrompt(messages=[]), {}, "def")
        doit(RenderedPrompt(messages=[RenderedMessage(role="user", content="foff")]), {}, {"ghi": "jkl"})
        doit(RenderedPrompt(messages=[]), {"magic": True}, [1, 2, 3])
        doit(RenderedPrompt(messages=[]), None, "abc2", already_set=True)
