import io
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from sycamore.llms import get_llm, MODELS
from sycamore.llms.chained_llm import ChainedLLM
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.bedrock import Bedrock, BedrockModels
from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.llms import FakeLLM, LLMMode, LLM
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


def test_default_llm_mode():
    llm = FakeLLM()
    assert llm.default_mode() == LLMMode.SYNC

    async_llm = FakeLLM(default_mode=LLMMode.ASYNC)
    assert async_llm.default_mode() == LLMMode.ASYNC


def test_merge_llm_kwargs():
    llm = FakeLLM(default_llm_kwargs={"temperature": 0.5, "max_tokens": 100})
    llm_kwargs = {"thinking_config": {"token_budget": 1000}, "max_tokens": 500}
    merged_kwargs = llm._merge_llm_kwargs(llm_kwargs)
    assert merged_kwargs == {"temperature": 0.5, "max_tokens": 500, "thinking_config": {"token_budget": 1000}}


def test_gemini_pickle():
    import pickle

    kwargs = {"max_output_tokens": 4092}

    gemini = Gemini(GeminiModels.GEMINI_2_5_FLASH_PREVIEW, default_llm_kwargs=kwargs)
    buf = io.BytesIO()
    pickle.dump(gemini, buf)

    buf.seek(0)

    g2 = pickle.loads(buf.getvalue())
    assert isinstance(g2, Gemini)
    assert g2._default_llm_kwargs == kwargs


@patch("boto3.client")
def test_get_llm(mock_boto3_client):
    assert isinstance(get_llm("openai." + OpenAIModels.TEXT_DAVINCI.value.name)(), OpenAI)
    assert isinstance(get_llm("bedrock." + BedrockModels.CLAUDE_3_5_SONNET.value.name)(), Bedrock)


class FooLLM(LLM):
    def __init__(self, model_name, default_mode: LLMMode):
        super().__init__(model_name, default_mode)

    def generate(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return "foo"

    def is_chat_mode(self) -> bool:
        return True


class BarLLM(LLM):
    def __init__(self, model_name, default_mode: LLMMode, throw_error: bool = False):
        super().__init__(model_name, default_mode)
        self.throw_error = throw_error

    def generate(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        if self.throw_error:
            raise RuntimeError("oops, something went wrong")
        return "bar"

    def is_chat_mode(self) -> bool:
        return True


def test_chained_llm():
    foo = FooLLM("foo_model", LLMMode.SYNC)
    bar = BarLLM("bar_model", LLMMode.SYNC)

    chained = ChainedLLM([foo, bar], model_name="", default_mode=LLMMode.SYNC)
    res = chained.generate(prompt=RenderedPrompt(messages=[]))
    assert res == "foo"  # Should return from the first LLM in the chain

    chained = ChainedLLM([bar, foo], model_name="", default_mode=LLMMode.SYNC)
    res = chained.generate(prompt=RenderedPrompt(messages=[]))
    assert res == "bar"  # Should return from the first LLM in the chain

    bar2 = BarLLM("bar_model", LLMMode.SYNC, throw_error=True)
    chained = ChainedLLM([bar2, foo], model_name="", default_mode=LLMMode.SYNC)
    res = chained.generate(prompt=RenderedPrompt(messages=[]))
    assert res == "foo"  # Should return from the first LLM in the chain


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
