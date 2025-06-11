import io
from pathlib import Path
from unittest.mock import patch

from sycamore.llms import get_llm, MODELS
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.bedrock import Bedrock, BedrockModels
from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.llms import FakeLLM, LLMMode
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


class TestLLMCachingWithModelOverride:
    DEFAULT_MODEL = "default-model-v1"
    MODEL_A = "override-model-a"
    MODEL_B = "override-model-b"
    MODEL_C = "other-model-c" # For cache miss test

    PROMPT_1 = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello")])
    PROMPT_2 = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hi there")])

    KWARGS_1 = {"temperature": 0.0, "max_tokens": 100}
    KWARGS_2 = {"temperature": 0.0, "max_tokens": 200} # Different from KWARGS_1

    RESULT_1_MODEL_A = "Response from model A for prompt 1"
    RESULT_2_MODEL_B = "Response from model B for prompt 1"

    @pytest.fixture
    def in_memory_cache(self):
        # DiskCache(None) should create an in-memory dictionary-based cache
        return DiskCache(None)

    @pytest.fixture
    def fake_llm_for_caching(self, in_memory_cache):
        return FakeLLM(model_name=self.DEFAULT_MODEL, cache=in_memory_cache)

    def test_llm_cache_key_uniqueness(self, fake_llm_for_caching):
        llm = fake_llm_for_caching

        key1 = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, self.KWARGS_1)

        # Different model name, same prompt and kwargs
        key2 = llm._llm_cache_key(self.PROMPT_1, self.MODEL_B, self.KWARGS_1)
        assert key1 != key2, "Cache keys should differ for different model names"

        # Same model name, same prompt, different kwargs
        key3 = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, self.KWARGS_2)
        assert key1 != key3, "Cache keys should differ for different llm_kwargs"

        # Same model name, same kwargs, different prompt
        key4 = llm._llm_cache_key(self.PROMPT_2, self.MODEL_A, self.KWARGS_1)
        assert key1 != key4, "Cache keys should differ for different prompts"

        # Key with None kwargs
        key5 = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, None)
        assert key1 != key5, "Cache keys should differ for None llm_kwargs vs provided"

        key6 = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, None) # Re-generate to ensure consistency
        assert key5 == key6, "Cache keys should be consistent for identical inputs"


    def test_llm_cache_set_and_get_different_models(self, fake_llm_for_caching):
        llm = fake_llm_for_caching

        # Set for model A
        llm._llm_cache_set(self.PROMPT_1, self.MODEL_A, self.KWARGS_1, self.RESULT_1_MODEL_A)

        # Set for model B (same prompt and kwargs, different model)
        llm._llm_cache_set(self.PROMPT_1, self.MODEL_B, self.KWARGS_1, self.RESULT_2_MODEL_B)

        # Get for model A
        retrieved_a = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, self.KWARGS_1)
        assert retrieved_a == self.RESULT_1_MODEL_A, "Should retrieve correct result for model A"

        # Get for model B
        retrieved_b = llm._llm_cache_get(self.PROMPT_1, self.MODEL_B, self.KWARGS_1)
        assert retrieved_b == self.RESULT_2_MODEL_B, "Should retrieve correct result for model B"

        # Get for model C (not set)
        retrieved_c = llm._llm_cache_get(self.PROMPT_1, self.MODEL_C, self.KWARGS_1)
        assert retrieved_c is None, "Should return None for a model not in cache"

    def test_llm_cache_get_hit_validation_model_mismatch(self, fake_llm_for_caching, in_memory_cache):
        import pickle
        import base64

        llm = fake_llm_for_caching # Uses self.DEFAULT_MODEL ("default-model-v1") by default in its own _model_name

        # Data that would be stored if "old-model" was used during _llm_cache_set
        data_with_old_model = {
            "prompt": self.PROMPT_1, # Assuming RenderedPrompt is pickleable as is
            "prompt.response_format": llm._pickleable_response_format(self.PROMPT_1),
            "llm_kwargs": self.KWARGS_1,
            "model_name": "old-model-for-direct-cache-entry", # The mismatched part
            "result": "Result from old-model",
        }
        pickled_data_with_old_model = base64.b64encode(pickle.dumps(data_with_old_model)).decode("utf-8")

        # Key for retrieving with MODEL_A
        # The _llm_cache_get method will use effective_model_name=MODEL_A to generate this key
        key_for_model_a = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, self.KWARGS_1)

        # Manually set the cache entry using the key for MODEL_A, but with data claiming to be from "old-model"
        # This simulates a scenario where the key matches, but the content's model_name metadata is different.
        in_memory_cache.set(key_for_model_a, pickled_data_with_old_model)

        # Attempt to get with MODEL_A.
        # _llm_cache_get should find the entry by key, but then fail the internal validation
        # `hit.get("model_name") == effective_model_name` because "old-model-for-direct-cache-entry" != MODEL_A.
        retrieved = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, self.KWARGS_1)

        assert retrieved is None, "Should return None due to model_name mismatch during cache hit validation"

    def test_cache_disabled_for_non_zero_temperature(self, fake_llm_for_caching):
        llm = fake_llm_for_caching
        kwargs_high_temp = {"temperature": 0.7, "max_tokens": 100}

        # Try to set and get with high temperature
        llm._llm_cache_set(self.PROMPT_1, self.MODEL_A, kwargs_high_temp, "High temp result")
        retrieved = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, kwargs_high_temp)

        assert retrieved is None, "Caching should be disabled for non-zero temperature"

    def test_cache_works_with_none_kwargs(self, fake_llm_for_caching):
        llm = fake_llm_for_caching
        result_none_kwargs = "Result for None kwargs"

        llm._llm_cache_set(self.PROMPT_1, self.MODEL_A, None, result_none_kwargs)
        retrieved = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, None)
        assert retrieved == result_none_kwargs

        # Ensure it's different from empty dict kwargs if the hashing is different
        # (though typically None and {} might behave similarly in practice for cache keys if not careful)
        key_none = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, None)
        key_empty_dict = llm._llm_cache_key(self.PROMPT_1, self.MODEL_A, {})

        # This assertion depends on how None vs {} is handled in _llm_cache_key.
        # If they produce the same key, this test needs adjustment or becomes a test of that behavior.
        # Based on current LLM._llm_cache_key, they should be different as llm_kwargs itself is part of the dict.
        if key_none != key_empty_dict:
            retrieved_empty_dict = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, {})
            assert retrieved_empty_dict is None, "Cache for None kwargs should be distinct from empty dict kwargs"
        else:
            # If None and {} produce the same key, then getting with {} should return the same result.
            retrieved_empty_dict_same_key = llm._llm_cache_get(self.PROMPT_1, self.MODEL_A, {})
            assert retrieved_empty_dict_same_key == result_none_kwargs, "If keys are same, result should be same"
