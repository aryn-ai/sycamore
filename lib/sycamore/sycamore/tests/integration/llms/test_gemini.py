from pathlib import Path
from typing import Any
import base64
import pickle

import pytest

from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.prompts.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import DiskCache


@pytest.fixture
def anyio_backend():
    return "asyncio"


def cacheget(cache: DiskCache, key: str):
    hit = cache.get(key)
    return pickle.loads(base64.b64decode(hit))  # type: ignore


def cacheset(cache: DiskCache, key: str, data: Any):
    databytes = pickle.dumps(data)
    cache.set(key, base64.b64encode(databytes).decode("utf-8"))


def test_gemini_defaults():
    llm = Gemini(GeminiModels.GEMINI_2_FLASH)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


@pytest.mark.anyio
async def test_gemini_async_defaults():
    llm = Gemini(GeminiModels.GEMINI_2_FLASH)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    res = await llm.generate_async(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_gemini_messages_defaults():
    llm = Gemini(GeminiModels.GEMINI_2_FLASH)
    messages = [
        RenderedMessage(
            role="user",
            content="Write a caption for a recent trip to a sunny beach",
        ),
    ]
    prompt = RenderedPrompt(messages=messages)

    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_cached_gemini(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Gemini(GeminiModels.GEMINI_2_FLASH, cache=cache)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    # pylint: disable=protected-access
    key = llm._llm_cache_key(prompt, {})

    res = llm.generate(prompt=prompt, llm_kwargs={})

    # assert result is cached
    assert cacheget(cache, key).get("result")["output"] == res
    assert cacheget(cache, key).get("prompt") == prompt
    assert cacheget(cache, key).get("prompt.response_format") is None
    assert cacheget(cache, key).get("llm_kwargs") == {}
    assert cacheget(cache, key).get("model_name") == GeminiModels.GEMINI_2_FLASH.value.name

    # assert llm.generate is using cached result
    custom_output: dict[str, Any] = {
        "result": {"output": "This is a custom response"},
        "prompt": prompt,
        "prompt.response_format": None,
        "llm_kwargs": {},
        "model_name": GeminiModels.GEMINI_2_FLASH.value.name,
    }
    cacheset(cache, key, custom_output)

    assert llm.generate(prompt=prompt, llm_kwargs={}) == custom_output["result"]["output"]


def test_cached_gemini_different_prompts(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Gemini(GeminiModels.GEMINI_2_FLASH, cache=cache)
    prompt_1 = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )
    prompt_2 = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a short limerick about large language models.")]
    )
    prompt_3 = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a poem about large language models.")]
    )
    prompt_4 = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a short poem about large language models.")]
    )

    key_1 = llm._llm_cache_key(prompt_1, {})
    key_2 = llm._llm_cache_key(prompt_2, {})
    key_3 = llm._llm_cache_key(prompt_3, {})
    key_4 = llm._llm_cache_key(prompt_4, {})
    keys = [key_1, key_2, key_3, key_4]

    assert len(keys) == len(
        set(keys)
    ), f"""
    Cached query keys are not unique:
    key_1: {key_1}
    key_2: {key_2}
    key_3: {key_3}
    key_4: {key_4}
    """


def test_cached_gemini_different_models(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm_FLASH = Gemini(GeminiModels.GEMINI_2_FLASH, cache=cache)
    llm_FLASH_LITE = Gemini(GeminiModels.GEMINI_2_FLASH_LITE, cache=cache)

    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    # populate cache
    key_FLASH = llm_FLASH._llm_cache_key(prompt, {})
    res_FLASH = llm_FLASH.generate(prompt=prompt, llm_kwargs={})
    key_FLASH_LITE = llm_FLASH_LITE._llm_cache_key(prompt, {})
    res_FLASH_LITE = llm_FLASH_LITE.generate(prompt=prompt, llm_kwargs={})

    # check proper cached results
    assert cacheget(cache, key_FLASH_LITE).get("result")["output"] == res_FLASH_LITE
    assert cacheget(cache, key_FLASH_LITE).get("prompt") == prompt
    assert cacheget(cache, key_FLASH_LITE).get("llm_kwargs") == {}
    assert cacheget(cache, key_FLASH_LITE).get("model_name") == GeminiModels.GEMINI_2_FLASH_LITE.value.name
    assert cacheget(cache, key_FLASH).get("result")["output"] == res_FLASH
    assert cacheget(cache, key_FLASH).get("prompt") == prompt
    assert cacheget(cache, key_FLASH).get("llm_kwargs") == {}
    assert cacheget(cache, key_FLASH).get("model_name") == GeminiModels.GEMINI_2_FLASH.value.name

    # check for difference with model change
    assert key_FLASH != key_FLASH_LITE
    assert res_FLASH != res_FLASH_LITE


def test_metadata():
    llm = Gemini(GeminiModels.GEMINI_2_FLASH)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    res = llm.generate_metadata(prompt=prompt, llm_kwargs={})

    assert "output" in res
    assert "wall_latency" in res
    assert "in_tokens" in res
    assert "out_tokens" in res


def test_default_llm_kwargs():
    llm = Gemini(GeminiModels.GEMINI_2_5_FLASH_LITE, default_llm_kwargs={"max_output_tokens": 5})
    res = llm.generate_metadata(
        prompt=RenderedPrompt(
            messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
        ),
        llm_kwargs={},
    )
    assert res["out_tokens"] <= 5
