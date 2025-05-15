from pathlib import Path
from typing import Any
import pickle
import base64

from sycamore.llms.bedrock import Bedrock, BedrockModels
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import DiskCache


def cacheget(cache: DiskCache, key: str):
    hit = cache.get(key)
    return pickle.loads(base64.b64decode(hit))  # type: ignore


def cacheset(cache: DiskCache, key: str, data: Any):
    databytes = pickle.dumps(data)
    cache.set(key, base64.b64encode(databytes).decode("utf-8"))


# Note: These tests assume your environment has been configured to access Amazon Bedrock.


def test_bedrock_defaults():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_bedrock_messages_defaults():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU)
    messages = [
        RenderedMessage(
            role="user",
            content="Write a caption for a recent trip to a sunny beach",
        ),
    ]
    prompt = RenderedPrompt(messages=messages)

    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_cached_bedrock(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU, cache=cache)
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
    assert cacheget(cache, key).get("model_name") == BedrockModels.CLAUDE_3_HAIKU.value.name

    # assert llm.generate is using cached result
    custom_output: dict[str, Any] = {
        "result": {"output": "This is a custom response"},
        "prompt": prompt,
        "prompt.response_format": None,
        "llm_kwargs": {},
        "model_name": BedrockModels.CLAUDE_3_HAIKU.value.name,
    }
    cacheset(cache, key, custom_output)

    assert llm.generate(prompt=prompt, llm_kwargs={}) == custom_output["result"]["output"]


def test_cached_bedrock_different_prompts(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU, cache=cache)
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


def test_cached_bedrock_different_models(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm_HAIKU = Bedrock(BedrockModels.CLAUDE_3_HAIKU, cache=cache)
    llm_SONNET = Bedrock(BedrockModels.CLAUDE_3_SONNET, cache=cache)

    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    # populate cache
    key_HAIKU = llm_HAIKU._llm_cache_key(prompt, {})
    res_HAIKU = llm_HAIKU.generate(prompt=prompt, llm_kwargs={})
    key_SONNET = llm_SONNET._llm_cache_key(prompt, {})
    res_SONNET = llm_SONNET.generate(prompt=prompt, llm_kwargs={})

    # check proper cached results
    assert cacheget(cache, key_HAIKU).get("result")["output"] == res_HAIKU
    assert cacheget(cache, key_HAIKU).get("prompt") == prompt
    assert cacheget(cache, key_HAIKU).get("llm_kwargs") == {}
    assert cacheget(cache, key_HAIKU).get("model_name") == BedrockModels.CLAUDE_3_HAIKU.value.name
    assert cacheget(cache, key_SONNET).get("result")["output"] == res_SONNET
    assert cacheget(cache, key_SONNET).get("prompt") == prompt
    assert cacheget(cache, key_SONNET).get("llm_kwargs") == {}
    assert cacheget(cache, key_SONNET).get("model_name") == BedrockModels.CLAUDE_3_SONNET.value.name

    # check for difference with model change
    assert key_HAIKU != key_SONNET
    assert res_HAIKU != res_SONNET


def test_metadata():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    res = llm.generate_metadata(prompt=prompt, llm_kwargs={})

    assert "output" in res
    assert "wall_latency" in res
    assert "server_latency" in res
    assert "in_tokens" in res
    assert "out_tokens" in res


def test_default_llm_kwargs():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU, default_llm_kwargs={"max_tokens": 5})
    res = llm.generate_metadata(
        prompt=RenderedPrompt(
            messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
        )
    )
    assert res["out_tokens"] <= 5
