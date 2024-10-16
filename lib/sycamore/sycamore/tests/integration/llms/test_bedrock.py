from pathlib import Path

from sycamore.llms import Bedrock, BedrockModels
from sycamore.utils.cache import DiskCache


# Note: These tests assume your environment has been configured to access Amazon Bedrock.


def test_bedrock_defaults():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    assert len(res) > 0


def test_bedrock_messages_defaults():
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU)
    messages = [
        {
            "role": "user",
            "content": "Write a caption for a recent trip to a sunny beach",
        },
    ]
    prompt_kwargs = {"messages": messages}

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    assert len(res) > 0


def test_cached_bedrock(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU, cache=cache)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    # pylint: disable=protected-access
    key = llm._get_cache_key(prompt_kwargs, {})

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    # assert result is cached
    assert cache.get(key).get("result") == res
    assert cache.get(key).get("prompt_kwargs") == prompt_kwargs
    assert cache.get(key).get("llm_kwargs") == {}
    assert cache.get(key).get("model_name") == BedrockModels.CLAUDE_3_HAIKU.value.name

    # assert llm.generate is using cached result
    custom_output = {
        "result": "This is a custom response",
        "prompt_kwargs": prompt_kwargs,
        "llm_kwargs": {},
        "model_name": BedrockModels.CLAUDE_3_HAIKU.value.name,
    }
    cache.set(key, custom_output)

    assert llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={}) == custom_output["result"]


def test_cached_bedrock_different_prompts(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = Bedrock(BedrockModels.CLAUDE_3_HAIKU, cache=cache)
    prompt_kwargs_1 = {"prompt": "Write a limerick about large language models."}
    prompt_kwargs_2 = {"prompt": "Write a short limerick about large language models."}
    prompt_kwargs_3 = {"prompt": "Write a poem about large language models."}
    prompt_kwargs_4 = {"prompt": "Write a short poem about large language models."}

    key_1 = llm._get_cache_key(prompt_kwargs_1, {})
    key_2 = llm._get_cache_key(prompt_kwargs_2, {})
    key_3 = llm._get_cache_key(prompt_kwargs_3, {})
    key_4 = llm._get_cache_key(prompt_kwargs_4, {})
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

    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    # populate cache
    key_HAIKU = llm_HAIKU._get_cache_key(prompt_kwargs, {})
    res_HAIKU = llm_HAIKU.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})
    key_SONNET = llm_SONNET._get_cache_key(prompt_kwargs, {})
    res_SONNET = llm_SONNET.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    # check proper cached results
    assert cache.get(key_HAIKU).get("result") == res_HAIKU
    assert cache.get(key_HAIKU).get("prompt_kwargs") == prompt_kwargs
    assert cache.get(key_HAIKU).get("llm_kwargs") == {}
    assert cache.get(key_HAIKU).get("model_name") == BedrockModels.CLAUDE_3_HAIKU.value.name
    assert cache.get(key_SONNET).get("result") == res_SONNET
    assert cache.get(key_SONNET).get("prompt_kwargs") == prompt_kwargs
    assert cache.get(key_SONNET).get("llm_kwargs") == {}
    assert cache.get(key_SONNET).get("model_name") == BedrockModels.CLAUDE_3_SONNET.value.name

    # check for difference with model change
    assert key_HAIKU != key_SONNET
    assert res_HAIKU != res_SONNET
