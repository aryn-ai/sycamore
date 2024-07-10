from pathlib import Path

from sycamore.llms import OpenAI, OpenAIModels, OpenAIClientWrapper
from sycamore.llms.openai import OpenAIModel, OpenAIClientType
from sycamore.llms.prompts.default_prompts import SimpleGuidancePrompt
from sycamore.utils.cache import DiskCache


# Note: These tests expect you to have OPENAI_API_KEY set in your environment.


def test_openai_defaults():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    assert len(res) > 0


def test_openai_messages_defaults():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    messages = [
        {
            "role": "system",
            "content": "You are a social media influencer",
        },
        {
            "role": "user",
            "content": "Write a caption for a recent trip to a sunny beach",
        },
    ]
    prompt_kwargs = {"messages": messages}

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    assert len(res) > 0


def test_cached_openai(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    key = llm._get_cache_key(prompt_kwargs, {})

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    # assert result is cached
    assert cache.get(key).get("result") == res

    # assert llm.generate is using cached result
    custom_output = {"result": "This is a custom response", "prompt_kwargs": prompt_kwargs, "llm_kwargs": {}}
    cache.set(key, custom_output)

    assert llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={}) == custom_output["result"]


def test_cached_guidance(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    prompt_kwargs = {"prompt": TestGuidancePrompt()}

    key = llm._get_cache_key(prompt_kwargs, None)

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=None)

    # assert result is cached
    assert cache.get(key).get("result") == res

    # assert llm.generate is using cached result
    custom_output = {
        "result": "This is a custom response",
        "prompt_kwargs": {"prompt": TestGuidancePrompt()},
        "llm_kwargs": None,
    }
    cache.set(key, custom_output)

    assert llm.generate(prompt_kwargs={"prompt": TestGuidancePrompt()}, llm_kwargs=None) == custom_output["result"]


def test_cached_openai_mismatch(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    key = llm._get_cache_key(prompt_kwargs, {})

    # store a modified result in the cache (changed prompt_kwargs), ensure there is a cache miss
    custom_output = {"result": "This is a custom response", "prompt_kwargs": {}, "llm_kwargs": {}}
    cache.set(key, custom_output)

    assert llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={}) != custom_output["result"]


class TestGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a skilled poet"
    user = "Write a limerick about large language models"


def test_openai_defaults_guidance_chat():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    print(res)
    assert len(res) > 0


def test_openai_defaults_guidance_instruct():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT)
    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0


def test_azure_defaults_guidance_chat():
    llm = OpenAI(
        # Note this deployment name is different from the official model name, which has a '.'
        OpenAIModel("gpt-35-turbo", is_chat=True),
        client_wrapper=OpenAIClientWrapper(
            client_type=OpenAIClientType.AZURE,
            azure_endpoint="https://aryn.openai.azure.com",
            api_version="2024-02-15-preview",
        ),
    )

    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0


def test_azure_defaults_guidance_instruct():
    llm = OpenAI(
        # Note this deployment name is different from the official model name, which has a '.'
        OpenAIModel("gpt-35-turbo-instruct", is_chat=False),
        client_wrapper=OpenAIClientWrapper(
            client_type=OpenAIClientType.AZURE,
            azure_endpoint="https://aryn.openai.azure.com",
            api_version="2024-02-15-preview",
        ),
    )

    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0
