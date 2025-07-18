from pathlib import Path
import pickle
import base64
import pytest
from typing import Any

from sycamore.functions.tokenizer import OpenAITokenizer
from sycamore.llms.openai import OpenAI, OpenAIModels, OpenAIClientWrapper
from sycamore.llms.openai import OpenAIModel, OpenAIClientType
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage, StaticPrompt
from sycamore.utils.cache import DiskCache

from pydantic import BaseModel


def cacheget(cache: DiskCache, key: str):
    hit = cache.get(key)
    return pickle.loads(base64.b64decode(hit))  # type: ignore


def cacheset(cache: DiskCache, key: str, data: Any):
    databytes = pickle.dumps(data)
    cache.set(key, base64.b64encode(databytes).decode("utf-8"))


# Note: These tests expect you to have OPENAI_API_KEY set in your environment.


def test_openai_defaults():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )
    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_openai_messages_defaults():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    messages = [
        RenderedMessage(
            role="system",
            content="You are a social media influencer",
        ),
        RenderedMessage(
            role="user",
            content="Write a caption for a recent trip to a sunny beach",
        ),
    ]
    prompt = RenderedPrompt(messages=messages)

    res = llm.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_cached_openai(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    key = llm._llm_cache_key(prompt, {})

    res = llm.generate(prompt=prompt, llm_kwargs={})

    # assert result is cached
    assert cacheget(cache, key).get("result") == res
    assert cacheget(cache, key).get("prompt") == prompt
    assert cacheget(cache, key).get("prompt.response_format") is None
    assert cacheget(cache, key).get("llm_kwargs") == {}
    assert cacheget(cache, key).get("model_name") == "gpt-3.5-turbo"

    # assert llm.generate is using cached result
    custom_output = {
        "result": "This is a custom response",
        "prompt": prompt,
        "prompt.response_format": None,
        "llm_kwargs": {},
        "model_name": "gpt-3.5-turbo",
    }
    cacheset(cache, key, custom_output)

    assert llm.generate(prompt=prompt, llm_kwargs={}) == custom_output["result"]


def test_cached_guidance(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    prompt = TestPrompt().render_generic()

    key = llm._llm_cache_key(prompt, None)

    res = llm.generate(prompt=prompt, llm_kwargs=None)

    # assert result is cached
    assert isinstance(cacheget(cache, key), dict)
    assert cacheget(cache, key).get("result") == res
    assert cacheget(cache, key).get("prompt") == prompt
    assert cacheget(cache, key).get("prompt.response_format") is None
    assert cacheget(cache, key).get("llm_kwargs") is None
    assert cacheget(cache, key).get("model_name") == "gpt-3.5-turbo"

    # assert llm.generate is using cached result
    custom_output = {
        "result": "This is a custom response",
        "prompt": TestPrompt().render_generic(),
        "prompt.response_format": None,
        "llm_kwargs": None,
        "model_name": "gpt-3.5-turbo",
    }
    cacheset(cache, key, custom_output)

    assert llm.generate(prompt=TestPrompt().render_generic(), llm_kwargs=None) == custom_output["result"]


def test_cached_openai_different_prompts(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
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


def test_cached_openai_different_models(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm_GPT_3_5_TURBO = OpenAI(OpenAIModels.GPT_3_5_TURBO, cache=cache)
    llm_GPT_4O_MINI = OpenAI(OpenAIModels.GPT_4O_MINI, cache=cache)

    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    # populate cache
    key_GPT_3_5_TURBO = llm_GPT_3_5_TURBO._llm_cache_key(prompt, {})
    res_GPT_3_5_TURBO = llm_GPT_3_5_TURBO.generate(prompt=prompt, llm_kwargs={})
    key_GPT_4O_MINI = llm_GPT_4O_MINI._llm_cache_key(prompt, {})
    res_GPT_4O_MINI = llm_GPT_4O_MINI.generate(prompt=prompt, llm_kwargs={})

    # check proper cached results
    assert cacheget(cache, key_GPT_3_5_TURBO).get("result") == res_GPT_3_5_TURBO
    assert cacheget(cache, key_GPT_3_5_TURBO).get("prompt") == prompt
    assert cacheget(cache, key_GPT_3_5_TURBO).get("llm_kwargs") == {}
    assert cacheget(cache, key_GPT_3_5_TURBO).get("model_name") == "gpt-3.5-turbo"
    assert cacheget(cache, key_GPT_4O_MINI).get("result") == res_GPT_4O_MINI
    assert cacheget(cache, key_GPT_4O_MINI).get("prompt") == prompt
    assert cacheget(cache, key_GPT_4O_MINI).get("llm_kwargs") == {}
    assert cacheget(cache, key_GPT_4O_MINI).get("model_name") == "gpt-4o-mini"

    # check for difference with model change
    assert key_GPT_3_5_TURBO != key_GPT_4O_MINI
    assert res_GPT_3_5_TURBO != res_GPT_4O_MINI


def test_cached_openai_pydantic_model(tmp_path: Path):
    cache = DiskCache(str(tmp_path))
    llm_GPT_4O_MINI = OpenAI(OpenAIModels.GPT_4O_MINI, cache=cache)

    class Statement(BaseModel):
        is_true: bool

    llm_kwargs = {}  # type: ignore
    llm_kwargs_cached = {}  # type: ignore

    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="2+2 = 4, is this statement true?")], response_format=Statement
    )

    # populate cache
    # pylint: disable=protected-access
    key_GPT_4O_MINI = llm_GPT_4O_MINI._llm_cache_key(prompt, llm_kwargs_cached)
    res_GPT_4O_MINI = llm_GPT_4O_MINI.generate(prompt=prompt, llm_kwargs=llm_kwargs)
    print(res_GPT_4O_MINI)
    assert key_GPT_4O_MINI is not None
    # check cache
    assert cacheget(cache, key_GPT_4O_MINI).get("result") == res_GPT_4O_MINI
    assert cacheget(cache, key_GPT_4O_MINI).get("prompt") == RenderedPrompt(messages=prompt.messages)
    assert cacheget(cache, key_GPT_4O_MINI).get(
        "prompt.response_format"
    ) == llm_GPT_4O_MINI._pickleable_response_format(prompt)
    assert cacheget(cache, key_GPT_4O_MINI).get("llm_kwargs") == llm_kwargs_cached
    assert cacheget(cache, key_GPT_4O_MINI).get("model_name") == "gpt-4o-mini"


class TestPrompt(StaticPrompt):
    def __init__(self):
        super().__init__(system="You are a skilled poet", user="Write a limerick about large language models")


def test_openai_defaults_guidance_chat():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    res = llm.generate(prompt=TestPrompt().render_generic())
    print(res)
    assert len(res) > 0


def test_openai_defaults_guidance_instruct():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT)
    res = llm.generate(prompt=TestPrompt().render_generic())
    assert len(res) > 0


def test_default_llm_kwargs():
    llm = OpenAI(OpenAIModels.GPT_4O_MINI, default_llm_kwargs={"max_tokens": 5})

    res = llm.generate(
        prompt=RenderedPrompt(
            messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
        )
    )

    num_tokens = len(OpenAITokenizer(OpenAIModels.GPT_4O_MINI.value.name).tokenize(res))
    assert num_tokens <= 5, f"Expected max_tokens to be 5, but got {num_tokens} tokens in the response: {res}"


def test_reasoning_model():
    llm = OpenAI(OpenAIModels.O4_MINI)
    res = llm.generate(prompt=TestPrompt().render_generic())
    assert len(res) > 0


@pytest.fixture(scope="module")
def azure_llm():
    # Note this deployment name is different from the official model name, which has a '.'

    return OpenAI(
        OpenAIModel("gpt-35-turbo", is_chat=True),
        client_wrapper=OpenAIClientWrapper(
            client_type=OpenAIClientType.AZURE,
            azure_endpoint="https://aryn.openai.azure.com",
            api_version="2024-02-15-preview",
        ),
    )


def test_azure_defaults_guidance_chat(azure_llm):
    prompt = TestPrompt().render_generic()
    res = azure_llm.generate(prompt=prompt)
    assert len(res) > 0


def test_azure_defaults_guidance_instruct(azure_llm):
    prompt = TestPrompt().render_generic()
    res = azure_llm.generate(prompt=prompt)
    assert len(res) > 0


def test_azure_pickle(azure_llm):
    pickled = pickle.dumps(azure_llm)
    _ = pickle.loads(pickled)
    assert True
