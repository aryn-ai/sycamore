import pytest
from unittest.mock import Mock, patch
from datetime import timedelta
from google.genai import types
from PIL import Image
from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.prompts.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import Cache


@pytest.fixture
def gemini_client():
    with patch("google.genai.Client") as mock_client:
        yield mock_client.return_value


@pytest.fixture
def gemini_llm(gemini_client):
    return Gemini(model_name=GeminiModels.GEMINI_2_PRO)


def test_init():
    llm = Gemini(model_name=GeminiModels.GEMINI_2_PRO)
    assert llm.model_name == GeminiModels.GEMINI_2_PRO
    assert llm.model.name == "gemini-2.0-pro-exp"
    assert llm.model.is_chat is True


def test_init_with_string():
    llm = Gemini(model_name="gemini-2.0-pro-exp")
    assert llm.model.name == "gemini-2.0-pro-exp"


def test_is_chat_mode():
    llm = Gemini(model_name=GeminiModels.GEMINI_2_PRO)
    assert llm.is_chat_mode() is True


def test_get_generate_kwargs(gemini_llm):
    prompt = RenderedPrompt(
        messages=[
            RenderedMessage(role="system", content="You are a helpful assistant"),
            RenderedMessage(role="user", content="Hello"),
            RenderedMessage(role="assistant", content="Hi there!"),
        ]
    )

    kwargs = gemini_llm.get_generate_kwargs(prompt)
    assert isinstance(kwargs["config"], types.GenerateContentConfig)
    assert isinstance(kwargs["content"], types.Content)


def test_generate_with_cache():
    cache = Cache()
    llm = Gemini(model_name=GeminiModels.GEMINI_2_PRO, cache=cache)
    prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="test")])

    cached_response = {
        "output": "cached response",
        "wall_latency": timedelta(seconds=1),
        "in_tokens": 10,
        "out_tokens": 20,
    }
    cache.set(prompt.cache_key(), cached_response)

    result = llm.generate(prompt=prompt)
    assert result == "cached response"


@pytest.mark.asyncio
async def test_generate_metadata(gemini_llm, gemini_client):
    mock_response = Mock()
    mock_response.candidates = [Mock(content="test response")]
    mock_response.usage_metadata = Mock(prompt_token_count=10, candidates_token_count=20)
    gemini_client.models.generate_content.return_value = mock_response

    prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="test")])
    result = gemini_llm.generate_metadata(prompt=prompt)

    assert "output" in result
    assert "wall_latency" in result
    assert "in_tokens" in result
    assert "out_tokens" in result
    assert result["output"] == "test response"
    assert result["in_tokens"] == 10
    assert result["out_tokens"] == 20


def test_generate_with_image():
    llm = Gemini(model_name=GeminiModels.GEMINI_2_PRO)
    test_image = Image.new("RGB", (100, 100))
    prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Describe this image", images=[test_image])])

    kwargs = llm.get_generate_kwargs(prompt)
    assert len(kwargs["content"].parts) == 2
    assert kwargs["content"].parts[0].text == "Describe this image"
