import asyncio
import tempfile
from unittest.mock import patch, AsyncMock

import pytest

from sycamore.llms.ollama import Ollama
from sycamore.llms.config import OllamaModels, OllamaModel
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import DiskCache


def ollama_chat_response(content: str, prompt_eval_count: int = 30, eval_count: int = 50) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_simple(mock_client_cls, mock_async_client_cls):
    mock_client_cls.return_value.chat.return_value = ollama_chat_response("Here is your result: 56")

    llm = Ollama("llama3.1")
    assert llm.is_chat_mode()
    assert llm._model_name == "llama3.1"

    result = llm.generate(
        prompt=RenderedPrompt(messages=[RenderedMessage(role="user", content="Roll 4d20 and tell me the sum.")])
    )
    assert result == "Here is your result: 56"

    mock_client_cls.assert_called_once_with(host=None)
    kwargs = mock_client_cls.return_value.chat.call_args.kwargs
    assert kwargs["model"] == "llama3.1"
    assert kwargs["messages"] == [{"role": "user", "content": "Roll 4d20 and tell me the sum."}]
    assert kwargs["options"] == {"temperature": 0}


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_unknown_model_name_does_not_raise(mock_client_cls, mock_async_client_cls):
    # Ollama's catalog is whatever the user has pulled, so unrecognized names should
    # build an ad hoc model rather than raising like the other providers do.
    llm = Ollama("some-custom-model-i-pulled")
    assert llm._model_name == "some-custom-model-i-pulled"
    assert isinstance(llm.model, OllamaModel)


def test_ollama_models_from_name_known():
    model = OllamaModels.from_name("llama3.1")
    assert model.name == "llama3.1"
    assert model is OllamaModels.LLAMA3_1.value


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_host_and_client_args(mock_client_cls, mock_async_client_cls):
    Ollama("llama3.1", host="http://my-ollama:11434", client_args={"timeout": 30})
    mock_client_cls.assert_called_once_with(host="http://my-ollama:11434", timeout=30)
    mock_async_client_cls.assert_called_once_with(host="http://my-ollama:11434", timeout=30)


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_system_message(mock_client_cls, mock_async_client_cls):
    mock_client_cls.return_value.chat.return_value = ollama_chat_response("ok")

    llm = Ollama("llama3.1")
    llm.generate(
        prompt=RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="You are a DM for a game of D&D."),
                RenderedMessage(role="user", content="Roll 4d20."),
            ]
        )
    )
    kwargs = mock_client_cls.return_value.chat.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are a DM for a game of D&D."},
        {"role": "user", "content": "Roll 4d20."},
    ]


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_with_llm_kwargs(mock_client_cls, mock_async_client_cls):
    mock_client_cls.return_value.chat.return_value = ollama_chat_response("ok")

    llm = Ollama("llama3.1")
    llm.generate(
        prompt=RenderedPrompt(messages=[RenderedMessage(role="user", content="hi")]),
        llm_kwargs={"options": {"temperature": 0.7, "num_ctx": 8192}},
    )
    kwargs = mock_client_cls.return_value.chat.call_args.kwargs
    assert kwargs["options"] == {"temperature": 0.7, "num_ctx": 8192}


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_with_cache(mock_client_cls, mock_async_client_cls):
    mock_client_cls.return_value.chat.return_value = ollama_chat_response("Here is your result: 56")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskCache(temp_dir)
        assert cache.get_hit_info() == (0, 0)

        llm = Ollama("llama3.1", cache=cache)
        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Roll 4d20.")])

        result = llm.generate(prompt=prompt)
        assert result == "Here is your result: 56"
        assert cache.get_hit_info() == (0, 1)

        result = llm.generate(prompt=prompt)
        assert result == "Here is your result: 56"
        assert cache.get_hit_info() == (1, 1)
        assert mock_client_cls.return_value.chat.call_count == 1


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_generate_async(mock_client_cls, mock_async_client_cls):
    mock_async_client_cls.return_value.chat = AsyncMock(return_value=ollama_chat_response("async result"))

    llm = Ollama("llama3.1")
    prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="hi")])
    result = asyncio.run(llm.generate_async(prompt=prompt))
    assert result == "async result"
    mock_async_client_cls.return_value.chat.assert_called_once()


@patch("ollama.AsyncClient")
@patch("ollama.Client")
def test_ollama_pickle_roundtrip(mock_client_cls, mock_async_client_cls):
    import pickle

    llm = Ollama("llama3.1", host="http://my-ollama:11434")
    restored = pickle.loads(pickle.dumps(llm))
    assert restored._model_name == "llama3.1"
    assert restored.host == "http://my-ollama:11434"


def test_ollama_requires_module_not_installed():
    with patch.dict("sys.modules", {"ollama": None}):
        with pytest.raises(ImportError):
            Ollama("llama3.1")
