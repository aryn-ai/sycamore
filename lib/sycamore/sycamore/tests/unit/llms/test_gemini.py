import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import datetime

# Assuming google.generativeai.types are available for mocking
try:
    from google.generativeai.types import GenerateContentResponse, GenerationConfig, Content, Part, UsageMetadata, Candidate, FinishReason
except ImportError:
    # Create dummy types if google.generativeai is not installed, for basic testing structure
    class GenerateContentResponse: pass
    class GenerationConfig: pass
    class Content: pass
    class Part: pass
    class UsageMetadata: pass
    class Candidate: pass
    class FinishReason: pass
    FinishReason.STOP = "STOP"


from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage

DEFAULT_MODEL_NAME = GeminiModels.GEMINI_PRO.value.name
OVERRIDE_MODEL_NAME = "gemini-1.5-pro-latest" # A hypothetical different model for testing override

@pytest.fixture
def mock_gemini_client():
    """Mocks the google.genai.Client and its relevant methods."""
    with patch("google.genai.Client") as MockClientConst:
        mock_client_instance = MockClientConst.return_value

        # Sync client's model part
        mock_models_sync = MagicMock()
        mock_client_instance.models = mock_models_sync

        # Async client's model part (aio)
        mock_aio_client = MagicMock() # This is the parent of aio.models
        mock_models_async = AsyncMock()
        mock_aio_client.models = mock_models_async
        mock_client_instance.aio = mock_aio_client

        yield mock_models_sync, mock_models_async


@pytest.fixture
def gemini_llm(mock_gemini_client): # Depends on mock_gemini_client to ensure client is mocked before LLM instantiation
    # Using a dummy API key as we are mocking the client calls
    return Gemini(model_name=DEFAULT_MODEL_NAME, api_key="test_key")

def create_mock_response(model_name_used: str, text_content: str = "Test response") -> GenerateContentResponse:
    """Helper to create a mock GenerateContentResponse."""
    response = MagicMock(spec=GenerateContentResponse)
    response.text = text_content # For simpler access if needed, though paths/parts are more accurate

    # Mocking the structure accurately
    mock_candidate = MagicMock(spec=Candidate)
    mock_candidate.finish_reason = FinishReason.STOP
    mock_part = MagicMock(spec=Part)
    mock_part.text = text_content
    mock_content = MagicMock(spec=Content)
    mock_content.parts = [mock_part]
    mock_content.role = "model"
    mock_candidate.content = mock_content

    response.candidates = [mock_candidate]

    mock_usage_metadata = MagicMock(spec=UsageMetadata)
    mock_usage_metadata.prompt_token_count = 10
    mock_usage_metadata.candidates_token_count = 5
    mock_usage_metadata.total_token_count = 15
    response.usage_metadata = mock_usage_metadata

    # If the response object itself has a 'model' attribute or similar to verify
    # This depends on actual API, for now, we check model in the call to generate_content
    return response


class TestGeminiModelOverride:
    def test_generate_model_override(self, gemini_llm, mock_gemini_client):
        mock_sync_models, _ = mock_gemini_client

        mock_response = create_mock_response(OVERRIDE_MODEL_NAME, "Response from override model")
        mock_sync_models.generate_content.return_value = mock_response

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello")])
        # generate calls generate_metadata internally
        gemini_llm.generate(prompt=prompt, model_name=OVERRIDE_MODEL_NAME)

        mock_sync_models.generate_content.assert_called_once()
        call_args = mock_sync_models.generate_content.call_args
        assert call_args.kwargs["model"] == OVERRIDE_MODEL_NAME

    def test_generate_model_fallback(self, gemini_llm, mock_gemini_client):
        mock_sync_models, _ = mock_gemini_client

        mock_response = create_mock_response(DEFAULT_MODEL_NAME, "Response from default model")
        mock_sync_models.generate_content.return_value = mock_response

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello")])
        gemini_llm.generate(prompt=prompt) # No model_name override

        mock_sync_models.generate_content.assert_called_once()
        call_args = mock_sync_models.generate_content.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_override(self, gemini_llm, mock_gemini_client):
        _, mock_async_models = mock_gemini_client

        mock_response = create_mock_response(OVERRIDE_MODEL_NAME, "Async response from override")
        mock_async_models.generate_content.return_value = mock_response # generate_content is async method on AsyncGenerativeModel

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello async")])
        await gemini_llm.generate_async(prompt=prompt, model_name=OVERRIDE_MODEL_NAME)

        mock_async_models.generate_content.assert_called_once()
        call_args = mock_async_models.generate_content.call_args
        assert call_args.kwargs["model"] == OVERRIDE_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_fallback(self, gemini_llm, mock_gemini_client):
        _, mock_async_models = mock_gemini_client

        mock_response = create_mock_response(DEFAULT_MODEL_NAME, "Async response from default")
        mock_async_models.generate_content.return_value = mock_response

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello async")])
        await gemini_llm.generate_async(prompt=prompt) # No model_name override

        mock_async_models.generate_content.assert_called_once()
        call_args = mock_async_models.generate_content.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL_NAME

```
