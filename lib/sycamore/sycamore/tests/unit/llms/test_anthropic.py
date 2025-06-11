import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import datetime

try:
    from anthropic.types import Message, Usage, ContentBlock, BatchResponse, MessageStreamEvent, BatchRequest
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
except ImportError:
    # Dummy types if anthropic is not installed, for basic testing structure
    class Message: pass
    class Usage: pass
    class ContentBlock: pass
    class BatchResponse: pass
    class MessageStreamEvent: pass # Not directly used in these tests but good for completeness
    class BatchRequest: pass
    class MessageCreateParamsNonStreaming: pass


from sycamore.llms.anthropic import Anthropic, AnthropicModels
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage

DEFAULT_ANTHROPIC_MODEL = AnthropicModels.CLAUDE_3_HAIKU
DEFAULT_ANTHROPIC_MODEL_NAME = DEFAULT_ANTHROPIC_MODEL.value
OVERRIDE_ANTHROPIC_MODEL_NAME = "claude-3-sonnet-20240229" # Example override


@pytest.fixture
def mock_anthropic_clients():
    """Mocks anthropic.Anthropic and anthropic.AsyncAnthropic clients."""
    with patch("anthropic.Anthropic") as MockAnthropic, \
         patch("anthropic.AsyncAnthropic") as MockAsyncAnthropic:

        mock_sync_client = MockAnthropic.return_value
        mock_async_client = MockAsyncAnthropic.return_value

        yield mock_sync_client, mock_async_client

@pytest.fixture
def anthropic_llm(mock_anthropic_clients): # Depends on mock_anthropic_clients
    # api_key is not strictly needed due to mocking, but good practice
    return Anthropic(model_name=DEFAULT_ANTHROPIC_MODEL, api_key="dummy_key")


def create_mock_anthropic_message(model_name_used: str, text_content: str = "Test response") -> Message:
    """Helper to create a mock anthropic.types.Message."""
    usage = Usage(input_tokens=10, output_tokens=5)
    content_block = ContentBlock(type="text", text=text_content)
    # Ensure model attribute exists on the Message object if the SDK sets it.
    # Based on current SDK, model is part of the request, not typically on response Message object itself.
    # The assertion is on the model passed to create() method.
    return Message(
        id="msg_xxxx",
        type="message",
        role="assistant",
        content=[content_block],
        model=model_name_used, # Anthropic SDK Message object includes model
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )

def create_mock_anthropic_batch_response(status="completed", num_requests=1, model_name_used=DEFAULT_ANTHROPIC_MODEL_NAME) -> BatchResponse:
    """Helper to create a mock anthropic.types.BatchResponse for retrieve."""
    mock_batch = MagicMock(spec=BatchResponse)
    mock_batch.id = "batch_mock_id"
    mock_batch.status = status
    mock_batch.processing_status = status # For simplicity, map status to processing_status
    mock_batch.request_counts.total = num_requests
    # Other attributes like created_at, completed_at, etc., can be added if needed
    return mock_batch

def create_mock_anthropic_batch_results(num_results=1, model_name_used=DEFAULT_ANTHROPIC_MODEL_NAME):
    """Helper to create mock results for client.messages.batches.results()."""
    results = []
    for i in range(num_results):
        mock_result = MagicMock() # Individual result item in the stream/list
        mock_result.custom_id = str(i)

        succeeded_result_data = MagicMock()
        succeeded_result_data.type = "succeeded"
        succeeded_result_data.message = create_mock_anthropic_message(
            model_name_used, f"Batch response for {i}"
        )
        mock_result.result = succeeded_result_data
        results.append(mock_result)
    return results


class TestAnthropicModelOverride:
    def test_generate_model_override(self, anthropic_llm, mock_anthropic_clients):
        mock_sync_client, _ = mock_anthropic_clients

        mock_response_message = create_mock_anthropic_message(OVERRIDE_ANTHROPIC_MODEL_NAME)
        mock_sync_client.messages.create.return_value = mock_response_message

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello")])
        anthropic_llm.generate(prompt=prompt, model_name=OVERRIDE_ANTHROPIC_MODEL_NAME)

        mock_sync_client.messages.create.assert_called_once()
        call_args = mock_sync_client.messages.create.call_args
        assert call_args.kwargs["model"] == OVERRIDE_ANTHROPIC_MODEL_NAME

    def test_generate_model_fallback(self, anthropic_llm, mock_anthropic_clients):
        mock_sync_client, _ = mock_anthropic_clients

        mock_response_message = create_mock_anthropic_message(DEFAULT_ANTHROPIC_MODEL_NAME)
        mock_sync_client.messages.create.return_value = mock_response_message

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello")])
        anthropic_llm.generate(prompt=prompt) # No model_name override

        mock_sync_client.messages.create.assert_called_once()
        call_args = mock_sync_client.messages.create.call_args
        assert call_args.kwargs["model"] == DEFAULT_ANTHROPIC_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_override(self, anthropic_llm, mock_anthropic_clients):
        _, mock_async_client = mock_anthropic_clients

        mock_response_message = create_mock_anthropic_message(OVERRIDE_ANTHROPIC_MODEL_NAME)
        mock_async_client.messages.create.return_value = mock_response_message

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello async")])
        await anthropic_llm.generate_async(prompt=prompt, model_name=OVERRIDE_ANTHROPIC_MODEL_NAME)

        mock_async_client.messages.create.assert_called_once()
        call_args = mock_async_client.messages.create.call_args
        assert call_args.kwargs["model"] == OVERRIDE_ANTHROPIC_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_fallback(self, anthropic_llm, mock_anthropic_clients):
        _, mock_async_client = mock_anthropic_clients

        mock_response_message = create_mock_anthropic_message(DEFAULT_ANTHROPIC_MODEL_NAME)
        mock_async_client.messages.create.return_value = mock_response_message

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello async")])
        await anthropic_llm.generate_async(prompt=prompt) # No model_name override

        mock_async_client.messages.create.assert_called_once()
        call_args = mock_async_client.messages.create.call_args
        assert call_args.kwargs["model"] == DEFAULT_ANTHROPIC_MODEL_NAME

    def test_generate_batch_model_override(self, anthropic_llm, mock_anthropic_clients):
        mock_sync_client, _ = mock_anthropic_clients

        mock_batch_creation_response = create_mock_anthropic_batch_response(status="creating", num_requests=1)
        mock_batch_completed_response = create_mock_anthropic_batch_response(status="completed", num_requests=1)

        mock_sync_client.messages.batches.create.return_value = mock_batch_creation_response
        # Simulate polling: first retrieve returns "in_progress", second "completed"
        mock_sync_client.messages.batches.retrieve.side_effect = [
            create_mock_anthropic_batch_response(status="in_progress", num_requests=1),
            mock_batch_completed_response
        ]
        mock_sync_client.messages.batches.results.return_value = create_mock_anthropic_batch_results(
            num_results=1, model_name_used=OVERRIDE_ANTHROPIC_MODEL_NAME
        )

        prompts = [RenderedPrompt(messages=[RenderedMessage(role="user", content="Batch hello 1")])]
        anthropic_llm.generate_batch(prompts=prompts, model_name=OVERRIDE_ANTHROPIC_MODEL_NAME)

        mock_sync_client.messages.batches.create.assert_called_once()
        # Check the model in the requests passed to batches.create
        # requests is a list of Request objects, each having params of type MessageCreateParamsNonStreaming
        batch_create_call_args = mock_sync_client.messages.batches.create.call_args
        assert len(batch_create_call_args.kwargs["requests"]) == 1
        request_param = batch_create_call_args.kwargs["requests"][0]
        assert isinstance(request_param, BatchRequest) # Should be BatchRequest containing MessageCreateParams
        assert request_param.params["model"] == OVERRIDE_ANTHROPIC_MODEL_NAME


    def test_generate_batch_model_fallback(self, anthropic_llm, mock_anthropic_clients):
        mock_sync_client, _ = mock_anthropic_clients

        mock_batch_creation_response = create_mock_anthropic_batch_response(status="creating", num_requests=1)
        mock_batch_completed_response = create_mock_anthropic_batch_response(status="completed", num_requests=1)

        mock_sync_client.messages.batches.create.return_value = mock_batch_creation_response
        mock_sync_client.messages.batches.retrieve.side_effect = [
            create_mock_anthropic_batch_response(status="in_progress", num_requests=1),
            mock_batch_completed_response
        ]
        mock_sync_client.messages.batches.results.return_value = create_mock_anthropic_batch_results(
            num_results=1, model_name_used=DEFAULT_ANTHROPIC_MODEL_NAME
        )

        prompts = [RenderedPrompt(messages=[RenderedMessage(role="user", content="Batch hello 2")])]
        anthropic_llm.generate_batch(prompts=prompts) # No model_name override

        mock_sync_client.messages.batches.create.assert_called_once()
        batch_create_call_args = mock_sync_client.messages.batches.create.call_args
        assert len(batch_create_call_args.kwargs["requests"]) == 1
        request_param = batch_create_call_args.kwargs["requests"][0]
        assert isinstance(request_param, BatchRequest)
        assert request_param.params["model"] == DEFAULT_ANTHROPIC_MODEL_NAME

```
