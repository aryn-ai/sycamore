import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion import Completion
from openai.types.completion_usage import CompletionUsage
from openai.types.file_object import FileObject
from openai.types.batch import Batch

from sycamore.llms.openai import OpenAI, OpenAIModels, OpenAIClientWrapper, OpenAIClientType
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage

DEFAULT_MODEL_NAME = OpenAIModels.GPT_3_5_TURBO.value.name
OVERRIDE_MODEL_NAME = OpenAIModels.GPT_4_TURBO_PREVIEW.value.name

@pytest.fixture
def openai_llm():
    # Using a dummy API key as we are mocking the client calls
    return OpenAI(model_name=DEFAULT_MODEL_NAME, api_key="test_key")

@pytest.fixture
def mock_openai_client_wrapper():
    with patch("sycamore.llms.openai.OpenAIClientWrapper") as MockWrapper:
        mock_wrapper_instance = MockWrapper.return_value

        # Sync client mock
        mock_sync_client = MagicMock()
        mock_wrapper_instance.get_client.return_value = mock_sync_client

        # Async client mock
        mock_async_client = AsyncMock()
        mock_wrapper_instance.get_async_client.return_value = mock_async_client

        yield mock_wrapper_instance, mock_sync_client, mock_async_client


class TestOpenAIModelOverride:
    def test_generate_model_override(self, openai_llm, mock_openai_client_wrapper):
        _, mock_sync_client, _ = mock_openai_client_wrapper

        mock_chat_completion = ChatCompletion(
            id="chatcmpl-xxxx",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": ChatCompletionMessage(
                        content="Hello from override model", role="assistant", function_call=None, tool_calls=None
                    ),
                    "logprobs": None,
                }
            ],
            created=12345,
            model=OVERRIDE_MODEL_NAME,
            object="chat.completion",
            system_fingerprint="fp_xxxx",
            usage=CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10),
        )
        mock_sync_client.chat.completions.create.return_value = mock_chat_completion

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hi")])
        openai_llm.generate(prompt=prompt, model_name=OVERRIDE_MODEL_NAME)

        mock_sync_client.chat.completions.create.assert_called_once()
        call_args = mock_sync_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == OVERRIDE_MODEL_NAME

    def test_generate_model_fallback(self, openai_llm, mock_openai_client_wrapper):
        _, mock_sync_client, _ = mock_openai_client_wrapper

        mock_chat_completion = ChatCompletion(
            id="chatcmpl-yyyy",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": ChatCompletionMessage(
                        content="Hello from default model", role="assistant", function_call=None, tool_calls=None
                    ),
                    "logprobs": None,
                }
            ],
            created=12345,
            model=DEFAULT_MODEL_NAME,
            object="chat.completion",
            system_fingerprint="fp_yyyy",
            usage=CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10),
        )
        mock_sync_client.chat.completions.create.return_value = mock_chat_completion

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hi")])
        openai_llm.generate(prompt=prompt) # No model_name override

        mock_sync_client.chat.completions.create.assert_called_once()
        call_args = mock_sync_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_override(self, openai_llm, mock_openai_client_wrapper):
        _, _, mock_async_client = mock_openai_client_wrapper

        mock_chat_completion = ChatCompletion(
            id="chatcmpl-zzzz",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": ChatCompletionMessage(
                        content="Async hello from override", role="assistant", function_call=None, tool_calls=None
                    ),
                    "logprobs": None,
                }
            ],
            created=12345,
            model=OVERRIDE_MODEL_NAME,
            object="chat.completion",
            system_fingerprint="fp_zzzz",
            usage=CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10),
        )
        mock_async_client.chat.completions.create.return_value = mock_chat_completion

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hi async")])
        await openai_llm.generate_async(prompt=prompt, model_name=OVERRIDE_MODEL_NAME)

        mock_async_client.chat.completions.create.assert_called_once()
        call_args = mock_async_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == OVERRIDE_MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_async_model_fallback(self, openai_llm, mock_openai_client_wrapper):
        _, _, mock_async_client = mock_openai_client_wrapper

        mock_chat_completion = ChatCompletion(
            id="chatcmpl-aaaa",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": ChatCompletionMessage(
                        content="Async hello from default", role="assistant", function_call=None, tool_calls=None
                    ),
                    "logprobs": None,
                }
            ],
            created=12345,
            model=DEFAULT_MODEL_NAME,
            object="chat.completion",
            system_fingerprint="fp_aaaa",
            usage=CompletionUsage(completion_tokens=5, prompt_tokens=5, total_tokens=10),
        )
        mock_async_client.chat.completions.create.return_value = mock_chat_completion

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hi async")])
        await openai_llm.generate_async(prompt=prompt) # No model_name override

        mock_async_client.chat.completions.create.assert_called_once()
        call_args = mock_async_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL_NAME

    def test_generate_batch_model_override(self, openai_llm, mock_openai_client_wrapper):
        _, mock_sync_client, _ = mock_openai_client_wrapper

        # Mock file creation
        mock_file_object = FileObject(id="file-xxxx", bytes=123, created_at=12345, filename="batch.jsonl", object="file", purpose="batch")
        mock_sync_client.files.create.return_value = mock_file_object

        # Mock batch creation and retrieval
        mock_batch_object = Batch(
            id="batch_override_xxxx",
            object="batch",
            endpoint="/v1/chat/completions",
            status="completed",
            input_file_id="file-xxxx",
            completion_window="24h",
            created_at=12345,
            output_file_id="file-yyyy",
            error_file_id=None
        )
        mock_sync_client.batches.create.return_value = mock_batch_object
        mock_sync_client.batches.retrieve.return_value = mock_batch_object

        # Mock file content (output from batch)
        # Construct a mock response similar to what client.files.content would return for batch output
        response_line = {
            "custom_id": "0",
            "response": {
                "body": {
                    "id": "chatcmpl-batch-xxxx",
                    "object": "chat.completion",
                    "created": 12345,
                    "model": OVERRIDE_MODEL_NAME, # This is what we want to check in the output data
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Batch response override"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }
            }
        }
        import json
        mock_file_content = MagicMock()
        mock_file_content.iter_lines.return_value = [json.dumps(response_line).encode('utf-8')]
        mock_sync_client.files.content.return_value = mock_file_content


        prompts = [RenderedPrompt(messages=[RenderedMessage(role="user", content="Batch hi 1")])]
        openai_llm.generate_batch(prompts=prompts, model_name=OVERRIDE_MODEL_NAME)

        mock_sync_client.files.create.assert_called_once()
        # Check that the model in the body of the request sent to files.create (which becomes input to batches.create) is correct
        # The input to files.create is a file-like object, so we need to get the bytes that were written to it.
        file_creation_call_args = mock_sync_client.files.create.call_args
        file_content_bytes = file_creation_call_args.kwargs['file'].getvalue()
        file_content_str = file_content_bytes.decode('utf-8')
        batch_request_line = json.loads(file_content_str) # Assuming one line for one prompt

        assert "body" in batch_request_line
        assert batch_request_line["body"]["model"] == OVERRIDE_MODEL_NAME
        mock_sync_client.batches.create.assert_called_once()
        # We can also check the model in the output processing if needed, as done by asserting response_line["response"]["body"]["model"]

    def test_generate_batch_model_fallback(self, openai_llm, mock_openai_client_wrapper):
        _, mock_sync_client, _ = mock_openai_client_wrapper

        mock_file_object = FileObject(id="file-aaaa", bytes=123, created_at=12345, filename="batch.jsonl", object="file", purpose="batch")
        mock_sync_client.files.create.return_value = mock_file_object

        mock_batch_object = Batch(
            id="batch_fallback_bbbb",
            object="batch",
            endpoint="/v1/chat/completions",
            status="completed",
            input_file_id="file-aaaa",
            completion_window="24h",
            created_at=12345,
            output_file_id="file-bbbb",
            error_file_id=None
        )
        mock_sync_client.batches.create.return_value = mock_batch_object
        mock_sync_client.batches.retrieve.return_value = mock_batch_object

        response_line = {
            "custom_id": "0",
            "response": {
                "body": {
                    "id": "chatcmpl-batch-yyyy",
                    "object": "chat.completion",
                    "created": 12345,
                    "model": DEFAULT_MODEL_NAME, # Fallback model
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Batch response fallback"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }
            }
        }
        import json
        mock_file_content = MagicMock()
        mock_file_content.iter_lines.return_value = [json.dumps(response_line).encode('utf-8')]
        mock_sync_client.files.content.return_value = mock_file_content

        prompts = [RenderedPrompt(messages=[RenderedMessage(role="user", content="Batch hi 2")])]
        openai_llm.generate_batch(prompts=prompts) # No model_name override

        mock_sync_client.files.create.assert_called_once()
        file_creation_call_args = mock_sync_client.files.create.call_args
        file_content_bytes = file_creation_call_args.kwargs['file'].getvalue()
        file_content_str = file_content_bytes.decode('utf-8')
        batch_request_line = json.loads(file_content_str)

        assert "body" in batch_request_line
        assert batch_request_line["body"]["model"] == DEFAULT_MODEL_NAME
        mock_sync_client.batches.create.assert_called_once()

```
