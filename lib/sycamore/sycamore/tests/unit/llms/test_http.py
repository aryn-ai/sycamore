import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from PIL import Image

from sycamore.llms.http import HttpLLM
from sycamore.llms.prompts.prompts import RenderedPrompt, RenderedMessage
from sycamore.utils.cache import Cache


class TestHttpLLM:

    def setup_method(self):
        self.base_url = "https://api.example.com"
        self.auth_token = "test-token-123"
        self.model_name = "test-model"

    def test_init_defaults(self):
        llm = HttpLLM(self.base_url, self.auth_token)

        assert llm.base_url == self.base_url
        assert llm.auth_token == self.auth_token
        assert llm._model_name == "gpt-4o-mini"
        assert llm.max_retries == 3
        assert llm.chat_completion_endpoint == "chatCompletion"

    def test_init_custom_parameters(self):
        cache = MagicMock(spec=Cache)
        custom_kwargs = {"temperature": 0.5}

        llm = HttpLLM(
            base_url=self.base_url,
            auth_token=self.auth_token,
            model_name=self.model_name,
            cache=cache,
            default_llm_kwargs=custom_kwargs,
            max_retries=5,
            chat_completion_endpoint="v2/chat",
        )

        assert llm.base_url == self.base_url
        assert llm.auth_token == self.auth_token
        assert llm._model_name == self.model_name
        assert llm.max_retries == 5
        assert llm.chat_completion_endpoint == "v2/chat"

    def test_base_url_trailing_slash_stripped(self):
        llm = HttpLLM("https://api.example.com/", self.auth_token)
        assert llm.base_url == "https://api.example.com"

    def test_is_chat_mode(self):
        llm = HttpLLM(self.base_url, self.auth_token)
        assert llm.is_chat_mode() is True

    def test_image_to_data_url(self):
        llm = HttpLLM(self.base_url, self.auth_token)

        image = Image.new("RGB", (10, 10), color="red")
        data_url = llm._image_to_data_url(image)

        assert data_url.startswith("data:image/jpeg;base64,")

    def test_format_image(self):
        llm = HttpLLM(self.base_url, self.auth_token)
        image = Image.new("RGB", (10, 10), color="blue")

        formatted = llm.format_image(image)

        assert formatted["type"] == "image_url"
        assert "image_url" in formatted
        assert "url" in formatted["image_url"]
        assert formatted["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_text_only_success(self, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": json.dumps({"content": "Test response"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Hello world")])

        result = await llm.generate_async(prompt=prompt)

        assert result == "Test response"

        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args

        assert call_args[1]["url"] == f"{self.base_url}/chatCompletion"
        assert call_args[1]["headers"]["Authorization"] == f"Bearer {self.auth_token}"

        payload = call_args[1]["json"]
        assert payload["engine"] == "gpt-4o-mini"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Hello world"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_with_images(self, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": json.dumps({"content": "Image analyzed"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        image = Image.new("RGB", (10, 10), color="green")
        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Describe this image", images=[image])])

        result = await llm.generate_async(prompt=prompt)

        assert result == "Image analyzed"

        call_args = mock_client_instance.post.call_args
        payload = call_args[1]["json"]

        user_message = payload["messages"][1]
        assert user_message["role"] == "user"
        assert isinstance(user_message["content"], list)

        content_parts = user_message["content"]
        assert len(content_parts) == 2
        assert content_parts[0]["type"] == "image_url"
        assert content_parts[1]["type"] == "text"
        assert content_parts[1]["text"] == "Describe this image"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_custom_endpoint(self, mock_client):
        custom_endpoint = "v2/custom-chat"
        llm = HttpLLM(self.base_url, self.auth_token, chat_completion_endpoint=custom_endpoint)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": json.dumps({"content": "Custom endpoint response"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        await llm.generate_async(prompt=prompt)

        call_args = mock_client_instance.post.call_args
        assert call_args[1]["url"] == f"{self.base_url}/{custom_endpoint}"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_http_error(self, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token, max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        with pytest.raises(ValueError, match="HTTP 500"):
            await llm.generate_async(prompt=prompt)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_invalid_response_format(self, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token, max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Missing result"}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        with pytest.raises(ValueError, match="Invalid response format"):
            await llm.generate_async(prompt=prompt)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_generate_async_retry_logic(self, mock_sleep, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token, max_retries=2)

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Server Error"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"result": json.dumps({"content": "Success after retry"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [mock_response_fail, mock_response_success]
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test retry")])

        result = await llm.generate_async(prompt=prompt)

        assert result == "Success after retry"
        assert mock_client_instance.post.call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_with_llm_kwargs(self, mock_client):
        llm = HttpLLM(self.base_url, self.auth_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": json.dumps({"content": "Response with custom kwargs"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        custom_kwargs = {"temperature": 0.8, "max_tokens": 1000}

        await llm.generate_async(prompt=prompt, llm_kwargs=custom_kwargs)

        call_args = mock_client_instance.post.call_args
        payload = call_args[1]["json"]

        assert payload["max_tokens"] == 1000
        assert payload["temperature"] == 0.8

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_with_cache_hit(self, mock_client):
        cache = MagicMock(spec=Cache)
        llm = HttpLLM(self.base_url, self.auth_token, cache=cache)

        llm._llm_cache_get = MagicMock(return_value="Cached response")

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        result = await llm.generate_async(prompt=prompt)

        assert result == "Cached response"
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_generate_async_with_cache_miss(self, mock_client):
        cache = MagicMock(spec=Cache)
        llm = HttpLLM(self.base_url, self.auth_token, cache=cache)

        llm._llm_cache_get = MagicMock(return_value=None)
        llm._llm_cache_set = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": json.dumps({"content": "Fresh response"})}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test")])

        result = await llm.generate_async(prompt=prompt)

        assert result == "Fresh response"
        llm._llm_cache_set.assert_called_once_with(prompt, None, "Fresh response")

    @patch.object(HttpLLM, "generate_async")
    def test_generate_sync_wrapper(self, mock_generate_async):
        llm = HttpLLM(self.base_url, self.auth_token)

        async def mock_async_generate(**kwargs):
            return "Sync result"

        mock_generate_async.side_effect = mock_async_generate

        prompt = RenderedPrompt(messages=[RenderedMessage(role="user", content="Test sync")])

        result = llm.generate(prompt=prompt)

        assert result == "Sync result"
        mock_generate_async.assert_called_once_with(prompt=prompt, llm_kwargs=None)

    @pytest.mark.asyncio
    async def test_generate_async_with_existing_system_message(self):
        llm = HttpLLM(self.base_url, self.auth_token)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": json.dumps({"content": "Response"})}

            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            prompt = RenderedPrompt(
                messages=[
                    RenderedMessage(role="system", content="Custom system prompt"),
                    RenderedMessage(role="user", content="User question"),
                ]
            )

            await llm.generate_async(prompt=prompt)

            call_args = mock_client_instance.post.call_args
            payload = call_args[1]["json"]

            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "Custom system prompt"
