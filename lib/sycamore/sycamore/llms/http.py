import asyncio
import base64
import io
import json
import logging
import random
from typing import Optional, Any, Union
from PIL import Image
import httpx

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.llms.config import LLMModel
from sycamore.utils.cache import Cache


class HttpLLM(LLM):

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        model_name: str = "gpt-4o-mini",
        cache: Optional[Cache] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
        max_retries: int = 3,
        chat_completion_endpoint: str = "chatCompletion",
    ):
        super().__init__(
            model_name=model_name,
            default_mode=LLMMode.ASYNC,
            cache=cache,
            default_llm_kwargs=default_llm_kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.chat_completion_endpoint = chat_completion_endpoint

    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        return asyncio.run(self.generate_async(prompt=prompt, llm_kwargs=llm_kwargs, model=model))

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        # Check cache first
        cached_result = self._llm_cache_get(prompt, llm_kwargs)
        if cached_result is not None:
            return cached_result

        merged_kwargs = self._merge_llm_kwargs(llm_kwargs)

        # Prepare headers
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        messages: list[dict[str, Union[str, list[dict[str, Any]]]]] = []
        for message in prompt.messages:
            if message.images:
                # Handle messages with images
                content_parts: list[dict[str, Any]] = []
                # Add images first
                for image in message.images:
                    content_parts.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}})
                # Add text content
                content_parts.append({"type": "text", "text": message.content})
                messages.append({"role": message.role, "content": content_parts})
            else:
                # Text-only message
                messages.append({"role": message.role, "content": message.content})

        # Add system message if not present
        if not messages or messages[0]["role"] != "system":
            system_msg: dict[str, Union[str, list[dict[str, Any]]]] = {
                "role": "system",
                "content": "You are a helpful AI assistant. Analyze the provided image and text, and respond according to the user's request.",
            }
            messages.insert(0, system_msg)

        payload = {
            "engine": self._model_name,
            "messages": messages,
            "max_tokens": merged_kwargs.get("max_output_tokens", merged_kwargs.get("max_tokens", 800)),
            "temperature": merged_kwargs.get("temperature", 0),
        }

        # Retry logic with exponential backoff
        for retry in range(self.max_retries):
            jitter = random.uniform(0.5, 1.5)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url=f"{self.base_url}/{self.chat_completion_endpoint}",
                        json=payload,
                        headers=headers,
                        timeout=60.0,
                    )

                    if response.status_code != 200:
                        raise ValueError(f"HTTP {response.status_code}: {response.text}")

                    response_data = response.json()
                    if "result" not in response_data:
                        raise ValueError(f"Invalid response format: {response_data}")

                    result = json.loads(response_data["result"])
                    content = result.get("content", "")

                    # Cache the result
                    self._llm_cache_set(prompt, llm_kwargs, content)
                    return content

            except Exception as e:
                if retry >= self.max_retries - 1:
                    logging.error(f"HTTP LLM request failed after {self.max_retries} retries: {e}")
                    raise
                await asyncio.sleep(2**retry * jitter)

        return ""

    def _image_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        jpeg_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{jpeg_data}"

    def is_chat_mode(self) -> bool:
        return True

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}}
