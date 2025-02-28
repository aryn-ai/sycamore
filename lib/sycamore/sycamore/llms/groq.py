from enum import Enum
from sycamore.llms import LLM
from groq import Groq as GroqClient
from groq import AsyncGroq as AsyncGroqClient
from sycamore.llms.prompts.prompts import RenderedPrompt
from typing import Optional
import io
import json
import time
import logging


BATCH_POLL_INTERVAL = 10


class GroqModels(Enum):
    LLAMA_3_3_VERSATILE = "llama-3.3-70b-versatile"


class Groq(LLM):
    def __init__(self, model: GroqModels):
        self._model = model
        self._client = GroqClient()
        self._a_client = AsyncGroqClient()

    def __reduce__(self):
        def rebuild(kwargs):
            return Groq(**kwargs)

        return rebuild, ({"model": self._model},)

    def is_chat_mode(self) -> bool:
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        comp = self._client.chat.completions.create(messages=messages, model=self._model.value)
        return comp.choices[0].message.content

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        comp = await self._a_client.chat.completions.create(messages=messages, model=self._model.value)
        return comp.choices[0].message.content

    def generate_batch(self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None) -> list[str]:
        messageses = [[{"role": m.role, "content": m.content} for m in prompt.messages] for prompt in prompts]
        calls = [
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": self._model.value, "messages": m},
            }
            for i, m in enumerate(messageses)
        ]
        f = io.BytesIO()
        for i, c in enumerate(calls):
            f.write(json.dumps(c).encode("utf-8"))
            if i != len(calls) - 1:
                f.write(b"\n")
        file = self._client.files.create(file=f, purpose="batch")
        assert file.id is not None
        batch = self._client.batches.create(
            input_file_id=file.id, completion_window="24h", endpoint="/v1/chat/completions"
        )
        while batch.status in ("validating", "in_progress", "finalizing"):
            time.sleep(BATCH_POLL_INTERVAL)
            batch = self._client.batches.retrieve(batch.id)
        if batch.error_file_id:
            errors = self._client.files.content(batch.error_file_id)
            logging.error(errors)
            raise ValueError(f"LLM batch call failed: {batch}")
        if batch.output_file_id:
            results = [""] * len(calls)
            responses = self._client.files.content(batch.output_file_id)
            for rs, call in zip(responses.splitlines(), calls):
                rdata = json.loads(rs)
                id = int(rdata["custom_id"])
                results[id] = rdata["response"]["body"]["choices"][0]["message"]["content"]
        return results
