import functools
import inspect
import logging
import os
from enum import Enum
from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import asyncio
import random
import json
import io
import time

from openai import AzureOpenAI as AzureOpenAIClient
from openai import AsyncAzureOpenAI as AsyncAzureOpenAIClient
from openai import OpenAI as OpenAIClient
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import max_retries as DEFAULT_MAX_RETRIES
from openai.lib.azure import AzureADTokenProvider
from openai.lib._parsing import type_to_response_format_param
from openai import APIConnectionError
from openai.types.chat.chat_completion import ChatCompletion

import pydantic

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.config import OpenAIModel, OpenAIModels
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import base64_data_url

logger = logging.getLogger(__name__)


# Base URL for Helicone API, if configured using the SYCAMORE_HELICONE_API_KEY environment variable.
HELICONE_BASE_URL = "https://oai.helicone.ai/v1"
INITIAL_BACKOFF = 0.2
BATCH_POLL_INTERVAL = 10


class OpenAIClientType(Enum):
    OPENAI = 0
    AZURE = 1


class OpenAIClientWrapper:
    def __init__(
        self,
        client_type: OpenAIClientType = OpenAIClientType.OPENAI,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        azure_deployment: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        disable_helicone: Optional[bool] = None,
        echo: bool = False,
        # Deprecated names that we support for backwards compatibility.
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        # Additional OpenAI Client parameters that will be passed in.
        **kwargs,
    ):
        if api_type is not None:
            logger.warning("WARNING: The api_type parameter is deprecated. Please use client_type instead.")
            if api_type in {"azure", "azure_ad", "azuread"}:
                client_type = OpenAIClientType.AZURE
            else:
                client_type = OpenAIClientType.OPENAI
        if api_base is not None:
            logger.warning(
                "WARNING: The api_base parameter is deprecated. Please use base_url or azure_endpoint instead."
            )
            if azure_endpoint is None:
                azure_endpoint = api_base
            else:
                raise ValueError("Can't set both api_base and azure_endpoint")
        # TODO: Add some parameter validation so we can fail fast. The openai library has a bunch of validation,
        # but that may not happen until much later in the job execution.

        if client_type == OpenAIClientType.AZURE:
            if azure_endpoint is None:
                raise ValueError("azure_endpoint must be specified for Azure clients.")
            if api_version is None:
                raise ValueError("api_version must be specified for Azure clients.")

        self.client_type = client_type
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.max_retries = max_retries
        self.azure_deployment = azure_deployment
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token
        self.azure_ad_token_provider = azure_ad_token_provider
        self.disable_helicone = disable_helicone
        self.extra_kwargs = kwargs
        self.echo = echo

        # The OpenAI Python library is happy to pull Azure creds from the AZURE_OPENAI_API_KEY environment variable,
        # but Guidance will error out if neither api_key nor azure_ad_token_provider are explicitly set.
        if client_type == OpenAIClientType.AZURE and api_key is None and azure_ad_token_provider is None:
            if "AZURE_OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "Must set either api_key, azure_ad_token_provider, or AZURE_OPENAI_API_KEY environment variable."
                )

            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    @functools.cache
    def get_client(self) -> OpenAIClient:
        if self.client_type == OpenAIClientType.OPENAI:
            # We currently only support Helicone with OpenAI.
            base_url = self.base_url
            extra_kwargs = self.extra_kwargs
            if "SYCAMORE_HELICONE_API_KEY" in os.environ and self.disable_helicone is not True:
                if self.base_url is not None:
                    logging.warning("SYCAMORE_HELICONE_API_KEY found in environment. Ignoring base_url.")
                base_url = HELICONE_BASE_URL
                if "default_headers" not in extra_kwargs:
                    extra_kwargs["default_headers"] = {}
                extra_kwargs["default_headers"].update(
                    {"Helicone-Auth": f"Bearer {os.environ['SYCAMORE_HELICONE_API_KEY']}"}
                )
                # Add SYCAMORE_HELICONE_TAG value to the Helicone-Property-Tag header if it is set.
                if "SYCAMORE_HELICONE_TAG" in os.environ:
                    extra_kwargs["default_headers"].update(
                        {"Helicone-Property-Tag": os.environ["SYCAMORE_HELICONE_TAG"]}
                    )
            return OpenAIClient(
                api_key=self.api_key,
                organization=self.organization,
                base_url=base_url,
                max_retries=self.max_retries,
                **extra_kwargs,
            )
        elif self.client_type == OpenAIClientType.AZURE:
            return AzureOpenAIClient(
                azure_endpoint=str(self.azure_endpoint),
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
                api_key=self.api_key,
                azure_ad_token=self.azure_ad_token,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                max_retries=self.max_retries,
                **self.extra_kwargs,
            )
        else:
            raise ValueError(f"Invalid client_type {self.client_type}")

    def close(self) -> None:
        # This is tricky.  We want to close the client, but avoid creating one
        # if there isn't one cached.  We can't close the async client from
        # a non-async context.  Attempts to use clients after calling close()
        # will fail.
        if self.get_client.cache_info().currsize:
            self.get_client().close()

    @functools.cache
    def get_async_client(self) -> AsyncOpenAIClient:
        if self.client_type == OpenAIClientType.OPENAI:
            # We currently only support Helicone with OpenAI.
            base_url = self.base_url
            extra_kwargs = self.extra_kwargs
            if "SYCAMORE_HELICONE_API_KEY" in os.environ and self.disable_helicone is not True:
                if self.base_url is not None:
                    logging.warning("SYCAMORE_HELICONE_API_KEY found in environment. Ignoring base_url.")
                base_url = HELICONE_BASE_URL
                if "default_headers" not in extra_kwargs:
                    extra_kwargs["default_headers"] = {}
                extra_kwargs["default_headers"].update(
                    {"Helicone-Auth": f"Bearer {os.environ['SYCAMORE_HELICONE_API_KEY']}"}
                )
                # Add SYCAMORE_HELICONE_TAG value to the Helicone-Property-Tag header if it is set.
                if "SYCAMORE_HELICONE_TAG" in os.environ:
                    extra_kwargs["default_headers"].update(
                        {"Helicone-Property-Tag": os.environ["SYCAMORE_HELICONE_TAG"]}
                    )
            return AsyncOpenAIClient(
                api_key=self.api_key,
                organization=self.organization,
                base_url=base_url,
                max_retries=self.max_retries,
                **extra_kwargs,
            )
        elif self.client_type == OpenAIClientType.AZURE:
            return AsyncAzureOpenAIClient(
                azure_endpoint=str(self.azure_endpoint),
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
                api_key=self.api_key,
                azure_ad_token=self.azure_ad_token,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                max_retries=self.max_retries,
                **self.extra_kwargs,
            )
        else:
            raise ValueError(f"Invalid client_type {self.client_type}")


# Allow rough backwards compatibility
OpenAIClientParameters = OpenAIClientWrapper


def openai_deserializer(kwargs):
    return OpenAI(**kwargs)


class OpenAI(LLM):
    """An LLM interface to OpenAI models.

    Args:
        model_name: The name of the OpenAI model to use. This can be an instance of OpenAIModels, an instance of
            OpenAIModel, or a string. If a string is provided, it must be the name of the model.
        api_key: The API key to use for the OpenAI client. If not provided, the key will be read from the
            OPENAI_API_KEY environment variable.
        client_wrapper: An instance of OpenAIClientWrapper to use for the OpenAI client. If not provided, a new
            instance will be created using the provided parameters.
        params: An instance of OpenAIClientParameters to use for the OpenAI client. If not provided, a new instance
            will be created using the provided parameters.
        cache: An instance of Cache to use for caching responses. If not provided, no caching will be used.
        default_mode: Default execution mode for the llm
        **kwargs: Additional parameters to pass to the OpenAI client.
    """

    def __init__(
        self,
        model_name: Union[OpenAIModels, OpenAIModel, str],
        api_key: Optional[str] = None,
        client_wrapper: Optional[OpenAIClientWrapper] = None,
        params: Optional[OpenAIClientParameters] = None,
        default_mode: LLMMode = LLMMode.ASYNC,
        cache: Optional[Cache] = None,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if isinstance(model_name, OpenAIModels):
            self.model = model_name.value
        elif isinstance(model_name, OpenAIModel):
            self.model = model_name
        elif isinstance(model_name, str):
            self.model = OpenAIModels.from_name(model_name).value
        else:
            raise TypeError("model_name must be an instance of str, OpenAIModel, or OpenAIModels")

        if self.model.name == OpenAIModels.TEXT_DAVINCI.value.name:
            logger.warning("text-davinci-003 is deprecated. Falling back to gpt-3.5-turbo-instruct")
            self.model = OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value
        super().__init__(self.model.name, default_mode, cache, default_llm_kwargs=default_llm_kwargs)

        # This is somewhat complex to provide a degree of backward compatibility.
        if client_wrapper is None:
            if params is not None:
                client_wrapper = params
            else:
                if api_key is not None:
                    kwargs.update({"api_key": api_key})

                client_wrapper = OpenAIClientWrapper(**kwargs)

        else:
            if api_key is not None:
                client_wrapper.api_key = api_key

        self.client_wrapper = client_wrapper

    # The actual openai client is not pickleable, This just says to pickle the wrapper, which can be used to
    # recreate the client on the other end.
    def __reduce__(self):

        kwargs = {
            "client_wrapper": self.client_wrapper,
            "model_name": self.model,
            "cache": self._cache,
            "default_mode": self._default_mode,
            "default_llm_kwargs": self._default_llm_kwargs,
        }

        return openai_deserializer, (kwargs,)

    def close(self) -> None:
        # After closing, don't expect method calls to succeed.
        self.client_wrapper.close()

    def is_chat_mode(self):
        return self.model.is_chat

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": base64_data_url(image)}}

    def validate_tokens(self, completion) -> Tuple[int, int]:
        if completion.usage is not None:
            completion_tokens = completion.usage.completion_tokens or 0
            prompt_tokens = completion.usage.prompt_tokens or 0
        else:
            completion_tokens = 0
            prompt_tokens = 0
        return completion_tokens, prompt_tokens

    def _convert_response_format(self, llm_kwargs: Optional[Dict]) -> Optional[Dict]:
        """Convert the response_format parameter to the appropriate OpenAI format."""
        if llm_kwargs is None:
            return None
        response_format = llm_kwargs.get("response_format")
        if response_format is None:
            return llm_kwargs
        if inspect.isclass(response_format) and issubclass(response_format, pydantic.BaseModel):
            retval = llm_kwargs.copy()
            retval["response_format"] = type_to_response_format_param(response_format)
            return retval
        return llm_kwargs

    def _get_generate_kwargs(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> dict:
        kwargs = {
            **(llm_kwargs or {}),
        }

        if not self.model.name.startswith("o"):
            kwargs["temperature"] = 0

        if "SYCAMORE_OPENAI_USER" in os.environ:
            kwargs.update({"user": os.environ.get("SYCAMORE_OPENAI_USER")})

        if not self.is_chat_mode():
            # If plain completions model, we want a 'prompt' arg with a
            # single string in it, not a list of messages. Make it by
            # concatenating the messages.
            pstring = "\n".join(m.content for m in prompt.messages)
            kwargs.update({"prompt": pstring})
            return kwargs

        messages_list = []
        for m in prompt.messages:
            if m.role == "system":
                # OpenAI docs say "developer" is the new "system"
                # but Azure don't like that
                role = "system"
            else:
                role = m.role
            if m.images is None:
                content: Union[str, list] = m.content
            else:
                content = [{"type": "text", "text": m.content}]
                assert isinstance(content, list)  # mypy!!!
                for im in m.images:
                    content.append(self.format_image(im))
            messages_list.append({"role": role, "content": content})

        kwargs.update({"messages": messages_list})
        return kwargs

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        llm_kwargs = self._convert_response_format(llm_kwargs)
        ret = self._llm_cache_get(prompt, llm_kwargs)
        if ret is not None:
            return ret

        if prompt.response_format is not None:
            ret = self._generate_using_openai_structured(prompt, llm_kwargs)
        else:
            ret = self._generate_using_openai(prompt, llm_kwargs)

        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    def _generate_using_openai(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict]) -> str:
        kwargs = self._get_generate_kwargs(prompt, llm_kwargs)
        logging.debug("OpenAI prompt: %s", kwargs)
        if self.is_chat_mode():
            starttime = datetime.now()
            completion = self.client_wrapper.get_client().chat.completions.create(model=self._model_name, **kwargs)
            logging.debug("OpenAI completion: %s", completion)
            wall_latency = datetime.now() - starttime
            response_text = completion.choices[0].message.content
        else:
            starttime = datetime.now()
            completion = self.client_wrapper.get_client().completions.create(model=self._model_name, **kwargs)
            logging.debug("OpenAI completion: %s", completion)
            wall_latency = datetime.now() - starttime
            response_text = completion.choices[0].text

        completion_tokens, prompt_tokens = self.validate_tokens(completion)
        self.add_llm_metadata(kwargs, response_text, wall_latency, completion_tokens, prompt_tokens)
        if not response_text:
            raise ValueError("OpenAI returned empty response")
        return response_text

    def _generate_using_openai_structured(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict]) -> str:
        try:
            kwargs = self._get_generate_kwargs(prompt, llm_kwargs)
            if self.is_chat_mode():
                starttime = datetime.now()
                completion = self.client_wrapper.get_client().beta.chat.completions.parse(
                    model=self._model_name, **kwargs
                )
                completion_tokens, prompt_tokens = self.validate_tokens(completion)
                wall_latency = datetime.now() - starttime
                response_text = completion.choices[0].message.content
                self.add_llm_metadata(kwargs, response_text, wall_latency, completion_tokens, prompt_tokens)
            else:
                raise ValueError("This method doesn't support instruct models. Please use a chat model.")
                # completion = self.client_wrapper.get_client().beta.completions.parse(model=self._model_name, **kwargs)
            assert response_text is not None, "OpenAI refused to respond to the query"
            return response_text
        except Exception as e:
            # OpenAI will not respond in two scenarios:
            # 1.) The LLM ran out of output context length(usually do to hallucination of repeating the same phrase)
            # 2.) The LLM refused to respond to the request because it did not meet guidelines
            raise e

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        ret = self._llm_cache_get(prompt, llm_kwargs)
        if ret is not None:
            return ret

        if llm_kwargs is None:
            raise ValueError("Must include llm_kwargs to generate future call")

        done = False
        retries = 0
        while not done:
            try:
                if prompt.response_format is not None:
                    ret = await self._generate_awaitable_using_openai_structured(prompt, llm_kwargs)
                else:
                    ret = await self._generate_awaitable_using_openai(prompt, llm_kwargs)
                done = True
            except APIConnectionError:
                backoff = INITIAL_BACKOFF * (2**retries)
                jitter = random.uniform(0, 0.1 * backoff)
                await asyncio.sleep(backoff + jitter)
                retries += 1

        self._llm_cache_set(prompt, llm_kwargs, ret)
        return ret

    async def _generate_awaitable_using_openai(self, prompt: RenderedPrompt, llm_kwargs: Optional[dict]) -> str:
        kwargs = self._get_generate_kwargs(prompt, llm_kwargs)
        starttime = datetime.now()
        if self.is_chat_mode():
            completion = await self.client_wrapper.get_async_client().chat.completions.create(
                model=self._model_name, **kwargs
            )
            response_text = completion.choices[0].message.content
            wall_latency = datetime.now() - starttime
        else:
            completion = await self.client_wrapper.get_async_client().completions.create(
                model=self._model_name, **kwargs
            )
            response_text = completion.choices[0].text
            wall_latency = datetime.now() - starttime
            response_text = completion.choices[0].message.content

        if completion.usage is not None:
            completion_tokens = completion.usage.completion_tokens or 0
            prompt_tokens = completion.usage.prompt_tokens or 0
        else:
            completion_tokens = 0
            prompt_tokens = 0

        self.add_llm_metadata(kwargs, response_text, wall_latency, completion_tokens, prompt_tokens)
        return response_text

    async def _generate_awaitable_using_openai_structured(
        self, prompt: RenderedPrompt, llm_kwargs: Optional[dict]
    ) -> str:
        try:
            kwargs = self._get_generate_kwargs(prompt, llm_kwargs)
            if self.is_chat_mode():
                starttime = datetime.now()
                completion = await self.client_wrapper.get_async_client().beta.chat.completions.parse(
                    model=self._model_name, **kwargs
                )
                wall_latency = datetime.now() - starttime
            else:
                raise ValueError("This method doesn't support instruct models. Please use a chat model.")
            response_text = completion.choices[0].message.content
            assert response_text is not None, "OpenAI refused to respond to the query"
            completion_tokens, prompt_tokens = self.validate_tokens(completion)
            self.add_llm_metadata(kwargs, response_text, wall_latency, completion_tokens, prompt_tokens)
            return response_text
        except Exception as e:
            # OpenAI will not respond in two scenarios:
            # 1.) The LLM ran out of output context length(usually do to hallucination of repeating the same phrase)
            # 2.) The LLM refused to respond to the request because it did not meet guidelines
            raise e

    def generate_batch(self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None) -> list[str]:
        llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        cache_hits = [self._llm_cache_get(p, llm_kwargs) for p in prompts]

        calls = []
        for p, ch, i in zip(prompts, cache_hits, range(len(prompts))):
            if ch is not None:
                continue
            kwargs = self._get_generate_kwargs(p, llm_kwargs)
            kwargs["model"] = self.model.name
            call = {"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": kwargs}
            calls.append(call)
        f = io.BytesIO()
        for i, c in enumerate(calls):
            f.write(json.dumps(c).encode("utf-8"))
            if i != len(calls) - 1:
                f.write(b"\n")
        client = self.client_wrapper.get_client()
        starttime = datetime.now()
        batch_in_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_in_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        while batch.status in ("validating", "in_progress", "finalizing"):
            time.sleep(BATCH_POLL_INTERVAL)
            batch = client.batches.retrieve(batch.id)

        wall_latency = datetime.now() - starttime
        if batch.error_file_id:
            errors = client.files.content(batch.error_file_id)
            logging.error(errors.text)
            raise ValueError(f"LLM batch call failed: {batch}")
        if batch.output_file_id:
            responses = client.files.content(batch.output_file_id)
            for rs, call in zip(responses.iter_lines(), calls):
                rdata = json.loads(rs)
                id = int(rdata["custom_id"])
                cc = ChatCompletion.model_construct(**rdata["response"]["body"])
                response_text = cc.choices[0].message.content
                ct, pt = self.validate_tokens(cc)
                kws = call["body"]
                self.add_llm_metadata(kws, response_text, wall_latency, ct, pt)
                cache_hits[id] = response_text
                self._llm_cache_set(prompts[id], llm_kwargs, response_text)
            return cache_hits
        raise ValueError(f"LLM batch call terminated with no output file or error file: {batch}")
