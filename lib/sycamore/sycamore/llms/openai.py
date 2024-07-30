import logging
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypedDict, Union, cast

from guidance.models import AzureOpenAIChat, AzureOpenAICompletion
from guidance.models import Model
from guidance.models import OpenAI as GuidanceOpenAI
from openai import AzureOpenAI as AzureOpenAIClient
from openai import OpenAI as OpenAIClient
from openai import max_retries as DEFAULT_MAX_RETRIES
from openai.lib.azure import AzureADTokenProvider
from openai.types.chat import ChatCompletionMessageParam

from sycamore.llms.llms import LLM
from sycamore.llms.prompts import GuidancePrompt
from sycamore.utils.cache import Cache

logger = logging.getLogger(__name__)


# Base URL for Helicone API, if configured using the SYCAMORE_HELICONE_API_KEY environment variable.
HELICONE_BASE_URL = "https://oai.helicone.ai/v1"


class OpenAIClientType(Enum):
    OPENAI = 0
    AZURE = 1


@dataclass
class OpenAIModel:
    name: str
    is_chat: bool = False


class OpenAIModels(Enum):
    TEXT_DAVINCI = OpenAIModel(name="text-davinci-003", is_chat=True)
    GPT_3_5_TURBO = OpenAIModel(name="gpt-3.5-turbo", is_chat=True)
    GPT_4_TURBO = OpenAIModel(name="gpt-4-turbo", is_chat=True)
    GPT_4O = OpenAIModel(name="gpt-4o", is_chat=True)
    GPT_4O_MINI = OpenAIModel(name="gpt-4o-mini", is_chat=True)
    GPT_3_5_TURBO_INSTRUCT = OpenAIModel(name="gpt-3.5-turbo-instruct", is_chat=False)

    @classmethod
    def from_name(cls, name: str):
        for m in iter(cls):
            if m.value.name == name:
                return m
        return None


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

        # The OpenAI Python library is happy to pull Azure creds from the AZURE_OPENAI_API_KEY environment variable,
        # but Guidance will error out if neither api_key nor azure_ad_token_provider are explicitly set.
        if client_type == OpenAIClientType.AZURE and api_key is None and azure_ad_token_provider is None:
            if "AZURE_OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "Must set either api_key, azure_ad_token_provider, or AZURE_OPENAI_API_KEY environment variable."
                )

            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

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
                        {"Helicone-Property-Tag": f"{os.environ['SYCAMORE_HELICONE_TAG']}"}
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

    def get_guidance_model(self, model) -> Model:
        if self.client_type == OpenAIClientType.OPENAI:
            return GuidanceOpenAI(
                model=model.name,
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                max_retries=self.max_retries,
                **self.extra_kwargs,
            )
        elif self.client_type == OpenAIClientType.AZURE:
            # Note: Theoretically the Guidance library automatically determines which
            # subclass to use, but this appears to be buggy and relies on a bunch of
            # specific assumptions about how deployed models are named that don't work
            # well with Azure. This is the only way I was able to get it to work
            # reliably.
            # p.s. mypy seems to get mad if cls is not defined as being either, hence
            # the union expression
            if model.is_chat:
                cls: Union[type[AzureOpenAIChat], type[AzureOpenAICompletion]] = AzureOpenAIChat
            else:
                cls = AzureOpenAICompletion

            # Shenanigans to defeat typechecking. AzureOpenAI
            # has params of type str that default to None, so
            # we create a typed dict and use it as a variadic
            # argument.
            class AzureOpenAIParams(TypedDict, total=False):
                model: str
                azure_endpoint: str
                azure_deployment: str
                azure_ad_token_provider: Optional[AzureADTokenProvider]
                api_key: str
                version: str
                azure_ad_token: Optional[str]
                organization: Optional[str]
                max_retries: int

            # azure_endpoint and api_key are not None if we're
            # in this branch, so we can safely cast strings to
            # strings. mypy thing.
            params: AzureOpenAIParams = {
                "model": model.name,
                "azure_endpoint": str(self.azure_endpoint),
                "azure_ad_token_provider": self.azure_ad_token_provider,
                "api_key": str(self.api_key),
                "azure_ad_token": self.azure_ad_token,
                "organization": self.organization,
                "max_retries": self.max_retries,
            }
            # Add these guys in if not None. The defaults are
            # None, but only strings are allowed as params.
            if self.api_version:
                params["version"] = self.api_version
            if self.azure_deployment:
                params["azure_deployment"] = self.azure_deployment
            # Tack on any extra args. need to do this untyped-ly
            cast(dict, params).update(self.extra_kwargs)
            return cls(**params)

        else:
            raise ValueError(f"Invalid client_type {self.client_type}")


# Allow rough backwards compatibility
OpenAIClientParameters = OpenAIClientWrapper


def openai_deserializer(kwargs):
    return OpenAI(**kwargs)


class OpenAI(LLM):
    def __init__(
        self,
        model_name: Union[OpenAIModels, OpenAIModel, str],
        api_key=None,
        client_wrapper: Optional[OpenAIClientWrapper] = None,
        params: Optional[OpenAIClientParameters] = None,
        cache: Optional[Cache] = None,
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
            logger.warn("text-davinci-003 is deprecated. Falling back to gpt-3.5-turbo-instruct")
            self.model = OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value
        super().__init__(self.model.name, cache)

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
        self._client = self.client_wrapper.get_client()

    # The actual openai client is not pickleable, This just says to pickle the wrapper, which can be used to
    # recreate the client on the other end.
    def __reduce__(self):

        kwargs = {"client_wrapper": self.client_wrapper, "model_name": self._model_name, "cache": self._cache}

        return openai_deserializer, (kwargs,)

    def is_chat_mode(self):
        return self.model.is_chat

    def _get_cache_key(self, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> str:
        assert self._cache
        combined = {"prompt_kwargs": prompt_kwargs, "llm_kwargs": llm_kwargs, "model_name": self.model.name}
        data = pickle.dumps(combined)
        return self._cache.get_hash_context(data).hexdigest()

    def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None) -> Any:
        cache_key = None
        if self._cache:
            cache_key = self._get_cache_key(prompt_kwargs, llm_kwargs)
            hit = self._cache.get(cache_key)
            if hit:
                if (
                    hit.get("prompt_kwargs") == prompt_kwargs
                    and hit.get("llm_kwargs") == llm_kwargs
                    and hit.get("model_name") == self.model.name
                ):
                    return hit.get("result")
                else:
                    logger.warning(
                        "Found cache content mismatch, key=%s prompt_kwargs=%s llm_kwargs=%s model_name=%s",
                        cache_key,
                        prompt_kwargs,
                        llm_kwargs,
                        self.model.name,
                    )

        if llm_kwargs is not None:
            result = self._generate_using_openai(prompt_kwargs, llm_kwargs)
        else:
            result = self._generate_using_guidance(prompt_kwargs)

        if self._cache:
            assert cache_key
            item = {
                "result": result,
                "prompt_kwargs": prompt_kwargs,
                "llm_kwargs": llm_kwargs,
                "model_name": self.model.name,
            }
            self._cache.set(cache_key, item)
        return result

    def _generate_using_openai(self, prompt_kwargs, llm_kwargs) -> str:
        kwargs = {
            "temperature": 0,
            **llm_kwargs,
        }

        if "SYCAMORE_OPENAI_USER" in os.environ:
            kwargs.update({"user": os.environ.get("SYCAMORE_OPENAI_USER")})

        if "prompt" in prompt_kwargs:
            prompt = prompt_kwargs.get("prompt")
            messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": f"{prompt}"}]
        elif "messages" in prompt_kwargs:
            messages = prompt_kwargs["messages"]
        else:
            raise ValueError("Either prompt or messages must be present in prompt_kwargs.")

        completion = self._client.chat.completions.create(model=self._model_name, messages=messages, **kwargs)
        return completion.choices[0].message.content

    def _generate_using_guidance(self, prompt_kwargs) -> str:
        guidance_model = self.client_wrapper.get_guidance_model(self.model)
        prompt: GuidancePrompt = prompt_kwargs.pop("prompt")
        prediction = prompt.execute(guidance_model, **prompt_kwargs)
        return prediction
