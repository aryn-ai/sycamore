import logging
from typing import Optional, Callable

from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts import RenderedPrompt


logger = logging.getLogger(__name__)


class ChainedLLM(LLM):
    """A ChainedLLM is a special LLM that allows for chaining multiple LLMs together."""

    def __init__(
        self,
        chain: list[LLM],
        response_checker: Optional[Callable[[str], bool]] = None,
        model_name: str = "",
        default_mode: LLMMode = LLMMode.ASYNC,
        cache=None,
        default_llm_kwargs=None,
    ):
        """
        Initializes a ChainedLLM instance.
        Args:
            chain:
            model_name:
            default_mode:
            cache:
            default_llm_kwargs:
        """
        super().__init__(model_name, default_mode, cache, default_llm_kwargs=default_llm_kwargs)

        self._chain: list[LLM] = chain
        self.chat_mode = True
        for llm in self.chain:
            if not llm.is_chat_mode():
                self.chat_mode = False
                break
        self.response_checker = response_checker

    @property
    def chain(self) -> list[LLM]:
        """
        Returns the list of LLMs in the chain.
        Returns:
            The list of LLMs in the chain.
        """
        return self._chain

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        """
        Generates a response by chaining multiple LLMs together.

        Args:
            prompt: The prompt to send to all LLMs in the chain.
            llm_kwargs: Additional keyword arguments for the LLMs.

        Returns:
            The first successful generated response from the last LLM in the chain.
        """

        # The current strategy is to try each LLM in the chain until one succeeds.
        assert self._chain is not None and len(self._chain) > 0, "ChainedLLM must have at least one LLM in the chain."

        last_exception: Exception = RuntimeError("unknown error")
        for llm in self._chain:
            try:
                logger.info(f"Generating response using LLM: {llm._model_name}")
                response = llm.generate(prompt=prompt, llm_kwargs=llm_kwargs)
                if self.response_checker:
                    if self.response_checker(response):
                        return response
                    else:
                        logger.info(
                            f"Response {response} from LLM did not pass the response checker, trying next LLM in the chain."
                        )
                else:
                    return response
            except Exception as e:
                logger.exception(e)
                last_exception = e

        raise last_exception

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters asynchronously."""
        assert self._chain is not None and len(self._chain) > 0, "ChainedLLM must have at least one LLM in the chain."

        last_exception: Exception = RuntimeError("unknown error")
        for llm in self._chain:
            try:
                response = await llm.generate_async(prompt=prompt, llm_kwargs=llm_kwargs)
                return response
            except Exception as e:
                logger.exception(e)
                last_exception = e

        raise last_exception

    def generate_batch(self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None) -> list[str]:
        """Generates a series of responses from the LLM for the given series of prompts. Order is preserved."""
        assert self._chain is not None and len(self._chain) > 0, "ChainedLLM must have at least one LLM in the chain."

        last_exception: Exception = RuntimeError("unknown error")
        for llm in self._chain:
            try:
                response = llm.generate_batch(prompts=prompts, llm_kwargs=llm_kwargs)
                return response
            except Exception as e:
                logger.exception(e)
                last_exception = e

        raise last_exception

    def is_chat_mode(self) -> bool:
        """
        Returns whether the LLM is in chat mode.
        Returns:
            True if all LLMs in the chain are in chat mode, False otherwise.
        """
        return self.chat_mode
