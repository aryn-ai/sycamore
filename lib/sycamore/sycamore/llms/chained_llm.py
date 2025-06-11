import logging
import traceback
from typing import Optional

from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts import RenderedPrompt


logger = logging.getLogger(__name__)


class ChainedLLM(LLM):
    """A ChainedLLM is a special LLM that allows for chaining multiple LLMs together."""

    def __init__(self, chain: list[LLM], model_name, default_mode: LLMMode, cache=None, default_llm_kwargs=None):
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
        for llm in self._chain:
            try:
                response = llm.generate(prompt=prompt, llm_kwargs=llm_kwargs)
                return response
            except Exception as e:
                logger.warning(f"Error in LLM {llm._model_name}: {traceback.format_exception(e)}")

        return ""  # If all LLMs fail, return an empty string

    def is_chat_mode(self) -> bool:
        """
        Returns whether the LLM is in chat mode.
        Returns:
            True if all LLMs in the chain are in chat mode, False otherwise.
        """
        return self.chat_mode
