import logging
from typing import Optional

from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts import RenderedPrompt


logger = logging.getLogger(__name__)


class ChainedLLM(LLM):
    """ A ChainedLLM is a special LLM that allows for chaining multiple LLMs together."""

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

        self.chain: list[LLM] = chain

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
        for llm in self.chain:
            try:
                response = llm.generate(prompt=prompt, llm_kwargs=llm_kwargs)
                return response
            except Exception as e:
                logger.warning(f"Error in LLM {llm.model_name}: {e}")

        return ""  # If all LLMs fail, return an empty string

    def is_chat_mode(self) -> bool:
        pass

