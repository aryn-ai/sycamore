from sycamore.llms import GeminiModels, OpenAIModels
from sycamore.llms.chained_llm import ChainedLLM
from sycamore.llms.gemini import Gemini
from sycamore.llms.llms import LLMMode
from sycamore.llms.openai import OpenAI
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.tests.utils import setup_debug_logging

setup_debug_logging("sycamore.llms.chained_llm")
setup_debug_logging("sycamore.llms.gemini")
setup_debug_logging("sycamore.llms.openai")


def test_default_llm_kwargs():
    gemini = Gemini(GeminiModels.GEMINI_2_FLASH_LITE, default_llm_kwargs={"max_output_tokens": 5})
    openai = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    chained = ChainedLLM(chain=[gemini, openai], model_name="", default_mode=LLMMode.SYNC)
    res = chained.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0


def test_model_override():
    gemini = Gemini(GeminiModels.GEMINI_2_FLASH_LITE, default_llm_kwargs={"max_output_tokens": 5})
    openai = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt = RenderedPrompt(
        messages=[RenderedMessage(role="user", content="Write a limerick about large language models.")]
    )

    chained = ChainedLLM(chain=[gemini, openai], model_name="", default_mode=LLMMode.SYNC)

    models = {0: GeminiModels.GEMINI_2_FLASH.value.name, 1: OpenAIModels.GPT_4O.value.name}
    res = chained.generate(prompt=prompt, llm_kwargs={"models": models})

    assert len(res) > 0

    chained = ChainedLLM(chain=[openai, gemini], model_name="", default_mode=LLMMode.SYNC)

    models = {1: GeminiModels.GEMINI_2_FLASH.value.name, 0: OpenAIModels.GPT_4O.value.name}
    res = chained.generate(prompt=prompt, llm_kwargs={"models": models})

    assert len(res) > 0
