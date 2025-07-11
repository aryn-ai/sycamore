from sycamore.llms import GeminiModels, OpenAIModels
from sycamore.llms.chained_llm import ChainedLLM
from sycamore.llms.gemini import Gemini
from sycamore.llms.openai import OpenAI
from sycamore.llms.prompts import RenderedMessage, RenderedPrompt


def test_chained_llm_generate_defaults():
    gemini = Gemini(GeminiModels.GEMINI_2_FLASH)
    messages = [
        RenderedMessage(
            role="user",
            content="Write a caption for a recent trip to a sunny beach",
        ),
    ]
    prompt = RenderedPrompt(messages=messages)

    openai = OpenAI(OpenAIModels.GPT_3_5_TURBO)

    chained = ChainedLLM([gemini, openai])

    res = chained.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0

    chained_other_way = ChainedLLM([openai, gemini])

    res = chained_other_way.generate(prompt=prompt, llm_kwargs={})

    assert len(res) > 0
