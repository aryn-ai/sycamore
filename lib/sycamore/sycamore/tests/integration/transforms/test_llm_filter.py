import sycamore
from sycamore.tests.config import TEST_DIR
from sycamore.llms.prompts.default_prompts import LlmFilterMessagesJinjaPrompt
from sycamore.llms.openai import OpenAI, OpenAIModels


def test_llm_filter_ntsb_temp_q():
    ctx = sycamore.init()
    ds = ctx.read.materialize(TEST_DIR / "resources/data/materialize/llmfilter-ntsb-temp")
    llm = OpenAI(OpenAIModels.GPT_4O_MINI)
    prompt = LlmFilterMessagesJinjaPrompt.fork(filter_question="Is the temperature less than 60F?")
    ds = ds.llm_filter(
        llm=llm,
        new_field="_autogen_LLMFilterOutput",
        prompt=prompt,
        use_elements=True,
    )
    assert ds.count() == 0
