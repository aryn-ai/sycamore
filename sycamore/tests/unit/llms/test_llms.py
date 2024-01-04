from sycamore.llms import OpenAI, OpenAIModels


class TestLLMs:
    def test_openai_davinci_fallback(self):
        llm = OpenAI(model_name=OpenAIModels.TEXT_DAVINCI.value)
        assert llm._model_name == OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value
