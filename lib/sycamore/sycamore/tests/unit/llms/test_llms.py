from sycamore.llms import OpenAI, OpenAIModels
from sycamore.llms.prompts import EntityExtractorFewShotGuidancePrompt, EntityExtractorZeroShotGuidancePrompt


class TestLLMs:
    def test_openai_davinci_fallback(self):
        llm = OpenAI(model_name=OpenAIModels.TEXT_DAVINCI.value)
        assert llm._model_name == OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value.name

    def test_deprecated_prompt_fallback(self):
        # Import from default_prompts package
        from sycamore.llms.prompts.default_prompts import ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT

        assert isinstance(ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT, EntityExtractorZeroShotGuidancePrompt)

        # Import from prompts package
        from sycamore.llms.prompts import ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT

        assert isinstance(ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT, EntityExtractorFewShotGuidancePrompt)
