from sycamore.llms import OpenAI, OpenAIModels, OpenAIClientWrapper
from sycamore.llms.openai import OpenAIModel, OpenAIClientType
from sycamore.llms.prompts.default_prompts import SimpleGuidancePrompt

# Note: These tests expect you to have OPENAI_API_KEY set in your environment.


def test_openai_defaults():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt_kwargs = {"prompt": "Write a limerick about large language models."}

    res = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    assert len(res.content) > 0


class TestGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a skilled poet"
    user = "Write a limerick about large language models"


def test_openai_defaults_guidance_chat():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO)
    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    print(res)
    assert len(res) > 0


def test_openai_defaults_guidance_instruct():
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT)
    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0


def test_azure_defaults_guidance_chat():
    llm = OpenAI(
        # Note this deployment name is different from the official model name, which has a '.'
        OpenAIModel("gpt-35-turbo", is_chat=True),
        client_wrapper=OpenAIClientWrapper(
            client_type=OpenAIClientType.AZURE,
            azure_endpoint="https://aryn.openai.azure.com",
            api_version="2024-02-15-preview",
        ),
    )

    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0


def test_azure_defaults_guidance_instruct():
    llm = OpenAI(
        # Note this deployment name is different from the official model name, which has a '.'
        OpenAIModel("gpt-35-turbo-instruct", is_chat=False),
        client_wrapper=OpenAIClientWrapper(
            client_type=OpenAIClientType.AZURE,
            azure_endpoint="https://aryn.openai.azure.com",
            api_version="2024-02-15-preview",
        ),
    )

    prompt_kwargs = {"prompt": TestGuidancePrompt()}
    res = llm.generate(prompt_kwargs=prompt_kwargs)
    assert len(res) > 0
