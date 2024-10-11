from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.data import Document, Element
from sycamore.llms.prompts.default_prompts import EntityExtractorFewShotGuidancePrompt


class FakeLLM:
    def __init__(self):
        pass

    def generate(self, prompt_kwargs):
        assert isinstance(prompt_kwargs["prompt"], EntityExtractorFewShotGuidancePrompt)
        assert prompt_kwargs["entity"] == "title"
        assert prompt_kwargs["query"] == "ELEMENT 1: ExampleTitle\nELEMENT 2: Footer\n"
        assert prompt_kwargs["examples"] == "An Example Template"
        return "ExampleTitle"


def test_extract_with_template():
    extractor = OpenAIEntityExtractor("title", llm=FakeLLM(), prompt_template="An Example Template")
    d = Document(
        elements=[
            Element(type="title", text_representation="ExampleTitle"),
            Element(type="footer", text_representation="Footer"),
        ]
    )

    extractor.extract_entity(d)

    assert d.properties["title"] == "ExampleTitle"
