from typing import Optional, Any
import json
from sycamore.data.document import Document
from sycamore.data.element import Element, ImageElement
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.tests.config import TEST_DIR
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.transforms.summarize_images import LLMImageSummarizer, SummarizeImages


def image_element() -> ImageElement:
    with open(TEST_DIR / "resources/data/imgs/sample-detr-image.png", "rb") as f:
        return ImageElement(binary_representation=f.read(), image_format="png")


class MockLLM(LLM):
    def __init__(self):
        super().__init__(model_name="dummy", default_mode=LLMMode.SYNC)

    def is_chat_mode(self):
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict[str, Any]] = None) -> str:
        promptstr = "\n".join(m.content for m in prompt.messages)
        return json.dumps({"summary": promptstr})


class TestSummarizeImages:
    @staticmethod
    def doc():
        return Document(
            elements=[
                image_element(),
                Element(type="Text", text_representation="text"),
                image_element(),
                Element(type="Section-header", text_representation="section-header"),
                image_element(),
                Element(type="Caption", text_representation="caption"),
                image_element(),
            ]
        )

    def test_summarize_images(self, mocker):
        d = self.doc()
        sum_images = LLMImageSummarizer(llm=MockLLM())
        si_transform = SummarizeImages(None, summarizer=sum_images)
        out = si_transform._local_process([d])[0]

        assert "The text preceding the image is: " not in out.elements[0].properties["summary"]["summary"]
        assert "The text following the image is: text" in out.elements[0].properties["summary"]["summary"]

        assert "The text preceding the image is: text" in out.elements[2].properties["summary"]["summary"]
        assert "The text following the image is: " not in out.elements[2].properties["summary"]["summary"]

        assert "The text preceding the image is: section-header" in out.elements[4].properties["summary"]["summary"]
        assert "The text following the image is: caption" in out.elements[4].properties["summary"]["summary"]

        assert "The text preceding the image is: caption" in out.elements[6].properties["summary"]["summary"]
        assert "The text following the image is: " not in out.elements[6].properties["summary"]["summary"]
