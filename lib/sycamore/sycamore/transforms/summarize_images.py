from typing import Optional


from sycamore.data import Document, Element
from sycamore.llms.openai import LLM, OpenAI, OpenAIClientWrapper, OpenAIModels
from sycamore.llms.prompts.default_prompts import SummarizeImagesJinjaPrompt
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.base_llm import LLMMapElements
from sycamore.transforms.map import Map
from sycamore.utils.extract_json import extract_json


def parse_summary_json(e: Element) -> Element:
    if "summary" in e.properties and isinstance(e.properties["summary"], str):
        e.properties["summary"] = extract_json(e.properties["summary"])
    return e


def _parse_summary_json_on_all_elts(d: Document) -> Document:
    d.elements = [parse_summary_json(e) for e in d.elements]
    return d


class LLMImageSummarizer:
    """Image Summarizer that uses an LLM to summarize the specified image.

    The image is passed to the LLM along with a text prompt and optionally the text elements
    immediately preceding and following the image.

    Args:
       llm: The LLM to use.
       prompt: The prompt to use to pass to the model, as a string.
       include_context: Whether to include the immediately preceding and following text elements as context.

    Example:
         The following code demonstrates how to partition a pdf DocSet and summarize the images it contains.
         This version uses a Claude model via Bedrock.

         .. code-block:: python
            llm = Bedrock(BedrockModels.CLAUDE_3_5_SONNET)

            context = sycamore.init()
            doc = context.read.binary(paths=paths, binary_format="pdf")\
                              .partition(partitioner=SycamorePartitioner(extract_images=True))\
                              .transform(SummarizeImages, summarizer=LLMImageSummarizer(llm=llm))\
                              .show()
    """

    DEFAULT_PROMPT = """You are given an image from a PDF document along with with some snippets of text preceding
            and following the image on the page. Based on this context, please decide whether the image is a
            graph or not. An image is a graph if it is a bar chart or a line graph. If the image is a graph,
            please summarize the axes, including their units, and provide a summary of the results in no more
            than 5 sentences.

            Return the results in the following JSON schema:

            {
              "is_graph": true,
              "x-axis": string,
              "y-axis": string,
              "summary": string
            }

            If the image is not a graph, please summarize the contents of the image in no more than five sentences
            in the following JSON format:

            {
              "is_graph": false,
              "summary": string
            }

            In all cases return only JSON and check your work.
            """

    def __init__(self, llm: LLM, prompt: Optional[str] = None, include_context: bool = True):
        self.llm = llm
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        self.prompt = prompt
        self.include_context = include_context


class OpenAIImageSummarizer(LLMImageSummarizer):
    """Implementation of the LLMImageSummarizer for OpenAI models.

    Args:
       openai_model: The OpenAI instance to use. If not set, one will be created.
       client_wrapper: The OpenAIClientWrapper to use when creating an OpenAI instance.
           Not used if openai_model is set.
       prompt: The prompt to use to pass to the model, as a string.
       include_context: Whether to include the immediately preceding and following text elements as context.
    """

    model = OpenAIModels.GPT_4O

    def __init__(
        self,
        openai_model: Optional[OpenAI] = None,
        client_wrapper: Optional[OpenAIClientWrapper] = None,
        prompt: Optional[str] = None,
        include_context: bool = True,
    ):
        if openai_model is not None:
            openai = openai_model
        else:
            openai = OpenAI(model_name=self.model, client_wrapper=client_wrapper)

        super().__init__(llm=openai, prompt=prompt, include_context=include_context)


class SummarizeImages(CompositeTransform):
    """SummarizeImages is a transform for summarizing context into text using an LLM.

    Args:
       child: The source node for the transform.
       summarizer: The class to use for summarization. The default uses OpenAI gpt-4-turbo.
       resource_args: Additional resource-related arguments that can be passed to the underlying runtime.

    Example:
         .. code-block:: python

            context = sycamore.init()
            doc = context.read.binary(paths=paths, binary_format="pdf")\
                              .partition(partitioner=SycamorePartitioner(extract_images=True))\
                              .transform(SummarizeImages)\
                              .show()
    """

    def __init__(self, child: Node, summarizer=OpenAIImageSummarizer(), **resource_args):
        super().__init__(child, [], **resource_args)
        prompt: SycamorePrompt = SummarizeImagesJinjaPrompt
        if summarizer.prompt != LLMImageSummarizer.DEFAULT_PROMPT:
            prompt = prompt.set(user=summarizer.prompt)
        prompt = prompt.set(include_context=summarizer.include_context)
        llm_map = LLMMapElements(
            child, prompt, output_field="summary", llm=summarizer.llm, filter=lambda e: e.type == "Image"
        )
        parse_summary = Map(llm_map, f=_parse_summary_json_on_all_elts)
        self.nodes = [llm_map, parse_summary]
        self.summarizer = summarizer
