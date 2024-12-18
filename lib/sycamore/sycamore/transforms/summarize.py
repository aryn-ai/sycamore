from abc import ABC, abstractmethod
from typing import Callable, Optional


from sycamore.data import Element, Document
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.llms.prompts import TextSummarizerGuidancePrompt
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, document: Document) -> Document:
        pass


class LLMElementTextSummarizer(Summarizer):
    """
    LLMElementTextSummarizer uses a specified LLM to summarize text data within elements of a document.

    Args:
        llm: An instance of an LLM class to use for text summarization.
        element_operator: A callable function that operates on the document and returns a list of elements to be
            summarized. Default is None.

    Example:
         .. code-block:: python

            llm_model = OpenAILanguageModel("gpt-3.5-turbo")
            element_operator = my_element_selector  # A custom element selection function
            summarizer = LLMElementTextSummarizer(llm_model, element_operator)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .summarize(summarizer=summarizer)
    """

    def __init__(self, llm: LLM, element_operator: Optional[Callable[[Element], bool]] = None):
        self._llm = llm
        self._element_operator = element_operator

    def summarize(self, document: Document) -> Document:
        elements = []
        if self._element_operator is not None:
            for element in document.elements:
                if self._element_operator(element):
                    elements.append(self._summarize_text_element(element))
                else:
                    elements.append(element)
        else:
            elements = [self._summarize_text_element(element) for element in document.elements]

        document.elements = elements
        return document

    @timetrace("SummText")
    def _summarize_text_element(self, element: Element) -> Element:
        prompt = TextSummarizerGuidancePrompt()

        if element.text_representation:
            response = self._llm.generate(prompt_kwargs={"prompt": prompt, "query": element.text_representation})
            element.properties["summary"] = response
        return element


class Summarize(NonCPUUser, NonGPUUser, Map):
    """
    The summarize transform generates summaries of documents or elements.
    """

    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, f=summarizer.summarize, **kwargs)
