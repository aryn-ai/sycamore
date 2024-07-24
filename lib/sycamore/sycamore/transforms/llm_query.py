from abc import ABC, abstractmethod
from typing import Optional

from sycamore.data import Element, Document
from sycamore.llms.openai import OpenAIModels
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import OpenAI, OpenAIClientWrapper
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace


class LLMQueryAgent(ABC):
    @abstractmethod
    def execute_query(self, document: Document) -> Document:
        pass


class LLMTextQueryAgent(LLMQueryAgent):
    """
    LLMTextQueryAgent uses a specified LLM to execute LLM queries about a document

    Args:
        prompt: A prompt to be passed into the underlying LLM execution engine
        openai_model: (Optional) The type of OpenAI model to be used in the execution. Defaults to GPT-4O
        client_wrapper: (Optional) Specifications for the OpenAI client wrapper. Ignored if openai_model is specified
        output_property: (Optional) The output property to add results in. Defaults to 'llm_response'
        llm_kwargs: (Optional) LLM keyword argument for the underlying execution engine
        per_element: (Optional) Whether to execute the call per each element or on the Document itself. Defaults to True.

    Example:
         .. code-block:: python

            prompt="Tell me the important numbers from this element"
            llm_query_agent = LLMElementTextSummarizer(prompt=prompt)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .llm_query(query_agent=llm_query_agent)
    """

    model = OpenAIModels.GPT_4O

    def __init__(
        self,
        prompt: str,
        openai_model: Optional[OpenAI] = None,
        client_wrapper: Optional[OpenAIClientWrapper] = None,
        output_property: str = "llm_response",
        llm_kwargs: dict = {},
        per_element: bool = True,
    ):
        if openai_model is not None:
            self._openai = openai_model
        else:
            self._openai = OpenAI(model_name=self.model, client_wrapper=client_wrapper, max_retries=5)
        self._prompt = prompt
        self._output_property = output_property
        self._llm_kwargs = llm_kwargs
        self._per_element = per_element

    def execute_query(self, document: Document) -> Document:
        if self._per_element:
            elements = []
            for element in document.elements:
                elements.append(self._summarize_text_element(element))
            document.elements = elements
        else:
            if document.text_representation:
                prompt = self._prompt + "\n" + document["text_representation"]
                prompt_kwargs = {"prompt": prompt}
                llm_resp = self._openai.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
                document["properties"][self._output_property] = llm_resp
        return document

    @timetrace("LLMQueryText")
    def _summarize_text_element(self, element: Element) -> Element:
        if element.text_representation:
            prompt = self._prompt
            prompt += "\n" + element["text_representation"]
            prompt_kwargs = {"prompt": prompt}
            llm_resp = self._openai.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
            element["properties"][self._output_property] = llm_resp
        return element


class LLMQuery(NonCPUUser, NonGPUUser, Map):
    """
    The LLM Query Transform executes user defined queries on a document or the elements within it.
    """

    def __init__(self, child: Node, query_agent: LLMQueryAgent, **kwargs):
        super().__init__(child, f=query_agent.execute_query, **kwargs)
